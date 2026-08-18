from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from app.damage_taxonomy import CLASS_TO_INDEX, DAMAGE_CLASSES


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    filename: str
    clean_source_sha256: str
    face_bbox_normalized: tuple[float, float, float, float]


@dataclass(frozen=True)
class DamageSample:
    image: np.ndarray
    mask: np.ndarray
    damage_class: str
    seed: int
    source_id: str


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_manifest(path: Path) -> list[SourceRecord]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    records: list[SourceRecord] = []
    for row in payload.get('sources', []):
        bbox = row.get('face_bbox_normalized')
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f'Invalid face bbox for {row.get("source_id")}')
        records.append(SourceRecord(
            source_id=str(row['source_id']),
            filename=str(row['filename']),
            clean_source_sha256=str(row['clean_source_sha256']).lower(),
            face_bbox_normalized=tuple(float(v) for v in bbox),
        ))
    if not records:
        raise ValueError('Source manifest is empty')
    return records


def load_face_crop(record: SourceRecord, source_dir: Path, *, size: int = 256) -> np.ndarray:
    path = source_dir / record.filename
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_path(path)
    if actual.lower() != record.clean_source_sha256:
        raise RuntimeError(f'Hash mismatch for {record.source_id}: {actual}')
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Cannot decode {path}')
    h, w = image.shape[:2]
    x1n, y1n, x2n, y2n = record.face_bbox_normalized
    x1 = max(0, min(w - 1, int(math.floor(x1n * w))))
    y1 = max(0, min(h - 1, int(math.floor(y1n * h))))
    x2 = max(x1 + 1, min(w, int(math.ceil(x2n * w))))
    y2 = max(y1 + 1, min(h, int(math.ceil(y2n * h))))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise RuntimeError(f'Empty crop for {record.source_id}')
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _random_region(rng: random.Random, size: int, *, central: bool = False) -> tuple[int, int, int, int]:
    if central:
        cx = rng.uniform(0.35, 0.65) * size
        cy = rng.uniform(0.28, 0.72) * size
    else:
        cx = rng.uniform(0.18, 0.82) * size
        cy = rng.uniform(0.18, 0.82) * size
    rw = rng.uniform(0.16, 0.46) * size
    rh = rng.uniform(0.10, 0.34) * size
    x1 = max(0, int(round(cx - rw / 2)))
    y1 = max(0, int(round(cy - rh / 2)))
    x2 = min(size, max(x1 + 2, int(round(cx + rw / 2))))
    y2 = min(size, max(y1 + 2, int(round(cy + rh / 2))))
    return x1, y1, x2, y2


def _region_mask(size: int, rect: tuple[int, int, int, int], *, ellipse: bool = False) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    x1, y1, x2, y2 = rect
    if ellipse:
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    return mask


def _composite(base: np.ndarray, changed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = base.copy()
    selected = mask > 0
    result[selected] = changed[selected]
    return result


def _motion_blur(image: np.ndarray, length: int, angle_degrees: float) -> np.ndarray:
    k = max(3, int(length) | 1)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    center = (k / 2 - 0.5, k / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (k, k), flags=cv2.INTER_LINEAR)
    total = float(kernel.sum())
    if total <= 0:
        kernel[k // 2, :] = 1.0
        total = float(kernel.sum())
    kernel /= total
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _pixelate_patch(image: np.ndarray, rect: tuple[int, int, int, int], blocks: int) -> np.ndarray:
    x1, y1, x2, y2 = rect
    result = image.copy()
    patch = image[y1:y2, x1:x2]
    ph, pw = patch.shape[:2]
    low_w = max(1, min(pw, int(blocks)))
    low_h = max(1, min(ph, int(round(blocks * ph / max(pw, 1)))))
    low = cv2.resize(patch, (low_w, low_h), interpolation=cv2.INTER_AREA)
    result[y1:y2, x1:x2] = cv2.resize(low, (pw, ph), interpolation=cv2.INTER_NEAREST)
    return result


def _jpeg_patch(image: np.ndarray, rect: tuple[int, int, int, int], quality: int) -> np.ndarray:
    x1, y1, x2, y2 = rect
    result = image.copy()
    patch = image[y1:y2, x1:x2]
    ok, encoded = cv2.imencode('.jpg', patch, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError('Synthetic JPEG encoding failed')
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError('Synthetic JPEG decoding failed')
    if decoded.shape != patch.shape:
        decoded = cv2.resize(decoded, (patch.shape[1], patch.shape[0]), interpolation=cv2.INTER_LINEAR)
    result[y1:y2, x1:x2] = decoded
    return result


def apply_exact_damage(face_bgr: np.ndarray, damage_class: str, seed: int) -> DamageSample:
    if damage_class not in CLASS_TO_INDEX or damage_class == 'HEALTHY':
        raise ValueError(f'Unsupported synthetic damage class: {damage_class}')
    if face_bgr.ndim != 3 or face_bgr.shape[2] != 3 or face_bgr.dtype != np.uint8:
        raise ValueError('face_bgr must be uint8 HxWx3')
    if face_bgr.shape[0] != face_bgr.shape[1]:
        raise ValueError('Synthetic generator expects square face crop')

    size = int(face_bgr.shape[0])
    rng = random.Random(int(seed))
    rect = _random_region(rng, size, central=damage_class in {'MISSING_COMPONENT', 'BLACK_BAR'})
    mask = _region_mask(size, rect, ellipse=damage_class in {'BLUR', 'MOTION_BLUR', 'MISSING_COMPONENT'})
    result = face_bgr.copy()

    if damage_class == 'BLUR':
        sigma = rng.uniform(2.0, 6.0)
        changed = cv2.GaussianBlur(face_bgr, (0, 0), sigma)
        result = _composite(face_bgr, changed, mask)
    elif damage_class == 'MOTION_BLUR':
        changed = _motion_blur(face_bgr, rng.randint(9, 27), rng.uniform(-70.0, 70.0))
        result = _composite(face_bgr, changed, mask)
    elif damage_class == 'PIXELATION':
        result = _pixelate_patch(face_bgr, rect, rng.randint(12, 26))
    elif damage_class == 'BLOCK_MOSAIC':
        result = _pixelate_patch(face_bgr, rect, rng.randint(3, 9))
    elif damage_class == 'JPEG_ARTIFACT':
        result = _jpeg_patch(face_bgr, rect, rng.randint(5, 28))
    elif damage_class == 'SCRIBBLE':
        mask = np.zeros((size, size), dtype=np.uint8)
        result = face_bgr.copy()
        points = []
        for _ in range(rng.randint(4, 9)):
            points.append((rng.randint(int(size * 0.15), int(size * 0.85)), rng.randint(int(size * 0.15), int(size * 0.85))))
        thickness = rng.randint(max(3, size // 45), max(6, size // 20))
        colour = tuple(rng.randint(0, 255) for _ in range(3))
        cv2.polylines(mask, [np.asarray(points, dtype=np.int32)], False, 255, thickness, cv2.LINE_AA)
        cv2.polylines(result, [np.asarray(points, dtype=np.int32)], False, colour, thickness, cv2.LINE_AA)
    elif damage_class == 'STICKER':
        ellipse = bool(rng.getrandbits(1))
        mask = _region_mask(size, rect, ellipse=ellipse)
        colour = np.full_like(face_bgr, tuple(rng.randint(30, 245) for _ in range(3)))
        result = _composite(face_bgr, colour, mask)
    elif damage_class == 'OPAQUE_BLOCK':
        mask = _region_mask(size, rect, ellipse=False)
        colour = np.full_like(face_bgr, tuple(rng.randint(0, 255) for _ in range(3)))
        result = _composite(face_bgr, colour, mask)
    elif damage_class == 'BLACK_BAR':
        x1, _, x2, _ = rect
        y_center = rng.randint(int(size * 0.25), int(size * 0.68))
        bar_h = rng.randint(max(6, size // 18), max(12, size // 8))
        rect = (x1, max(0, y_center - bar_h // 2), x2, min(size, y_center + bar_h // 2))
        mask = _region_mask(size, rect)
        result[mask > 0] = 0
    elif damage_class == 'PARTIAL_OCCLUSION':
        mask = np.zeros((size, size), dtype=np.uint8)
        center_x = rng.randint(int(size * 0.25), int(size * 0.75))
        center_y = rng.randint(int(size * 0.25), int(size * 0.75))
        radius = rng.randint(int(size * 0.08), int(size * 0.20))
        vertices = []
        for index in range(rng.randint(5, 8)):
            angle = (2 * math.pi * index / 7.0) + rng.uniform(-0.25, 0.25)
            local = radius * rng.uniform(0.65, 1.25)
            vertices.append((int(center_x + math.cos(angle) * local), int(center_y + math.sin(angle) * local)))
        polygon = np.asarray(vertices, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255, cv2.LINE_AA)
        overlay = np.full_like(face_bgr, tuple(rng.randint(15, 235) for _ in range(3)))
        alpha = rng.uniform(0.55, 0.90)
        changed = cv2.addWeighted(face_bgr, 1.0 - alpha, overlay, alpha, 0)
        result = _composite(face_bgr, changed, mask)
    elif damage_class == 'MISSING_COMPONENT':
        zones = (
            (0.16, 0.25, 0.48, 0.43),
            (0.52, 0.25, 0.84, 0.43),
            (0.34, 0.38, 0.66, 0.67),
            (0.25, 0.61, 0.75, 0.80),
        )
        zx1, zy1, zx2, zy2 = rng.choice(zones)
        rect = (int(zx1 * size), int(zy1 * size), int(zx2 * size), int(zy2 * size))
        mask = _region_mask(size, rect, ellipse=True)
        neutral = cv2.GaussianBlur(face_bgr, (0, 0), max(6.0, size / 18.0))
        neutral = cv2.addWeighted(neutral, 0.25, np.full_like(neutral, int(rng.uniform(80, 180))), 0.75, 0)
        result = _composite(face_bgr, neutral, mask)

    label = np.zeros((size, size), dtype=np.uint8)
    label[mask > 0] = np.uint8(CLASS_TO_INDEX[damage_class])
    return DamageSample(
        image=result,
        mask=label,
        damage_class=damage_class,
        seed=int(seed),
        source_id='',
    )


def iter_balanced_samples(
    face: np.ndarray,
    source_id: str,
    *,
    samples_per_damage_class: int,
    base_seed: int,
) -> Iterable[DamageSample]:
    for class_index, name in enumerate(DAMAGE_CLASSES[1:], start=1):
        for repetition in range(int(samples_per_damage_class)):
            seed = int(base_seed + class_index * 10007 + repetition * 997)
            sample = apply_exact_damage(face, name, seed)
            yield DamageSample(
                image=sample.image,
                mask=sample.mask,
                damage_class=name,
                seed=seed,
                source_id=source_id,
            )
