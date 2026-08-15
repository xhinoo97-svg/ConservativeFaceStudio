from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import cv2
import numpy as np


Kind = Literal[
    "gaussian_blur", "motion_blur", "defocus_blur", "anisotropic_blur", "resize_blur",
    "pixelation", "jpeg", "noise", "low_light", "marker_strokes",
    "scribble", "opaque_paint", "opaque_sticker", "blur_rectangle",
    "smartphone_mixed",
]


@dataclass(frozen=True)
class DegradationRecord:
    kind: str
    severity: int
    seed: int
    damaged_pixels: int
    face_target_fraction: float
    abstention_expected: bool
    parameters: dict[str, float | int | str]


def _validated(image: np.ndarray, face_mask: np.ndarray, severity: int) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be uint8 BGR")
    if face_mask.shape != image.shape[:2]:
        raise ValueError("face_mask shape mismatch")
    if severity not in range(1, 6):
        raise ValueError("severity must be 1..5")
    mask = np.where(face_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        raise ValueError("face_mask is empty")
    return image.copy(), mask


def _face_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _region_mask(mask: np.ndarray, rng: np.random.Generator, severity: int, *, irregular: bool) -> np.ndarray:
    x0, y0, x1, y1 = _face_box(mask)
    h, w = y1 - y0, x1 - x0
    scale = (0.18, 0.30, 0.45, 0.68, 0.88)[severity - 1]
    rw, rh = max(3, int(w * scale)), max(3, int(h * scale))
    cx = int(rng.integers(x0 + rw // 2, max(x0 + rw // 2 + 1, x1 - rw // 2)))
    cy = int(rng.integers(y0 + rh // 2, max(y0 + rh // 2 + 1, y1 - rh // 2)))
    out = np.zeros_like(mask)
    if irregular:
        points = []
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            jitter = float(rng.uniform(0.65, 1.15))
            points.append((int(cx + np.cos(angle) * rw * 0.5 * jitter), int(cy + np.sin(angle) * rh * 0.5 * jitter)))
        cv2.fillPoly(out, [np.asarray(points, dtype=np.int32)], 255)
    else:
        cv2.rectangle(out, (cx - rw // 2, cy - rh // 2), (cx + rw // 2, cy + rh // 2), 255, -1)
    return cv2.bitwise_and(out, mask)


def _motion_kernel(size: int, angle: float) -> np.ndarray:
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    matrix = cv2.getRotationMatrix2D((size / 2 - 0.5, size / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (size, size))
    return kernel / max(float(kernel.sum()), 1e-6)


def _masked_replace(base: np.ndarray, altered: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = base.copy()
    active = mask > 0
    out[active] = altered[active]
    return out


def apply_degradation(
    image: np.ndarray,
    face_mask: np.ndarray,
    *,
    kind: Kind,
    severity: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, DegradationRecord]:
    """Apply one deterministic synthetic corruption inside the supplied face domain."""
    clean, face = _validated(image, face_mask, severity)
    rng = np.random.default_rng(seed)
    local = kind in {"marker_strokes", "scribble", "opaque_paint", "opaque_sticker", "blur_rectangle"}
    damage = _region_mask(face, rng, severity, irregular=kind in {"marker_strokes", "scribble", "opaque_paint"}) if local else face.copy()
    params: dict[str, float | int | str] = {}

    if kind == "gaussian_blur":
        sigma = (1.2, 2.5, 5.0, 9.0, 14.0)[severity - 1]
        altered = cv2.GaussianBlur(clean, (0, 0), sigma)
        params["sigma"] = sigma
    elif kind == "motion_blur":
        size = (5, 9, 17, 29, 41)[severity - 1] | 1
        angle = float(rng.uniform(-80, 80))
        altered = cv2.filter2D(clean, -1, _motion_kernel(size, angle))
        params.update(kernel=size, angle=angle)
    elif kind == "defocus_blur":
        radius = (2, 4, 7, 11, 15)[severity - 1]
        kernel = np.zeros((radius * 2 + 1, radius * 2 + 1), np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1, -1)
        kernel /= kernel.sum()
        altered = cv2.filter2D(clean, -1, kernel)
        params["radius"] = radius
    elif kind == "anisotropic_blur":
        sigma_x = (1.5, 3.0, 5.5, 9.0, 13.0)[severity - 1]
        sigma_y = max(0.6, sigma_x * float(rng.uniform(0.18, 0.48)))
        angle = float(rng.uniform(-90.0, 90.0))
        radius = max(2, int(np.ceil(3.0 * sigma_x)))
        axis = np.arange(-radius, radius + 1, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)
        radians = np.deg2rad(angle)
        rotated_x = xx * np.cos(radians) + yy * np.sin(radians)
        rotated_y = -xx * np.sin(radians) + yy * np.cos(radians)
        kernel = np.exp(-0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2)).astype(np.float32)
        kernel /= max(float(kernel.sum()), 1e-6)
        altered = cv2.filter2D(clean, -1, kernel)
        params.update(sigma_x=sigma_x, sigma_y=sigma_y, angle=angle)
    elif kind == "resize_blur":
        factor = (0.75, 0.50, 0.30, 0.18, 0.10)[severity - 1]
        h, w = clean.shape[:2]
        small = cv2.resize(clean, (max(2, int(w * factor)), max(2, int(h * factor))), interpolation=cv2.INTER_AREA)
        altered = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        params["scale"] = factor
    elif kind == "pixelation":
        block = (4, 7, 12, 20, 30)[severity - 1]
        h, w = clean.shape[:2]
        small = cv2.resize(clean, (max(2, w // block), max(2, h // block)), interpolation=cv2.INTER_AREA)
        altered = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        params["block"] = block
    elif kind == "jpeg":
        quality = (65, 45, 28, 14, 7)[severity - 1]
        ok, encoded = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        altered = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        params["quality"] = quality
    elif kind == "noise":
        sigma = (4, 8, 16, 28, 40)[severity - 1]
        altered = np.clip(clean.astype(np.float32) + rng.normal(0, sigma, clean.shape), 0, 255).astype(np.uint8)
        params["sigma"] = sigma
    elif kind == "low_light":
        gain = (0.75, 0.58, 0.40, 0.25, 0.15)[severity - 1]
        altered = np.clip(clean.astype(np.float32) * gain + rng.normal(0, 4 + severity * 2, clean.shape), 0, 255).astype(np.uint8)
        params["gain"] = gain
    elif kind in {"marker_strokes", "scribble"}:
        altered = clean.copy()
        x0, y0, x1, y1 = _face_box(damage)
        count = 2 + severity * 2
        width = max(2, int((x1 - x0) * (0.025 + severity * 0.015)))
        stroke_mask = np.zeros_like(face)
        for _ in range(count):
            pts = np.column_stack((rng.integers(x0, max(x0 + 1, x1), 5), rng.integers(y0, max(y0 + 1, y1), 5))).astype(np.int32)
            cv2.polylines(stroke_mask, [pts], False, 255, width, cv2.LINE_AA)
        damage = cv2.bitwise_and(stroke_mask, damage)
        altered[damage > 0] = (3, 3, 3)
        params.update(strokes=count, width=width)
    elif kind in {"opaque_paint", "opaque_sticker"}:
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        altered = clean.copy()
        altered[damage > 0] = color
        params["color_bgr"] = str(color)
    elif kind == "blur_rectangle":
        sigma = (2.0, 4.0, 7.0, 11.0, 16.0)[severity - 1]
        altered = cv2.GaussianBlur(clean, (0, 0), sigma)
        params["sigma"] = sigma
    elif kind == "smartphone_mixed":
        sigma = (0.8, 1.5, 3.0, 5.0, 8.0)[severity - 1]
        quality = (75, 58, 40, 24, 12)[severity - 1]
        altered = cv2.GaussianBlur(clean, (0, 0), sigma)
        altered = np.clip(altered.astype(np.float32) + rng.normal(0, 2 + severity * 2, altered.shape), 0, 255).astype(np.uint8)
        ok, encoded = cv2.imencode(".jpg", altered, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("mixed JPEG encoding failed")
        altered = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        params.update(sigma=sigma, quality=quality)
    else:
        raise ValueError(f"unsupported degradation: {kind}")

    degraded = _masked_replace(clean, altered, damage)
    damaged_pixels = int(np.count_nonzero(damage))
    face_fraction = float(np.count_nonzero((damage > 0) & (face > 0)) / max(1, damaged_pixels))
    record = DegradationRecord(
        kind=kind,
        severity=severity,
        seed=seed,
        damaged_pixels=damaged_pixels,
        face_target_fraction=face_fraction,
        abstention_expected=bool(severity == 5 and kind in {"opaque_paint", "opaque_sticker", "marker_strokes", "scribble"}),
        parameters=params,
    )
    return degraded, damage, record


def record_dict(record: DegradationRecord) -> dict[str, object]:
    return asdict(record)
