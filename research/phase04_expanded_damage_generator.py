from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from phase04_damage_evaluation import EvaluationCase


@dataclass(frozen=True)
class ExpandedDamageSample:
    image: np.ndarray
    binary_mask: np.ndarray
    metadata: dict[str, Any]


def _rng(seed: int, case_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{int(seed)}:{case_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big", signed=False)
    return np.random.default_rng(value)


def _position_center(position: str) -> tuple[float, float]:
    return {
        "LEFT_EYE": (0.34, 0.37),
        "RIGHT_EYE": (0.66, 0.37),
        "NOSE": (0.50, 0.52),
        "MOUTH": (0.50, 0.68),
        "LEFT_CHEEK": (0.32, 0.57),
        "RIGHT_CHEEK": (0.68, 0.57),
        "FOREHEAD": (0.50, 0.24),
        "GLOBAL": (0.50, 0.50),
    }[position]


def _size_fraction(size: str) -> tuple[float, float]:
    return {
        "SMALL": (0.11, 0.08),
        "MEDIUM": (0.18, 0.12),
        "LARGE": (0.26, 0.18),
        "NONE": (0.0, 0.0),
    }[size]


def _severity_index(severity: str) -> int:
    return {"LIGHT": 0, "MEDIUM": 1, "SEVERE": 2, "NONE": 0}[severity]


def _ellipse_mask(shape: tuple[int, int], case: EvaluationCase, scale: float = 1.0) -> np.ndarray:
    h, w = shape
    cx, cy = _position_center(case.position)
    sx, sy = _size_fraction(case.size)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(cx * w)), int(round(cy * h))),
        (max(1, int(round(sx * w * scale))), max(1, int(round(sy * h * scale)))),
        0,
        0,
        360,
        255,
        -1,
        cv2.LINE_AA,
    )
    return mask


def _rect_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0, 1, 1
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _composite(base: np.ndarray, changed: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = base.copy()
    selected = mask > 0
    result[selected] = changed[selected]
    return result


def _motion_blur(image: np.ndarray, length: int, angle: float) -> np.ndarray:
    k = max(3, int(length) | 1)
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    center = (k / 2 - 0.5, k / 2 - 0.5)
    matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    kernel = cv2.warpAffine(kernel, matrix, (k, k), flags=cv2.INTER_LINEAR)
    kernel /= max(float(kernel.sum()), 1e-9)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _pixelate(image: np.ndarray, mask: np.ndarray, blocks: int) -> np.ndarray:
    x1, y1, x2, y2 = _rect_from_mask(mask)
    patch = image[y1:y2, x1:x2]
    ph, pw = patch.shape[:2]
    low_w = max(1, min(pw, int(blocks)))
    low_h = max(1, min(ph, int(round(blocks * ph / max(pw, 1)))))
    low = cv2.resize(patch, (low_w, low_h), interpolation=cv2.INTER_AREA)
    changed = image.copy()
    changed[y1:y2, x1:x2] = cv2.resize(low, (pw, ph), interpolation=cv2.INTER_NEAREST)
    return _composite(image, changed, mask)


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decode failed")
    return decoded


def _scribble(image: np.ndarray, case: EvaluationCase, *, black: bool, thick: bool, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    cx, cy = _position_center(case.position)
    sx, sy = _size_fraction(case.size)
    center = np.array([cx * w, cy * h], dtype=np.float64)
    span = np.array([max(10.0, sx * w * 1.8), max(8.0, sy * h * 1.8)], dtype=np.float64)
    points: list[tuple[int, int]] = []
    for index in range(6):
        t = index / 5.0
        x = center[0] - span[0] + 2.0 * span[0] * t
        y = center[1] + math.sin(t * math.pi * 3.0) * span[1] * 0.55
        x += float(rng.uniform(-0.06, 0.06) * span[0])
        y += float(rng.uniform(-0.10, 0.10) * span[1])
        points.append((int(round(x)), int(round(y))))
    thickness = max(2, int(round((0.018 if thick else 0.008) * min(h, w))))
    if case.severity == "SEVERE":
        thickness += max(1, min(h, w) // 120)
    colour = (8, 8, 8) if black else tuple(int(v) for v in rng.integers(15, 241, size=3))
    result = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.asarray(points, dtype=np.int32)
    cv2.polylines(result, [poly], False, colour, thickness, cv2.LINE_AA)
    cv2.polylines(mask, [poly], False, 255, thickness, cv2.LINE_AA)
    return result, mask


def apply_expanded_damage(clean_bgr: np.ndarray, case: EvaluationCase, *, seed: int) -> ExpandedDamageSample:
    clean = np.asarray(clean_bgr)
    if clean.dtype != np.uint8 or clean.ndim != 3 or clean.shape[2] != 3:
        raise ValueError("clean_bgr must be uint8 HxWx3")
    h, w = clean.shape[:2]
    if min(h, w) < 64:
        raise ValueError("image is too small")
    rng = _rng(seed, case.case_id)
    severity = _severity_index(case.severity)
    damage_type = case.damage_type

    if damage_type == "HEALTHY":
        return ExpandedDamageSample(clean.copy(), np.zeros((h, w), dtype=np.uint8), {"generator": "identity"})

    local = _ellipse_mask((h, w), case)
    result = clean.copy()
    mask = local.copy()

    if damage_type == "OPAQUE_STICKER":
        colour = np.full_like(clean, tuple(int(v) for v in rng.integers(20, 236, size=3)))
        result = _composite(clean, colour, mask)
    elif damage_type == "TRANSLUCENT_STICKER":
        alpha = {"LOW": 0.35, "MEDIUM": 0.55, "HIGH": 0.75}[case.opacity]
        colour = np.full_like(clean, tuple(int(v) for v in rng.integers(20, 236, size=3)))
        overlay = cv2.addWeighted(clean, 1.0 - alpha, colour, alpha, 0.0)
        result = _composite(clean, overlay, mask)
    elif damage_type == "EMOJI":
        result = clean.copy()
        mask = _ellipse_mask((h, w), case, scale=1.05)
        x1, y1, x2, y2 = _rect_from_mask(mask)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = max(4, min(x2 - x1, y2 - y1) // 2)
        cv2.circle(result, center, radius, (0, 220, 255), -1, cv2.LINE_AA)
        eye_r = max(1, radius // 9)
        cv2.circle(result, (center[0] - radius // 3, center[1] - radius // 4), eye_r, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(result, (center[0] + radius // 3, center[1] - radius // 4), eye_r, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.ellipse(result, (center[0], center[1] + radius // 5), (radius // 2, radius // 3), 0, 10, 170, (20, 20, 20), max(1, radius // 10), cv2.LINE_AA)
        mask = np.any(result != clean, axis=2).astype(np.uint8) * 255
    elif damage_type == "TEXT":
        result = clean.copy()
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = _position_center(case.position)
        scale = {"SMALL": 0.45, "MEDIUM": 0.65, "LARGE": 0.9}[case.size]
        thickness = 1 + severity
        text = ("X", "FACE", "BLOCK")[severity]
        origin = (max(1, int((cx - 0.15) * w)), max(12, int(cy * h)))
        colour = tuple(int(v) for v in rng.integers(0, 256, size=3))
        cv2.putText(result, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness, cv2.LINE_AA)
        cv2.putText(mask, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness, cv2.LINE_AA)
    elif damage_type.startswith("SCRIBBLE_"):
        result, mask = _scribble(clean, case, black="BLACK" in damage_type, thick="THICK" in damage_type, rng=rng)
    elif damage_type == "BLUR_LOCAL":
        sigma = (2.0, 4.0, 7.0)[severity]
        changed = cv2.GaussianBlur(clean, (0, 0), sigma)
        result = _composite(clean, changed, mask)
    elif damage_type == "BLUR_GLOBAL":
        sigma = (1.6, 3.2, 6.0)[severity]
        result = cv2.GaussianBlur(clean, (0, 0), sigma)
        mask = np.full((h, w), 255, dtype=np.uint8)
    elif damage_type == "MOTION_BLUR":
        changed = _motion_blur(clean, (9, 17, 29)[severity], (-35.0, 15.0, 48.0)[severity])
        result = _composite(clean, changed, mask)
    elif damage_type == "DEFOCUS":
        radius = (3, 5, 9)[severity]
        kernel_size = radius * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1.0, -1, cv2.LINE_AA)
        kernel /= max(float(kernel.sum()), 1e-9)
        changed = cv2.filter2D(clean, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        result = _composite(clean, changed, mask)
    elif damage_type == "BLOCK_MOSAIC":
        result = _pixelate(clean, mask, (9, 6, 3)[severity])
    elif damage_type == "PIXELATION":
        result = _pixelate(clean, mask, (24, 16, 10)[severity])
    elif damage_type == "JPEG_ARTIFACT":
        result = _jpeg(clean, (45, 22, 8)[severity])
        mask = np.full((h, w), 255, dtype=np.uint8)
    elif damage_type == "NOISE":
        sigma = (8.0, 18.0, 34.0)[severity]
        noise = rng.normal(0.0, sigma, clean.shape).astype(np.float32)
        result = np.clip(clean.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        mask = np.full((h, w), 255, dtype=np.uint8)
    elif damage_type == "MIXED_DAMAGE":
        blur_mask = _ellipse_mask((h, w), case, scale=1.15)
        blurred = cv2.GaussianBlur(clean, (0, 0), (2.5, 4.0, 6.0)[severity])
        result = _composite(clean, blurred, blur_mask)
        cx, cy = _position_center(case.position)
        second = np.zeros((h, w), dtype=np.uint8)
        offset = int(round(0.08 * w))
        p1 = (max(0, int(cx * w) - offset), max(0, int(cy * h) - offset))
        p2 = (min(w - 1, int(cx * w) + offset), min(h - 1, int(cy * h) + offset))
        cv2.line(second, p1, p2, 255, max(3, w // 45), cv2.LINE_AA)
        changed = result.copy()
        cv2.line(changed, p1, p2, (5, 5, 5), max(3, w // 45), cv2.LINE_AA)
        result = changed
        mask = cv2.bitwise_or(blur_mask, second)
    else:
        raise ValueError(f"unsupported damage type: {damage_type}")

    # Authority truth is exactly the pixels generated by this operation. This avoids
    # counting anti-aliased mask fringes where the encoded image is byte-identical.
    changed_pixels = np.any(result != clean, axis=2)
    truth = np.where((mask > 0) & changed_pixels, 255, 0).astype(np.uint8)
    if not np.any(truth):
        raise RuntimeError(f"damage generator produced no changed pixels: {case.case_id}")
    # Enforce the safety property used by downstream restoration: outside truth the
    # source image remains exactly byte-identical.
    result[truth == 0] = clean[truth == 0]
    return ExpandedDamageSample(
        result,
        truth,
        {
            "generator": "phase04_expanded_damage_v1",
            "damage_type": damage_type,
            "position": case.position,
            "size": case.size,
            "severity": case.severity,
            "opacity": case.opacity,
            "seed": int(seed),
        },
    )
