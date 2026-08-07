from __future__ import annotations

import cv2
import numpy as np


def fit_to_canvas(image: np.ndarray, canvas_shape: tuple[int, int]) -> tuple[np.ndarray, dict[str, float | int]]:
    """Resize preserving aspect ratio, then center-pad to the requested HxW canvas.

    This avoids the geometric stretching that would otherwise change facial proportions
    before landmark/feature alignment.
    """
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Sono supportate immagini BGR a 3 canali")
    target_h, target_w = map(int, canvas_shape)
    if target_h <= 0 or target_w <= 0:
        raise ValueError("Dimensioni canvas non valide")

    source_h, source_w = image.shape[:2]
    scale = min(target_w / float(source_w), target_h / float(source_h))
    resized_w = max(1, min(target_w, int(round(source_w * scale))))
    resized_h = max(1, min(target_h, int(round(source_h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

    left = (target_w - resized_w) // 2
    right = target_w - resized_w - left
    top = (target_h - resized_h) // 2
    bottom = target_h - resized_h - top
    canvas = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REFLECT_101,
    )
    if canvas.shape[:2] != (target_h, target_w):
        raise RuntimeError("Normalizzazione geometrica non ha prodotto il canvas richiesto")
    metadata: dict[str, float | int] = {
        "scale": float(scale),
        "source_width": int(source_w),
        "source_height": int(source_h),
        "target_width": int(target_w),
        "target_height": int(target_h),
        "pad_left": int(left),
        "pad_right": int(right),
        "pad_top": int(top),
        "pad_bottom": int(bottom),
    }
    return canvas, metadata
