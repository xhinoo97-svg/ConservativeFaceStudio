from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import numpy as np


_INSTALLED = False


def _strong_blur_evidence(image: np.ndarray) -> tuple[bool, dict[str, float]]:
    """Return True only for clearly blurred observed photographs.

    The preflight deblur stage is evidence-preserving: a pretrained restoration model
    must not touch already sharp primary/reference pixels merely because weights are
    available.  We therefore require two independent low-detail signals before NAFNet
    is allowed: Laplacian variance and median Sobel magnitude on the central image area.
    Thresholds are intentionally conservative and target heavy blur, not mild softness.
    """
    if image is None or image.size == 0:
        return False, {"laplacian_variance": 0.0, "median_gradient": 0.0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    y0, y1 = int(round(0.10 * h)), int(round(0.90 * h))
    x0, x1 = int(round(0.10 * w)), int(round(0.90 * w))
    crop = gray[y0:y1, x0:x1]
    if crop.size < 256:
        crop = gray
    lap = cv2.Laplacian(crop, cv2.CV_32F, ksize=3)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    lap_var = float(np.var(lap))
    median_gradient = float(np.median(gradient))
    # Both conditions are required.  Flat backgrounds alone cannot trigger NAFNet.
    strong = bool(lap_var < 85.0 and median_gradient < 24.0)
    return strong, {
        "laplacian_variance": lap_var,
        "median_gradient": median_gradient,
        "laplacian_threshold": 85.0,
        "gradient_threshold": 24.0,
    }


def install_selective_preflight_deblur_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.preflight as module

    original_deblur_all = module._deblur_all

    @wraps(original_deblur_all)
    def selective_deblur_all(
        images: list[np.ndarray],
        model_path: Path | None,
        hardware_policy: dict[str, Any],
    ) -> tuple[list[np.ndarray], int]:
        if model_path is None or not Path(model_path).is_file():
            return [item.copy() for item in images], 0

        output: list[np.ndarray] = []
        applied = 0
        diagnostics: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            strong, metrics = _strong_blur_evidence(image)
            if not strong:
                output.append(image.copy())
                diagnostics.append({"source_index": index, "nafnet_applied": False, **metrics})
                continue
            restored, count = original_deblur_all([image], model_path, hardware_policy)
            output.append(restored[0])
            applied += int(count)
            diagnostics.append({"source_index": index, "nafnet_applied": bool(count), **metrics})

        # Diagnostics are returned indirectly through the module for callers that want
        # to expose them; no global image data is retained.
        module._last_selective_deblur_diagnostics = diagnostics
        return output, applied

    module._deblur_all = selective_deblur_all
    _INSTALLED = True
