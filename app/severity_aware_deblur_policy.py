from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.restoration import detect_occlusion_candidates, detail_reliability_map

_INSTALLED = False


def _central_face_like_domain(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(
        mask,
        (w // 2, int(round(h * 0.48))),
        (max(8, int(round(w * 0.30))), max(8, int(round(h * 0.38)))),
        0, 0, 360, 255, -1,
    )
    return mask > 0


def classify_blur(image: np.ndarray) -> dict[str, Any]:
    """Classify observed blur without treating occlusion edges as useful detail."""
    occ = detect_occlusion_candidates(image)
    rel = detail_reliability_map(image, occ)
    domain = _central_face_like_domain(image.shape[:2]) & (occ == 0)
    if int(np.count_nonzero(domain)) < 128:
        domain = occ == 0
    if int(np.count_nonzero(domain)) < 128:
        return {"level": "unknown", "score": 0.0, "low_fraction": 0.0, "median_reliability": 0.0}

    values = rel[domain].astype(np.float32)
    low_fraction = float(np.mean(values < 45.0))
    median_rel = float(np.median(values))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lap_p75 = float(np.percentile(lap[domain], 75.0))

    strong = low_fraction >= 0.62 and median_rel <= 28.0 and lap_p75 <= 18.0
    medium = low_fraction >= 0.38 and median_rel <= 52.0 and lap_p75 <= 30.0
    mild = low_fraction >= 0.20 and median_rel <= 75.0
    level = "strong" if strong else "medium" if medium else "mild" if mild else "none"
    score = float(np.clip(0.55 * low_fraction + 0.45 * (1.0 - median_rel / 255.0), 0.0, 1.0))
    return {
        "level": level,
        "score": score,
        "low_fraction": low_fraction,
        "median_reliability": median_rel,
        "laplacian_p75": lap_p75,
    }


def install_severity_aware_deblur_policy() -> None:
    """Expose severity routing without ever deblurring an already deblurred candidate.

    The selective preflight wrapper classifies immutable inputs before NAFNet and runs
    learned inference only for MEDIUM/STRONG photographs. This layer records the same
    severity diagnostics for downstream planning. It deliberately performs no second
    learned pass: stronger alternatives must be independent candidates derived from the
    original image (for example a future Restormer provider), never NAFNet(NAFNet(x)).
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import app.preflight as module

    previous = module._deblur_all

    @wraps(previous)
    def routed(images: list[np.ndarray], model_path: Path | None, hardware_policy: dict[str, Any]):
        diagnostics = [classify_blur(np.asarray(image)) for image in images]
        output, evaluated = previous(images, model_path, hardware_policy)
        module._last_blur_diagnostics = [dict(item) for item in diagnostics]
        module._last_strong_blur_second_pass_indices = []
        module._last_deblur_candidate_policy = "classify-original-once-no-cumulative-warp-or-deblur"
        return output, evaluated

    module._deblur_all = routed
    _INSTALLED = True
