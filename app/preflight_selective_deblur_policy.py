from __future__ import annotations

from functools import wraps
from pathlib import Path

import cv2
import numpy as np

from app.restoration import detect_occlusion_candidates, detail_reliability_map


_INSTALLED = False


def _observed_fraction(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    return float(np.mean(np.max(image, axis=2) > 2))


def _selective_learned_blend(original: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Keep learned deblur only where the observed image truly lacks detail.

    The candidate has already passed through the conservative NAFNet blend. This
    function does not strengthen it; it only restores original pixels in reliable or
    occluded regions. Sparse component-sheet references are returned byte-identical so
    a deblur network can never populate their intentionally empty canvas.
    """
    if candidate.shape != original.shape:
        return original.copy(), {"active_fraction": 0.0, "observed_fraction": _observed_fraction(original)}

    observed_fraction = _observed_fraction(original)
    if observed_fraction < 0.30:
        return original.copy(), {"active_fraction": 0.0, "observed_fraction": observed_fraction}

    occlusion = detect_occlusion_candidates(original)
    reliability = detail_reliability_map(original, occlusion)
    observed = np.max(original, axis=2) > 2

    # Low-detail is evidence for deblur only when it is not also an occlusion hint.
    active = observed & (occlusion == 0) & (reliability < 40)
    if int(np.count_nonzero(active)) < max(64, int(round(original.shape[0] * original.shape[1] * 0.004))):
        return original.copy(), {"active_fraction": 0.0, "observed_fraction": observed_fraction}

    mask = active.astype(np.float32)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = cv2.GaussianBlur(mask, (0, 0), 1.6)
    mask = np.clip(mask, 0.0, 1.0)[..., None]

    original_f = original.astype(np.float32)
    candidate_f = candidate.astype(np.float32)
    blended = np.clip(original_f * (1.0 - mask) + candidate_f * mask, 0.0, 255.0).astype(np.uint8)
    return blended, {
        "active_fraction": float(np.mean(active)),
        "observed_fraction": observed_fraction,
    }


def install_preflight_selective_deblur_policy() -> None:
    """Prevent global NAFNet drift while retaining real blur recovery.

    Preflight still evaluates every imported image once. A verified NAFNet candidate is
    accepted only in low-reliability, non-occluded observed regions. Reliable pixels,
    stickers/scribbles and sparse component-reference padding remain exactly observed.
    Returning the number of evaluated images also prevents an unnecessary second NAFNet
    pass in block 2; it does not claim that every pixel was modified.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import app.preflight as module

    original_deblur_all = module._deblur_all

    @wraps(original_deblur_all)
    def patched_deblur_all(images, model_path: Path | None, hardware_policy):
        candidates, applied = original_deblur_all(images, model_path, hardware_policy)
        if model_path is None or not Path(model_path).is_file() or applied <= 0:
            return candidates, applied

        output: list[np.ndarray] = []
        diagnostics: list[dict[str, float]] = []
        for original, candidate in zip(images, candidates):
            selected, details = _selective_learned_blend(original, candidate)
            output.append(selected)
            diagnostics.append(details)

        # In preflight semantics this count means that every imported image received a
        # deblur decision. This intentionally suppresses the old unconditional second
        # pass in the DEBLUR block, which otherwise reintroduced global learned drift.
        return output, len(images)

    module._deblur_all = patched_deblur_all
    _INSTALLED = True
