from __future__ import annotations

from functools import wraps
from pathlib import Path

import cv2
import numpy as np

from app.severity_aware_deblur_policy import classify_blur


_INSTALLED = False


def _observed_fraction(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    return float(np.mean(np.max(image, axis=2) > 2))


def _learned_deblur_indices(images: list[np.ndarray]) -> tuple[list[int], list[dict]]:
    """Return only source indices that justify learned NAFNet inference."""
    diagnostics = [classify_blur(np.asarray(item)) for item in images]
    indices = [
        index
        for index, info in enumerate(diagnostics)
        if str(info.get("level", "none")) in {"medium", "strong"}
        and _observed_fraction(np.asarray(images[index])) >= 0.30
    ]
    return indices, [dict(item) for item in diagnostics]


def _selective_learned_blend(original: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Keep a NAFNet candidate only where the original genuinely lacks detail."""
    if candidate.shape != original.shape:
        return original.copy(), {"active_fraction": 0.0, "observed_fraction": _observed_fraction(original)}

    observed_fraction = _observed_fraction(original)
    if observed_fraction < 0.30:
        return original.copy(), {"active_fraction": 0.0, "observed_fraction": observed_fraction}

    from app.restoration import detect_occlusion_candidates, detail_reliability_map

    occlusion = detect_occlusion_candidates(original)
    reliability = detail_reliability_map(original, occlusion)
    observed = np.max(original, axis=2) > 2
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
    """Classify first, then run NAFNet only where learned deblur is justified.

    CLEAN and LIGHT photographs are never sent through NAFNet merely because weights
    exist. MEDIUM/STRONG candidates are inferred from the immutable observed input once,
    then selectively accepted only in low-detail, non-occluded pixels. Returning the
    number of evaluated images preserves the historical preflight contract: block 2 can
    reuse the decision without re-running learned deblur on already evaluated inputs.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import app.preflight as module

    original_deblur_all = module._deblur_all

    @wraps(original_deblur_all)
    def patched_deblur_all(images, model_path: Path | None, hardware_policy):
        originals = [np.asarray(item).copy() for item in images]
        learned_indices, diagnostics = _learned_deblur_indices(originals)

        module._last_preflight_blur_diagnostics = diagnostics
        module._last_preflight_nafnet_indices = list(learned_indices)

        if model_path is None or not Path(model_path).is_file() or not learned_indices:
            return originals, len(originals)

        selected_inputs = [originals[index] for index in learned_indices]
        candidates, _ = original_deblur_all(selected_inputs, model_path, hardware_policy)

        output = [item.copy() for item in originals]
        selective_diagnostics: list[dict[str, float]] = [
            {"active_fraction": 0.0, "observed_fraction": _observed_fraction(item)}
            for item in originals
        ]
        for local_index, source_index in enumerate(learned_indices):
            candidate = candidates[local_index] if local_index < len(candidates) else originals[source_index]
            selected, details = _selective_learned_blend(originals[source_index], candidate)
            output[source_index] = selected
            selective_diagnostics[source_index] = details

        module._last_preflight_selective_deblur = selective_diagnostics
        return output, len(originals)

    module._deblur_all = patched_deblur_all
    _INSTALLED = True
