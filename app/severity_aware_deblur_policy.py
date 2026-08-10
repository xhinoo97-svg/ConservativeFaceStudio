from __future__ import annotations

from functools import wraps
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.opencv_nafnet import NafNetDeblurEngine
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

    # Two independent signals are required for strong blur. This prevents smooth skin
    # or a plain background from triggering an unnecessary second learned pass.
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


def _selective_mix(original: np.ndarray, learned: np.ndarray, strength: float) -> np.ndarray:
    if learned.shape != original.shape:
        learned = cv2.resize(learned, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    occ = detect_occlusion_candidates(original)
    rel = detail_reliability_map(original, occ)
    observed = np.max(original, axis=2) > 2
    active = observed & (occ == 0) & (rel < 70)
    if not np.any(active):
        return original.copy()
    weight = np.zeros(original.shape[:2], dtype=np.float32)
    weight[active] = np.clip((70.0 - rel[active].astype(np.float32)) / 70.0, 0.15, 1.0)
    weight *= float(np.clip(strength, 0.0, 1.0))
    weight = cv2.GaussianBlur(weight, (0, 0), 1.4)[..., None]
    a = original.astype(np.float32)
    b = learned.astype(np.float32)
    return np.clip(np.rint(a * (1.0 - weight) + b * weight), 0, 255).astype(np.uint8)


def install_severity_aware_deblur_policy() -> None:
    """Route mild/medium/strong blur while keeping all ten images evidence-aware.

    The existing preflight still performs the first conservative NAFNet decision for
    every imported image. This wrapper only adds a second tiled pass to images that
    meet the strict 'strong blur' classifier. Mild/medium images are never repeatedly
    sharpened. Occluded pixels never receive learned deblur and frozen pre-restoration
    reliability remains the evidence authority downstream.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    import app.preflight as module

    previous = module._deblur_all

    @wraps(previous)
    def routed(images: list[np.ndarray], model_path: Path | None, hardware_policy: dict[str, Any]):
        first, evaluated = previous(images, model_path, hardware_policy)
        diagnostics = [classify_blur(image) for image in images]
        if model_path is None or not Path(model_path).is_file():
            return first, evaluated

        target = str(hardware_policy.get("dnn_target", "cpu")).lower()
        target = "opencl" if target == "opencl" else "cpu"
        tile = max(128, int(hardware_policy.get("heavy_tile_size", 384)))
        engines: dict[str, NafNetDeblurEngine] = {}

        def engine(name: str) -> NafNetDeblurEngine:
            if name not in engines:
                engines[name] = NafNetDeblurEngine(model_path, target=name, tile_size=tile, overlap=32)
            return engines[name]

        output: list[np.ndarray] = []
        second_pass_indices: list[int] = []
        for index, (original, current, info) in enumerate(zip(images, first, diagnostics)):
            level = str(info.get("level", "none"))
            if level != "strong":
                output.append(current)
                continue
            try:
                try:
                    learned = engine(target).infer(current)
                except Exception:
                    if target != "opencl":
                        raise
                    learned = engine("cpu").infer(current)
                candidate = _selective_mix(current, learned, 0.72)
                # Reject a second pass that creates excessive pixel changes outside
                # genuinely low-detail, non-occluded domains.
                occ = detect_occlusion_candidates(original)
                rel = detail_reliability_map(original, occ)
                allowed = (occ == 0) & (rel < 80) & (np.max(original, axis=2) > 2)
                changed = np.any(candidate != current, axis=2)
                outside_fraction = float(np.count_nonzero(changed & ~allowed) / max(1, changed.size))
                if outside_fraction > 0.005:
                    output.append(current)
                else:
                    output.append(candidate)
                    second_pass_indices.append(index)
            except Exception:
                output.append(current)

        # Expose diagnostics without changing the historical return contract.
        module._last_blur_diagnostics = diagnostics
        module._last_strong_blur_second_pass_indices = second_pass_indices
        return output, len(images)

    module._deblur_all = routed
    _INSTALLED = True
