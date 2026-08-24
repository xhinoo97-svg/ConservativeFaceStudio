from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ObservedRestorationDecision:
    runtime_index: int
    original_source_index: int
    sharpness_score: float
    action: str
    changed_pixels: int
    face_sharpness_score: float = 0.0
    component_sharpness_score: float = 0.0
    blur_category: str = "unknown"


def normalized_laplacian_variance(image: np.ndarray, *, analysis_size: int = 320) -> float:
    """Resolution-normalized blur cue measured only on observed non-black pixels."""
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        return 0.0
    sample = image
    h, w = sample.shape[:2]
    scale = min(1.0, float(analysis_size) / max(h, w))
    if scale < 1.0:
        sample = cv2.resize(
            sample,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    observed = np.max(sample, axis=2) > 2
    if int(np.count_nonzero(observed)) < 128:
        return 0.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.var(lap[observed]))


def conservative_strong_defocus_repair(image: np.ndarray) -> np.ndarray:
    """Deterministic unsharp restoration for extreme smooth defocus.

    It never creates geometry or texture from another source. The radius is bounded so
    high-resolution input does not turn into a large halo operation.
    """
    h, w = image.shape[:2]
    sigma = float(np.clip(min(h, w) / 110.0, 1.5, 3.5))
    smooth = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 3.0, smooth, -2.0, 0.0)


def face_local_blur_analysis(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> tuple[float, float, str]:
    """Measure blur on face and identity-bearing components, never the whole canvas."""
    h, w = image.shape[:2]
    if bbox is None:
        x, y, width, height = int(w * 0.2), int(h * 0.12), int(w * 0.6), int(h * 0.72)
    else:
        x, y, width, height = (int(value) for value in bbox)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + max(1, width)), min(h, y + max(1, height))
    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return 0.0, 0.0, "unknown"
    face_score = normalized_laplacian_variance(face)
    fh, fw = face.shape[:2]
    regions = (
        face[int(fh * 0.22):int(fh * 0.48), int(fw * 0.12):int(fw * 0.88)],
        face[int(fh * 0.35):int(fh * 0.72), int(fw * 0.30):int(fw * 0.70)],
        face[int(fh * 0.62):int(fh * 0.88), int(fw * 0.22):int(fw * 0.78)],
    )
    scores = [normalized_laplacian_variance(region, analysis_size=160) for region in regions if region.size]
    component_score = float(np.median(scores)) if scores else face_score
    score = min(face_score, component_score)
    if score >= 120.0:
        category = "clean"
    elif score >= 55.0:
        category = "mild_blur"
    elif score >= 20.0:
        category = "medium_blur"
    elif score >= 7.0:
        category = "heavy_blur"
    else:
        category = "severe_blur_or_destroyed_information"
    return float(face_score), component_score, category


def apply_observed_restoration_policy(
    workspace,
    originals: list[np.ndarray],
    *,
    strong_defocus_threshold: float = 12.0,
    preserve_observed_threshold: float = 120.0,
) -> tuple[ObservedRestorationDecision, ...]:
    """Choose between observed pixels, learned preflight output and safe defocus repair.

    Preflight is allowed to run NAFNet for face analysis, but a sharp photograph is
    restored to its exact observed pixels before the restoration pipeline. This keeps
    clean/full/component references from becoming network-generated donor evidence.
    Extremely smooth defocus uses a deterministic bounded unsharp operation because
    the current deblur network can otherwise worsen that blur family. Intermediate
    blur retains the learned preflight result.
    """
    runtime = [workspace.primary, *workspace.references]
    order_raw = workspace.metadata.get("runtime_source_order")
    if isinstance(order_raw, list) and len(order_raw) == len(runtime):
        try:
            order = [int(value) for value in order_raw]
        except (TypeError, ValueError):
            order = list(range(len(runtime)))
    else:
        order = list(range(len(runtime)))

    decided: list[np.ndarray] = []
    diagnostics: list[ObservedRestorationDecision] = []
    bboxes = workspace.metadata.get("preflight_face_bboxes")
    for runtime_index, current in enumerate(runtime):
        source_index = order[runtime_index] if runtime_index < len(order) else runtime_index
        if source_index < 0 or source_index >= len(originals):
            original = current
            source_index = runtime_index
        else:
            original = originals[source_index]
        if original.shape != current.shape:
            original = current

        sharpness = normalized_laplacian_variance(original)
        bbox = None
        if isinstance(bboxes, list) and source_index < len(bboxes) and bboxes[source_index] is not None:
            bbox = tuple(int(value) for value in bboxes[source_index])
        face_sharpness, component_sharpness, blur_category = face_local_blur_analysis(original, bbox)
        if sharpness <= float(strong_defocus_threshold) and face_sharpness <= float(strong_defocus_threshold):
            chosen = conservative_strong_defocus_repair(original)
            action = "bounded-classical-strong-defocus"
        elif sharpness >= float(preserve_observed_threshold) and min(face_sharpness, component_sharpness) >= 55.0:
            chosen = original.copy()
            action = "preserve-observed"
        else:
            chosen = current.copy()
            action = "retain-preflight-nafnet"

        changed = int(np.count_nonzero(np.any(chosen != original, axis=2)))
        decided.append(chosen)
        diagnostics.append(
            ObservedRestorationDecision(
                runtime_index=int(runtime_index),
                original_source_index=int(source_index),
                sharpness_score=float(sharpness),
                action=action,
                changed_pixels=changed,
                face_sharpness_score=face_sharpness,
                component_sharpness_score=component_sharpness,
                blur_category=blur_category,
            )
        )

    workspace.primary = decided[0]
    workspace.references = [item.copy() for item in decided[1:]]
    workspace.metadata["observed_restoration_policy"] = [asdict(item) for item in diagnostics]
    workspace.metadata["observed_restoration_preserved_count"] = int(
        sum(item.action == "preserve-observed" for item in diagnostics)
    )
    workspace.metadata["observed_restoration_strong_defocus_count"] = int(
        sum(item.action == "bounded-classical-strong-defocus" for item in diagnostics)
    )
    return tuple(diagnostics)
