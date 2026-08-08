from __future__ import annotations

from functools import wraps
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.pipeline import BlockKind, BlockSpec
from app.strict_repair import face_support_mask


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        raise ValueError("Maschera non compatibile")
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _original_source_indices(workspace, runtime_indices: list[int], count: int) -> list[int]:
    explicit = workspace.metadata.get("aligned_reference_original_source_indices")
    if isinstance(explicit, list) and len(explicit) == count:
        return [max(1, int(value)) for value in explicit]
    order_raw = workspace.metadata.get("runtime_source_order")
    order = [int(value) for value in order_raw] if isinstance(order_raw, list) else []
    resolved: list[int] = []
    for runtime_reference_index in runtime_indices:
        slot = int(runtime_reference_index) + 1
        original = order[slot] if 0 <= slot < len(order) else slot
        resolved.append(max(1, int(original)))
    return resolved


def _verified_donor_slots(workspace, runtime_indices: list[int], originals: list[int]) -> tuple[set[int], set[int]]:
    verified_runtime: set[int] = set()
    for key in ("verified_same_canvas_alignment", "same_canvas_partial_alignment_diagnostics"):
        diagnostics = workspace.metadata.get(key)
        if not isinstance(diagnostics, list):
            continue
        for item in diagnostics:
            if not isinstance(item, dict) or item.get("runtime_reference_index") is None:
                continue
            method = str(item.get("method", ""))
            if method in {"verified-same-canvas-observed", "verified-same-canvas-partial"}:
                verified_runtime.add(int(item["runtime_reference_index"]))

    anchor = workspace.metadata.get("same_canvas_primary_anchor")
    verified_original = {
        int(value)
        for value in anchor.get("matched_original_reference_indices", [])
    } if isinstance(anchor, dict) and isinstance(anchor.get("matched_original_reference_indices"), list) else set()
    return verified_runtime, verified_original


def _damage_seed(workspace, shape: tuple[int, int]) -> np.ndarray:
    seed = np.zeros(shape, dtype=np.uint8)
    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and frozen:
        try:
            seed = cv2.bitwise_or(seed, _binary(np.asarray(frozen[0]), shape))
        except (TypeError, ValueError):
            pass
    masks = workspace.occlusion_masks
    if isinstance(masks, list) and masks:
        try:
            seed = cv2.bitwise_or(seed, _binary(np.asarray(masks[0]), shape))
        except (TypeError, ValueError):
            pass
    target = workspace.metadata.get("inpaint_target_mask")
    if isinstance(target, np.ndarray) and target.shape == shape:
        seed = cv2.bitwise_or(seed, _binary(target, shape))
    return seed


def _frozen_primary(workspace) -> np.ndarray:
    frozen = workspace.metadata.get("same_canvas_imported_primary")
    if isinstance(frozen, np.ndarray) and frozen.shape == workspace.primary.shape:
        return frozen
    return workspace.primary


def _adaptive_difference_threshold(
    difference: np.ndarray,
    observed: np.ndarray,
    seed_bool: np.ndarray,
    maximum_threshold: float,
) -> tuple[float, dict[str, float]]:
    baseline = difference[observed & ~seed_bool]
    ceiling = float(max(0.02, maximum_threshold))
    if baseline.size < 64:
        return ceiling, {"baseline_median": 0.0, "baseline_p95": 0.0, "baseline_mad": 0.0}
    median = float(np.median(baseline))
    p95 = float(np.percentile(baseline, 95.0))
    mad = float(np.median(np.abs(baseline - median)))
    robust = max(0.025, p95 + 0.012, median + 6.0 * 1.4826 * mad)
    threshold = float(np.clip(min(ceiling, robust), 0.025, max(0.025, ceiling)))
    return threshold, {"baseline_median": median, "baseline_p95": p95, "baseline_mad": mad}


def _filled_component(component: np.ndarray) -> np.ndarray:
    """Fill only holes enclosed by one verified residual component.

    Dark donor pixels can numerically resemble a dark occluder and therefore fall below
    the residual threshold even though they are spatially inside a verified damage patch.
    Filling the external contour recovers those enclosed pixels without growing beyond
    the observed component boundary. The caller still intersects this mask with donor
    support and the face mask.
    """
    mask = np.where(component, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled > 0


def _strong_components(
    strong: np.ndarray,
    seed_reach: np.ndarray,
    difference: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, int]:
    binary = np.where(strong, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=bool), 0
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    keep = np.zeros(binary.shape, dtype=bool)
    unseeded_pixels = 0
    seed_bool = seed_reach > 0
    very_strong_cutoff = max(float(threshold) * 1.6, float(threshold) + 0.035)
    for label in range(1, count):
        component = labels == label
        area = int(np.count_nonzero(component))
        if area <= 0:
            continue
        if np.any(component & seed_bool):
            keep |= _filled_component(component)
            continue
        component_strength = float(np.median(difference[component]))
        if area >= 6 and component_strength >= very_strong_cutoff:
            keep |= component
            unseeded_pixels += area
    return keep, unseeded_pixels


def _limit_expansion(
    seed_selected: np.ndarray,
    expansion: np.ndarray,
    strength: np.ndarray,
    maximum_pixels: int,
) -> np.ndarray:
    """Never discard verified seed pixels; cap only evidence expansion around them."""
    selected = seed_selected.copy()
    seed_count = int(np.count_nonzero(selected))
    if maximum_pixels <= seed_count:
        return selected
    extra = expansion & ~selected
    extra_count = int(np.count_nonzero(extra))
    remaining = maximum_pixels - seed_count
    if extra_count <= remaining:
        selected |= extra
        return selected
    coords = np.flatnonzero(extra)
    values = strength.ravel()[coords]
    keep = coords[np.argpartition(values, -remaining)[-remaining:]] if remaining > 0 else np.asarray([], dtype=np.int64)
    if keep.size:
        selected.ravel()[keep] = True
    return selected


def exact_same_canvas_observed_repair(
    workspace,
    image: np.ndarray,
    *,
    difference_threshold: float = 0.075,
    maximum_face_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Repair only from donor pixels whose same-canvas geometry is already verified.

    The immutable imported primary is used for donor-difference expansion. The detector
    mask remains a seed, not a hard ceiling: verified same-canvas residual components may
    extend the target when the detector has low recall. Partial same-canvas references are
    accepted only inside their actually observed support. No synthesis or interpolation is
    used and visible pixels outside verified residual evidence remain unchanged.
    """
    shape = workspace.primary.shape[:2]
    aligned = list(workspace.aligned_references)
    runtime_indices_raw = workspace.metadata.get("aligned_reference_source_indices")
    supports_raw = workspace.metadata.get("aligned_reference_support_masks")
    if not aligned:
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "no_aligned_references", "repaired_pixels": 0}
    if not isinstance(runtime_indices_raw, list) or len(runtime_indices_raw) != len(aligned):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "missing_runtime_source_mapping", "repaired_pixels": 0}
    if not isinstance(supports_raw, list) or len(supports_raw) != len(aligned):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "missing_observed_support", "repaired_pixels": 0}

    runtime_indices = [int(value) for value in runtime_indices_raw]
    originals = _original_source_indices(workspace, runtime_indices, len(aligned))
    verified_runtime, verified_original = _verified_donor_slots(workspace, runtime_indices, originals)
    verified_slots = [
        runtime_indices[index] in verified_runtime or originals[index] in verified_original
        for index in range(len(aligned))
    ]
    if not any(verified_slots):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {
            "applied": False,
            "reason": "no_verified_same_canvas_reference",
            "repaired_pixels": 0,
            "verified_runtime_indices": sorted(verified_runtime),
            "verified_original_indices": sorted(verified_original),
        }

    seed = _damage_seed(workspace, shape)
    if not np.any(seed):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "no_observed_damage_seed", "repaired_pixels": 0}

    bbox_raw = workspace.metadata.get("primary_bbox")
    bbox = tuple(int(v) for v in bbox_raw) if bbox_raw is not None else None
    face = face_support_mask(shape, bbox) > 0
    face_pixels = max(1, int(np.count_nonzero(face)))
    maximum_pixels = max(0, int(round(face_pixels * float(maximum_face_fraction))))
    seed_bool = (seed > 0) & face
    seed_reach = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)

    frozen_primary = _frozen_primary(workspace)
    base_lab = cv2.cvtColor(frozen_primary, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

    result = image.copy()
    provenance = np.zeros(shape, dtype=np.uint16)
    repaired_union = np.zeros(shape, dtype=bool)
    source_counts: dict[int, int] = {}
    seed_pixel_count = 0
    expanded_pixel_count = 0
    unseeded_strong_pixel_count = 0
    threshold_diagnostics: list[dict[str, Any]] = []

    for slot, (reference, support_raw, original_index) in enumerate(zip(aligned, supports_raw, originals)):
        if not verified_slots[slot] or reference.shape != workspace.primary.shape:
            continue
        support = _binary(np.asarray(support_raw), shape) > 0
        reference_valid = np.max(reference, axis=2) > 2
        observed = support & reference_valid & face
        if not np.any(observed):
            continue

        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        difference = np.mean(np.abs(base_lab - ref_lab), axis=2)
        adaptive_threshold, baseline_stats = _adaptive_difference_threshold(
            difference,
            observed,
            seed_bool,
            float(difference_threshold),
        )
        strong = observed & (difference >= adaptive_threshold)
        expansion, unseeded_pixels = _strong_components(strong, seed_reach, difference, adaptive_threshold)
        expansion &= observed & ~repaired_union
        unseeded_strong_pixel_count += int(unseeded_pixels)
        threshold_diagnostics.append({
            "slot": int(slot),
            "runtime_reference_index": int(runtime_indices[slot]),
            "original_source_index": int(original_index),
            "adaptive_difference_threshold": float(adaptive_threshold),
            **baseline_stats,
        })

        seeded_observed = seed_bool & observed & ~repaired_union
        selected = _limit_expansion(
            seeded_observed,
            expansion,
            difference,
            maximum_pixels - int(np.count_nonzero(repaired_union)),
        )
        selected &= ~repaired_union
        if not np.any(selected):
            continue

        result[selected] = reference[selected]
        code = np.uint16(max(1, int(original_index)))
        provenance[selected] = code
        repaired_union |= selected
        source_counts[int(code)] = source_counts.get(int(code), 0) + int(np.count_nonzero(selected))
        seed_pixel_count += int(np.count_nonzero(selected & seeded_observed))
        expanded_pixel_count += int(np.count_nonzero(selected & ~seeded_observed))

    repaired_pixels = int(np.count_nonzero(repaired_union))
    return result, provenance, {
        "applied": repaired_pixels > 0,
        "reason": "exact_observed_transfer" if repaired_pixels else "no_seeded_observed_or_strong_difference",
        "verified_reference_count": int(sum(verified_slots)),
        "verified_runtime_indices": sorted(verified_runtime),
        "verified_original_indices": sorted(verified_original),
        "repaired_pixels": repaired_pixels,
        "seed_repaired_pixels": int(seed_pixel_count),
        "expanded_repaired_pixels": int(expanded_pixel_count),
        "unseeded_strong_component_pixels": int(unseeded_strong_pixel_count),
        "source_pixel_counts": source_counts,
        "difference_threshold_ceiling": float(difference_threshold),
        "threshold_diagnostics": threshold_diagnostics,
        "maximum_face_fraction": float(maximum_face_fraction),
        "difference_anchor": "frozen_imported_primary" if isinstance(workspace.metadata.get("same_canvas_imported_primary"), np.ndarray) else "runtime_primary_fallback",
        "seed_pixels_are_never_discarded_by_expansion_cap": True,
        "partial_same_canvas_supported": True,
        "interpolation": "none",
        "generated_pixels": 0,
    }


def _merge_exact_state(executor, local_provenance: np.ndarray) -> None:
    current = executor.workspace.provenance_map
    if not isinstance(current, np.ndarray) or current.shape != local_provenance.shape:
        current = np.zeros(local_provenance.shape, dtype=np.uint16)
    else:
        current = current.copy()
    used = local_provenance > 0
    current[used] = local_provenance[used]
    executor.workspace.provenance_map = current

    target = executor.workspace.metadata.get("inpaint_target_mask")
    target = _binary(target, local_provenance.shape) if isinstance(target, np.ndarray) and target.shape == local_provenance.shape else np.zeros(local_provenance.shape, dtype=np.uint8)
    target[used] = 255
    executor.workspace.metadata["inpaint_target_mask"] = target

    observed = executor.workspace.metadata.get("inpaint_observed_mask")
    observed = _binary(observed, local_provenance.shape) if isinstance(observed, np.ndarray) and observed.shape == local_provenance.shape else np.zeros(local_provenance.shape, dtype=np.uint8)
    observed[used] = 255
    executor.workspace.metadata["inpaint_observed_mask"] = observed


def _wrap_exact_repair(executor, kind: BlockKind, detail_key: str) -> None:
    original = executor._handlers.get(kind)
    if original is None:
        return

    @wraps(original)
    def handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        base_result = original(block, parameters)
        repaired, local_provenance, diagnostics = exact_same_canvas_observed_repair(
            executor.workspace,
            base_result.image,
            difference_threshold=float(parameters.get("same_canvas_difference_threshold", 0.075)),
            maximum_face_fraction=float(parameters.get("maximum_occlusion_fraction", 0.25)),
        )
        details = dict(base_result.details)
        details[detail_key] = diagnostics
        if diagnostics.get("applied"):
            _merge_exact_state(executor, local_provenance)
        return ExecutionResult(base_result.block, repaired, details)

    executor._handlers[kind] = handler


def install_same_canvas_repair_runtime(executor) -> None:
    _wrap_exact_repair(executor, BlockKind.INPAINT, "same_canvas_exact_repair")
    _wrap_exact_repair(executor, BlockKind.FUSION, "post_fusion_same_canvas_exact_repair")
