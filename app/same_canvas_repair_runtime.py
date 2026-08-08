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


def _seeded_components(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    binary = np.where(candidate, 255, 0).astype(np.uint8)
    if not np.any(binary) or not np.any(seed):
        return np.zeros(binary.shape, dtype=bool)
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    keep = np.zeros(binary.shape, dtype=bool)
    seed_bool = seed > 0
    for label in range(1, count):
        component = labels == label
        if np.any(component & seed_bool):
            keep |= component
    return keep


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
    diagnostics = workspace.metadata.get("verified_same_canvas_alignment")
    verified_runtime = {
        int(item.get("runtime_reference_index"))
        for item in diagnostics
        if isinstance(item, dict)
        and item.get("method") == "verified-same-canvas-observed"
        and item.get("runtime_reference_index") is not None
    } if isinstance(diagnostics, list) else set()

    anchor = workspace.metadata.get("same_canvas_primary_anchor")
    verified_original = {
        int(value)
        for value in anchor.get("matched_original_reference_indices", [])
    } if isinstance(anchor, dict) and isinstance(anchor.get("matched_original_reference_indices"), list) else set()
    return verified_runtime, verified_original


def exact_same_canvas_observed_repair(
    workspace,
    image: np.ndarray,
    *,
    difference_threshold: float = 0.075,
    maximum_face_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Replace damage from donors whose same-canvas geometry was already verified.

    Detector pixels are transferred directly. LAB-difference components can expand the
    seed only when they touch it and remain inside verified observed donor support. No
    synthesis, mirroring or interpolation is used by this pass.
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

    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and frozen:
        try:
            seed = _binary(np.asarray(frozen[0]), shape)
        except (TypeError, ValueError):
            seed = np.zeros(shape, dtype=np.uint8)
    else:
        seed = np.zeros(shape, dtype=np.uint8)
    masks = workspace.occlusion_masks
    if isinstance(masks, list) and masks:
        try:
            seed = cv2.bitwise_or(seed, _binary(np.asarray(masks[0]), shape))
        except (TypeError, ValueError):
            pass
    if not np.any(seed):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "no_observed_damage_seed", "repaired_pixels": 0}

    bbox_raw = workspace.metadata.get("primary_bbox")
    bbox = tuple(int(v) for v in bbox_raw) if bbox_raw is not None else None
    face = face_support_mask(shape, bbox) > 0
    face_pixels = max(1, int(np.count_nonzero(face)))
    seed_bool = seed > 0
    seed_reach = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    base_lab = cv2.cvtColor(workspace.primary, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0

    result = image.copy()
    provenance = np.zeros(shape, dtype=np.uint16)
    repaired_union = np.zeros(shape, dtype=bool)
    source_counts: dict[int, int] = {}

    for slot, (reference, support_raw, original_index) in enumerate(zip(aligned, supports_raw, originals)):
        if not verified_slots[slot] or reference.shape != workspace.primary.shape:
            continue
        support = _binary(np.asarray(support_raw), shape) > 0
        if not np.any(support):
            continue
        reference_valid = np.max(reference, axis=2) > 2
        observed = support & reference_valid & face
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
        difference = np.mean(np.abs(base_lab - ref_lab), axis=2)
        strong = observed & (difference >= float(difference_threshold))
        strong_mask = strong.astype(np.uint8) * 255
        if np.any(strong):
            strong_mask = cv2.morphologyEx(
                strong_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            )
            strong = (strong_mask > 0) & observed

        seeded_observed = seed_bool & observed
        selected = seeded_observed | _seeded_components(strong, seed_reach)
        selected &= ~repaired_union
        if not np.any(selected):
            continue
        proposed_total = int(np.count_nonzero(repaired_union | selected))
        if proposed_total > int(round(face_pixels * float(maximum_face_fraction))):
            continue
        result[selected] = reference[selected]
        code = np.uint16(max(1, int(original_index)))
        provenance[selected] = code
        repaired_union |= selected
        source_counts[int(code)] = source_counts.get(int(code), 0) + int(np.count_nonzero(selected))

    repaired_pixels = int(np.count_nonzero(repaired_union))
    return result, provenance, {
        "applied": repaired_pixels > 0,
        "reason": "exact_observed_transfer" if repaired_pixels else "no_seeded_observed_or_strong_difference",
        "verified_reference_count": int(sum(verified_slots)),
        "verified_runtime_indices": sorted(verified_runtime),
        "verified_original_indices": sorted(verified_original),
        "repaired_pixels": repaired_pixels,
        "source_pixel_counts": source_counts,
        "difference_threshold": float(difference_threshold),
        "maximum_face_fraction": float(maximum_face_fraction),
        "seeded_pixels_require_difference_threshold": False,
        "strong_difference_expands_seed": True,
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
    # FUSION may overwrite pixels repaired at INPAINT. Apply the exact observed donor
    # transfer again after FUSION so the identity guardrail evaluates the preserved
    # evidence rather than a later blend that erased it.
    _wrap_exact_repair(executor, BlockKind.INPAINT, "same_canvas_exact_repair")
    _wrap_exact_repair(executor, BlockKind.FUSION, "post_fusion_same_canvas_exact_repair")
