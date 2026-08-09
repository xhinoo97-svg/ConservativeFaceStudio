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


def _target_mask(workspace, shape: tuple[int, int]) -> np.ndarray:
    target = np.zeros(shape, dtype=np.uint8)
    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    if isinstance(frozen, list) and frozen:
        try:
            target = cv2.bitwise_or(target, _binary(np.asarray(frozen[0]), shape))
        except (TypeError, ValueError):
            pass
    masks = workspace.occlusion_masks
    if isinstance(masks, list) and masks:
        try:
            target = cv2.bitwise_or(target, _binary(np.asarray(masks[0]), shape))
        except (TypeError, ValueError):
            pass
    current = workspace.metadata.get("inpaint_target_mask")
    if isinstance(current, np.ndarray) and current.shape == shape:
        target = cv2.bitwise_or(target, _binary(current, shape))
    return target


def _aligned_original_indices(workspace, count: int) -> list[int]:
    explicit = workspace.metadata.get("aligned_reference_original_source_indices")
    if isinstance(explicit, list) and len(explicit) == count:
        try:
            return [max(1, int(value)) for value in explicit]
        except (TypeError, ValueError):
            pass
    runtime_indices_raw = workspace.metadata.get("aligned_reference_source_indices")
    runtime_indices = (
        [int(value) for value in runtime_indices_raw]
        if isinstance(runtime_indices_raw, list) and len(runtime_indices_raw) == count
        else list(range(count))
    )
    runtime_order_raw = workspace.metadata.get("runtime_source_order")
    runtime_order = (
        [int(value) for value in runtime_order_raw]
        if isinstance(runtime_order_raw, list) and len(runtime_order_raw) >= 2
        else None
    )
    resolved: list[int] = []
    for runtime_reference_index in runtime_indices:
        original = runtime_reference_index + 1
        if runtime_order is not None:
            runtime_slot = runtime_reference_index + 1
            if 0 <= runtime_slot < len(runtime_order):
                original = int(runtime_order[runtime_slot])
        resolved.append(max(1, int(original)))
    return resolved


def _trusted_slots(workspace, count: int) -> list[bool]:
    identity = workspace.metadata.get("aligned_reference_identity_verified")
    partial = workspace.metadata.get("aligned_reference_partial_geometry_verified")
    identity_flags = [False] * count
    partial_flags = [False] * count
    if isinstance(identity, list) and len(identity) == count:
        identity_flags = [bool(value) for value in identity]
    if isinstance(partial, list) and len(partial) == count:
        partial_flags = [bool(value) for value in partial]

    runtime_indices_raw = workspace.metadata.get("aligned_reference_source_indices")
    runtime_indices = (
        [int(value) for value in runtime_indices_raw]
        if isinstance(runtime_indices_raw, list) and len(runtime_indices_raw) == count
        else list(range(count))
    )
    original_indices = _aligned_original_indices(workspace, count)

    same_canvas = workspace.metadata.get("verified_same_canvas_alignment")
    same_canvas_runtime = {
        int(item.get("runtime_reference_index"))
        for item in same_canvas
        if isinstance(item, dict)
        and item.get("method") == "verified-same-canvas-observed"
        and item.get("runtime_reference_index") is not None
    } if isinstance(same_canvas, list) else set()

    anchor = workspace.metadata.get("same_canvas_primary_anchor")
    anchor_original = {
        int(value) for value in anchor.get("matched_original_reference_indices", [])
    } if isinstance(anchor, dict) and isinstance(anchor.get("matched_original_reference_indices"), list) else set()

    candidates = workspace.metadata.get("preflight_candidates")
    accepted_original: set[int] = set()
    if isinstance(candidates, list):
        for item in candidates:
            if not isinstance(item, dict) or not bool(item.get("accepted_identity", False)):
                continue
            try:
                accepted_original.add(int(item.get("source_index")))
            except (TypeError, ValueError):
                continue

    return [
        identity_flags[index]
        or partial_flags[index]
        or runtime_indices[index] in same_canvas_runtime
        or original_indices[index] in anchor_original
        or original_indices[index] in accepted_original
        for index in range(count)
    ]


def repair_observed_target(
    workspace,
    image: np.ndarray,
    *,
    minimum_reliability: int = 0,
    agreement_colour_threshold: float = 24.0,
    maximum_face_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill damaged pixels from trusted, actually observed aligned donors only.

    The replacement limit applies only as a safety ceiling over the face. The default
    is intentionally 1.0: a verified damage target may legitimately cover most of the
    face, and reference coverage must not be truncated merely because the occlusion is
    large. Pixels outside the frozen damage target are never eligible here.

    ``aligned_reference_support_masks`` is the authoritative observed footprint. Pixel
    intensity is deliberately not used as a support test: genuine photographed pixels
    can be exactly or nearly black (hair, pupils, eyeliner, deep shadow). Warp borders
    and crop padding must instead be excluded by the geometric support mask so exact
    provenance is preserved without introducing appearance-dependent priors.
    """
    shape = workspace.primary.shape[:2]
    aligned = list(workspace.aligned_references)
    if not aligned:
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "no_aligned_references", "repaired_pixels": 0}

    trusted = _trusted_slots(workspace, len(aligned))
    if not any(trusted):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "no_trusted_aligned_reference", "repaired_pixels": 0}

    target = _target_mask(workspace, shape) > 0
    if not np.any(target):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "no_damage_target", "repaired_pixels": 0}

    supports_raw = workspace.metadata.get("aligned_reference_support_masks")
    reliabilities_raw = workspace.metadata.get("aligned_reference_detail_reliability_maps")
    supports = [np.asarray(value) for value in supports_raw] if isinstance(supports_raw, list) and len(supports_raw) == len(aligned) else [np.full(shape, 255, np.uint8) for _ in aligned]
    reliabilities = [np.asarray(value) for value in reliabilities_raw] if isinstance(reliabilities_raw, list) and len(reliabilities_raw) == len(aligned) else [np.full(shape, 255, np.uint8) for _ in aligned]
    originals = _aligned_original_indices(workspace, len(aligned))

    bbox_raw = workspace.metadata.get("primary_bbox")
    bbox = tuple(int(v) for v in bbox_raw) if bbox_raw is not None else None
    face = face_support_mask(shape, bbox) > 0
    target &= face
    target_pixels = int(np.count_nonzero(target))
    face_pixels = max(1, int(np.count_nonzero(face)))
    maximum_pixels = min(target_pixels, int(round(face_pixels * float(np.clip(maximum_face_fraction, 0.0, 1.0)))))

    valid_masks: list[np.ndarray] = []
    reliable_maps: list[np.ndarray] = []
    used_images: list[np.ndarray] = []
    used_codes: list[int] = []
    threshold = float(max(0, int(minimum_reliability)))
    for slot, (reference, support_raw, reliability_raw, code) in enumerate(zip(aligned, supports, reliabilities, originals)):
        if not trusted[slot] or reference.shape != workspace.primary.shape:
            continue
        try:
            support = _binary(support_raw, shape) > 0
        except ValueError:
            continue
        reliability = np.asarray(reliability_raw)
        if reliability.shape != shape:
            reliability = np.full(shape, 255, dtype=np.uint8)
        reliability = np.nan_to_num(
            reliability.astype(np.float32),
            nan=-np.inf,
            posinf=255.0,
            neginf=-np.inf,
        )
        observed = support
        valid = target & observed & (reliability >= threshold)
        if not np.any(valid):
            continue
        valid_masks.append(valid)
        reliable_maps.append(reliability)
        used_images.append(reference)
        used_codes.append(code)

    if not used_images:
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "trusted_references_do_not_cover_target", "repaired_pixels": 0, "target_pixels": target_pixels, "damage_reference_coverage": 0.0}

    valid_stack = np.stack(valid_masks, axis=0)
    reliability_stack = np.stack(reliable_maps, axis=0)
    reliability_stack[~valid_stack] = -np.inf
    accepted = np.any(valid_stack, axis=0)

    if len(used_images) > 1:
        labs = [cv2.cvtColor(item, cv2.COLOR_BGR2LAB).astype(np.float32) for item in used_images]
        for first in range(len(used_images) - 1):
            for second in range(first + 1, len(used_images)):
                overlap = valid_stack[first] & valid_stack[second]
                if not np.any(overlap):
                    continue
                gap = np.mean(np.abs(labs[first] - labs[second]), axis=2)
                conflict = overlap & (gap > float(agreement_colour_threshold))
                if not np.any(conflict):
                    continue
                reliability_gap = np.zeros(shape, dtype=np.float32)
                np.subtract(
                    reliability_stack[first],
                    reliability_stack[second],
                    out=reliability_gap,
                    where=overlap,
                )
                np.abs(reliability_gap, out=reliability_gap)
                ambiguous_conflict = conflict & (reliability_gap < 12.0)
                accepted[ambiguous_conflict] = False

    if maximum_pixels >= 0 and np.count_nonzero(accepted) > maximum_pixels:
        best_reliability = np.max(reliability_stack, axis=0)
        coords = np.flatnonzero(accepted)
        keep_count = max(0, maximum_pixels)
        limited = np.zeros(shape, dtype=bool)
        if keep_count > 0:
            values = best_reliability.ravel()[coords]
            keep = coords[np.argpartition(values, -keep_count)[-keep_count:]]
            limited.ravel()[keep] = True
        accepted = limited

    if not np.any(accepted):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {"applied": False, "reason": "agreement_or_fraction_gate_rejected_target", "repaired_pixels": 0, "target_pixels": target_pixels, "damage_reference_coverage": 0.0}

    best_slot = np.argmax(reliability_stack, axis=0)
    rows, cols = np.indices(shape)
    stack = np.stack(used_images, axis=0)
    chosen = stack[best_slot, rows, cols]
    result = image.copy()
    result[accepted] = chosen[accepted]

    code_array = np.asarray(used_codes, dtype=np.uint16)
    provenance = np.zeros(shape, dtype=np.uint16)
    provenance[accepted] = code_array[best_slot[accepted]]
    repaired_pixels = int(np.count_nonzero(accepted))
    coverage = float(repaired_pixels / max(1, target_pixels))
    return result, provenance, {
        "applied": repaired_pixels > 0,
        "reason": "trusted_observed_target_transfer",
        "trusted_reference_count": len(used_images),
        "trusted_slots": [bool(value) for value in trusted],
        "original_source_indices": [int(value) for value in originals],
        "repaired_pixels": repaired_pixels,
        "target_pixels": target_pixels,
        "damage_reference_coverage": coverage,
        "uncovered_damage_fraction": float(1.0 - coverage),
        "minimum_reliability": int(minimum_reliability),
        "reliability_role": "donor_ranking_with_optional_manual_floor",
        "observed_support_source": "aligned_reference_support_masks",
        "agreement_colour_threshold": float(agreement_colour_threshold),
        "maximum_face_fraction": float(maximum_face_fraction),
        "generated_pixels": 0,
        "visible_primary_pixels_modified": 0,
    }


def _merge_runtime_state(executor, local_provenance: np.ndarray) -> None:
    current = executor.workspace.provenance_map
    if not isinstance(current, np.ndarray) or current.shape != local_provenance.shape:
        current = np.zeros(local_provenance.shape, dtype=np.uint16)
    else:
        current = current.copy()
    used = local_provenance > 0
    current[used] = local_provenance[used]
    executor.workspace.provenance_map = current

    observed = executor.workspace.metadata.get("inpaint_observed_mask")
    observed = _binary(observed, local_provenance.shape) if isinstance(observed, np.ndarray) and observed.shape == local_provenance.shape else np.zeros(local_provenance.shape, np.uint8)
    observed[used] = 255
    executor.workspace.metadata["inpaint_observed_mask"] = observed

    target = executor.workspace.metadata.get("inpaint_target_mask")
    target = _binary(target, local_provenance.shape) if isinstance(target, np.ndarray) and target.shape == local_provenance.shape else np.zeros(local_provenance.shape, np.uint8)
    target[used] = 255
    executor.workspace.metadata["inpaint_target_mask"] = target


def _restore_outside_target(executor, image: np.ndarray, preservation_anchor: np.ndarray) -> tuple[np.ndarray, int]:
    shape = image.shape[:2]
    if preservation_anchor.shape != image.shape:
        return image, 0
    target = _target_mask(executor.workspace, shape) > 0
    if not np.any(target):
        return image, 0
    outside = ~target
    changed = np.max(cv2.absdiff(image, preservation_anchor), axis=2) > 0
    restored_pixels = int(np.count_nonzero(outside & changed))
    if restored_pixels == 0:
        return image, 0
    result = image.copy()
    result[outside] = preservation_anchor[outside]

    provenance = executor.workspace.provenance_map
    if isinstance(provenance, np.ndarray) and provenance.shape == shape:
        provenance = provenance.copy()
        provenance[outside] = 0
        executor.workspace.provenance_map = provenance
    observed = executor.workspace.metadata.get("inpaint_observed_mask")
    if isinstance(observed, np.ndarray) and observed.shape == shape:
        observed = _binary(observed, shape)
        observed[outside] = 0
        executor.workspace.metadata["inpaint_observed_mask"] = observed
    symmetry = executor.workspace.metadata.get("inpaint_symmetry_mask")
    if isinstance(symmetry, np.ndarray) and symmetry.shape == shape:
        symmetry = _binary(symmetry, shape)
        symmetry[outside] = 0
        executor.workspace.metadata["inpaint_symmetry_mask"] = symmetry
    generated = executor.workspace.metadata.get("inpaint_generated_mask")
    if isinstance(generated, np.ndarray) and generated.shape == shape:
        generated = _binary(generated, shape)
        generated[outside] = 0
        executor.workspace.metadata["inpaint_generated_mask"] = generated
    return result, restored_pixels


def _wrap_target_repair(executor, kind: BlockKind, detail_key: str, preservation_anchor: np.ndarray) -> None:
    original = executor._handlers.get(kind)
    if original is None:
        return

    @wraps(original)
    def handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        base_result = original(block, parameters)
        repaired, local_provenance, diagnostics = repair_observed_target(
            executor.workspace,
            base_result.image,
            minimum_reliability=int(parameters.get("observed_target_minimum_reliability", 0)),
            agreement_colour_threshold=float(parameters.get("observed_target_agreement_colour_threshold", 24.0)),
            maximum_face_fraction=float(parameters.get("observed_target_maximum_face_fraction", 1.0)),
        )
        if diagnostics.get("applied"):
            _merge_runtime_state(executor, local_provenance)
        repaired, restored_pixels = _restore_outside_target(executor, repaired, preservation_anchor)
        diagnostics = dict(diagnostics)
        diagnostics["outside_target_restored_pixels"] = int(restored_pixels)
        diagnostics["outside_target_preservation"] = "exact_imported_primary" if restored_pixels else "unchanged"
        details = dict(base_result.details)
        details[detail_key] = diagnostics
        return ExecutionResult(base_result.block, repaired, details)

    executor._handlers[kind] = handler


def install_observed_target_repair_runtime(executor) -> None:
    preservation_anchor = executor.workspace.primary.copy()
    _wrap_target_repair(executor, BlockKind.INPAINT, "observed_target_repair", preservation_anchor)
    _wrap_target_repair(executor, BlockKind.FUSION, "post_fusion_observed_target_repair", preservation_anchor)
