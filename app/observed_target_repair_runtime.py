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


def _trusted_slots(workspace, count: int) -> list[bool]:
    identity = workspace.metadata.get("aligned_reference_identity_verified")
    partial = workspace.metadata.get("aligned_reference_partial_geometry_verified")
    identity_flags = [False] * count
    partial_flags = [False] * count
    if isinstance(identity, list) and len(identity) == count:
        identity_flags = [bool(value) for value in identity]
    if isinstance(partial, list) and len(partial) == count:
        partial_flags = [bool(value) for value in partial]
    return [a or b for a, b in zip(identity_flags, partial_flags)]


def repair_observed_target(
    workspace,
    image: np.ndarray,
    *,
    minimum_reliability: int = 96,
    agreement_colour_threshold: float = 24.0,
    maximum_face_fraction: float = 0.40,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill damaged pixels from trusted, actually observed aligned donors only.

    This pass is deliberately asymmetric: damaged target pixels may be replaced when
    trusted observed evidence exists; visible primary pixels are never touched. It does
    not synthesize, mirror, sharpen, or extrapolate donor data.
    """
    shape = workspace.primary.shape[:2]
    aligned = list(workspace.aligned_references)
    if not aligned:
        return image.copy(), np.zeros(shape, dtype=np.uint16), {
            "applied": False, "reason": "no_aligned_references", "repaired_pixels": 0,
        }

    trusted = _trusted_slots(workspace, len(aligned))
    if not any(trusted):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {
            "applied": False, "reason": "no_trusted_aligned_reference", "repaired_pixels": 0,
        }

    target = _target_mask(workspace, shape) > 0
    if not np.any(target):
        return image.copy(), np.zeros(shape, dtype=np.uint16), {
            "applied": False, "reason": "no_damage_target", "repaired_pixels": 0,
        }

    supports_raw = workspace.metadata.get("aligned_reference_support_masks")
    reliabilities_raw = workspace.metadata.get("aligned_reference_detail_reliability_maps")
    originals_raw = workspace.metadata.get("aligned_reference_original_source_indices")
    supports = (
        [np.asarray(value) for value in supports_raw]
        if isinstance(supports_raw, list) and len(supports_raw) == len(aligned)
        else [np.full(shape, 255, dtype=np.uint8) for _ in aligned]
    )
    reliabilities = (
        [np.asarray(value) for value in reliabilities_raw]
        if isinstance(reliabilities_raw, list) and len(reliabilities_raw) == len(aligned)
        else [np.full(shape, 255, dtype=np.uint8) for _ in aligned]
    )
    originals = (
        [max(1, int(value)) for value in originals_raw]
        if isinstance(originals_raw, list) and len(originals_raw) == len(aligned)
        else [index + 1 for index in range(len(aligned))]
    )

    bbox_raw = workspace.metadata.get("primary_bbox")
    bbox = tuple(int(v) for v in bbox_raw) if bbox_raw is not None else None
    face = face_support_mask(shape, bbox) > 0
    target &= face
    face_pixels = max(1, int(np.count_nonzero(face)))
    maximum_pixels = int(round(face_pixels * float(maximum_face_fraction)))

    valid_masks: list[np.ndarray] = []
    reliable_maps: list[np.ndarray] = []
    used_images: list[np.ndarray] = []
    used_codes: list[int] = []
    for slot, (reference, support_raw, reliability_raw, code) in enumerate(
        zip(aligned, supports, reliabilities, originals)
    ):
        if not trusted[slot] or reference.shape != workspace.primary.shape:
            continue
        try:
            support = _binary(support_raw, shape) > 0
        except ValueError:
            continue
        reliability = np.asarray(reliability_raw)
        if reliability.shape != shape:
            reliability = np.full(shape, 255, dtype=np.uint8)
        reliability = reliability.astype(np.float32)
        observed = support & (np.max(reference, axis=2) > 2)
        valid = target & observed & (reliability >= float(minimum_reliability))
        if not np.any(valid):
            continue
        valid_masks.append(valid)
        reliable_maps.append(reliability)
        used_images.append(reference)
        used_codes.append(code)

    if not used_images:
        return image.copy(), np.zeros(shape, dtype=np.uint16), {
            "applied": False, "reason": "trusted_references_do_not_cover_target", "repaired_pixels": 0,
        }

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
                accepted[overlap] &= gap[overlap] <= float(agreement_colour_threshold)

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
        return image.copy(), np.zeros(shape, dtype=np.uint16), {
            "applied": False, "reason": "agreement_or_fraction_gate_rejected_target", "repaired_pixels": 0,
        }

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
    return result, provenance, {
        "applied": repaired_pixels > 0,
        "reason": "trusted_observed_target_transfer",
        "trusted_reference_count": len(used_images),
        "repaired_pixels": repaired_pixels,
        "target_pixels": int(np.count_nonzero(target)),
        "minimum_reliability": int(minimum_reliability),
        "agreement_colour_threshold": float(agreement_colour_threshold),
        "maximum_face_fraction": float(maximum_face_fraction),
        "generated_pixels": 0,
        "visible_primary_pixels_modified": 0,
    }


def install_observed_target_repair_runtime(executor) -> None:
    original = executor._handlers.get(BlockKind.INPAINT)
    if original is None:
        return

    @wraps(original)
    def handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        base_result = original(block, parameters)
        repaired, local_provenance, diagnostics = repair_observed_target(
            executor.workspace,
            base_result.image,
            minimum_reliability=int(parameters.get("observed_target_minimum_reliability", 96)),
            agreement_colour_threshold=float(parameters.get("observed_target_agreement_colour_threshold", 24.0)),
            maximum_face_fraction=float(parameters.get("observed_target_maximum_face_fraction", 0.40)),
        )
        details = dict(base_result.details)
        details["observed_target_repair"] = diagnostics
        if diagnostics.get("applied"):
            current = executor.workspace.provenance_map
            if not isinstance(current, np.ndarray) or current.shape != local_provenance.shape:
                current = np.zeros(local_provenance.shape, dtype=np.uint16)
            else:
                current = current.copy()
            used = local_provenance > 0
            current[used] = local_provenance[used]
            executor.workspace.provenance_map = current

            observed = executor.workspace.metadata.get("inpaint_observed_mask")
            if isinstance(observed, np.ndarray) and observed.shape == local_provenance.shape:
                observed = _binary(observed, local_provenance.shape)
            else:
                observed = np.zeros(local_provenance.shape, dtype=np.uint8)
            observed[used] = 255
            executor.workspace.metadata["inpaint_observed_mask"] = observed

            target = executor.workspace.metadata.get("inpaint_target_mask")
            if isinstance(target, np.ndarray) and target.shape == local_provenance.shape:
                target = _binary(target, local_provenance.shape)
            else:
                target = np.zeros(local_provenance.shape, dtype=np.uint8)
            target[used] = 255
            executor.workspace.metadata["inpaint_target_mask"] = target
        return ExecutionResult(base_result.block, repaired, details)

    executor._handlers[BlockKind.INPAINT] = handler
