from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.pipeline import BlockKind, BlockSpec
from app.reference_hint_runtime import expand_verified_single_reference_hint
from app.restoration import detail_reliability_map
from app.strict_repair import face_support_mask


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        raise ValueError("Support/occlusion mask non compatibile")
    return np.where(item > 0, 255, 0).astype(np.uint8)


def _frozen_primary_occlusion(workspace) -> np.ndarray | None:
    """Return the detector proposal measured before any learned restoration."""
    stored = workspace.metadata.get("preflight_original_occlusion_masks")
    if not isinstance(stored, list) or not stored:
        return None
    try:
        return _binary(np.asarray(stored[0]), workspace.primary.shape[:2])
    except (TypeError, ValueError):
        return None


def _merge_frozen_primary_hint(workspace) -> int:
    """Keep observed occlusion evidence from being erased by a deblur network."""
    frozen = _frozen_primary_occlusion(workspace)
    if frozen is None:
        return 0
    shape = workspace.primary.shape[:2]
    existing = workspace.metadata.get("reference_consensus_occlusion")
    if not isinstance(existing, np.ndarray) or existing.shape != shape:
        existing = np.zeros(shape, dtype=np.uint8)
    merged = cv2.bitwise_or(_binary(existing, shape), frozen)
    added = int(np.count_nonzero((merged > 0) & (existing == 0)))
    workspace.metadata["reference_consensus_occlusion"] = merged
    return added


def _effective_masks(workspace) -> tuple[list[np.ndarray] | None, int, int]:
    references = list(workspace.aligned_references)
    if not references:
        return None, 0, 0
    support_masks = workspace.metadata.get("aligned_reference_support_masks")
    if not isinstance(support_masks, list) or len(support_masks) != len(references):
        return None, 0, 0

    shape = workspace.primary.shape[:2]
    current = workspace.occlusion_masks
    if isinstance(current, list) and len(current) == len(references) + 1:
        primary_mask = _binary(current[0], shape)
        ref_masks = [_binary(item, shape) for item in current[1:]]
    else:
        primary_mask = np.zeros(shape, dtype=np.uint8)
        ref_masks = [np.zeros(shape, dtype=np.uint8) for _ in references]

    reliability_threshold = int(np.clip(workspace.metadata.get("detail_reliability_threshold", 40), 0, 255))

    # Prefer the maps frozen on the original photographs and geometrically propagated
    # during alignment. Recomputing after NAFNet would incorrectly turn network-created
    # sharpness into observed evidence. The fallback exists only for old projects that
    # predate the preflight evidence maps.
    stored_reliability = workspace.metadata.get("aligned_reference_detail_reliability_maps")
    if isinstance(stored_reliability, list) and len(stored_reliability) == len(references):
        reliability_maps = []
        for item in stored_reliability:
            array = np.asarray(item)
            if array.shape != shape:
                reliability_maps = []
                break
            reliability_maps.append(array.astype(np.uint8, copy=False))
    else:
        reliability_maps = []

    reliability_source = "pre-deblur-aligned"
    if len(reliability_maps) != len(references):
        reliability_source = "post-deblur-fallback"
        reliability_maps = [
            detail_reliability_map(reference, existing)
            for reference, existing in zip(references, ref_masks)
        ]

    workspace.metadata["aligned_reference_detail_reliability_maps"] = [item.copy() for item in reliability_maps]
    workspace.metadata["detail_reliability_threshold"] = reliability_threshold
    workspace.metadata["detail_reliability_source"] = reliability_source

    effective = [primary_mask]
    support_gated_pixels = 0
    low_detail_gated_pixels = 0
    for existing, support, reliability in zip(ref_masks, support_masks, reliability_maps):
        observed = _binary(support, shape)
        unsupported = cv2.bitwise_not(observed)
        low_detail = np.where((observed > 0) & (reliability < reliability_threshold), 255, 0).astype(np.uint8)
        support_gated_pixels += int(np.count_nonzero((unsupported > 0) & (existing == 0)))
        low_detail_gated_pixels += int(np.count_nonzero((low_detail > 0) & (existing == 0)))
        blocked = cv2.bitwise_or(unsupported, low_detail)
        effective.append(cv2.bitwise_or(existing, blocked))
    return effective, support_gated_pixels, low_detail_gated_pixels


def _expand_verified_full_reference_hint(workspace) -> dict[str, Any]:
    references = list(workspace.aligned_references)
    if len(references) != 1:
        return {"eligible": False, "reason": "not_single_reference", "added_pixels": 0}
    if not bool(workspace.metadata.get("reference_identity_verification_available", False)):
        return {"eligible": False, "reason": "identity_not_verified", "added_pixels": 0}

    shape = workspace.primary.shape[:2]
    current = workspace.occlusion_masks
    reference_mask = (
        _binary(current[1], shape)
        if isinstance(current, list) and len(current) == 2
        else np.zeros(shape, dtype=np.uint8)
    )
    bbox_raw = workspace.metadata.get("primary_bbox")
    bbox = tuple(int(v) for v in bbox_raw) if bbox_raw is not None else None
    face = face_support_mask(shape, bbox)
    existing = workspace.metadata.get("reference_consensus_occlusion")
    if not isinstance(existing, np.ndarray) or existing.shape != shape:
        existing = np.zeros(shape, dtype=np.uint8)
    expanded, diagnostics = expand_verified_single_reference_hint(
        workspace.primary,
        references[0],
        reference_mask,
        face,
        existing,
    )
    workspace.metadata["reference_consensus_occlusion"] = expanded
    return diagnostics


@contextmanager
def _temporary_partial_gate(workspace, *, disable_second_identity_gate: bool) -> Iterator[tuple[bool, int, int]]:
    effective, gated_pixels, low_detail_pixels = _effective_masks(workspace)
    old_masks = workspace.occlusion_masks
    old_identity = workspace.metadata.get("reference_identity_verification_available")
    applied = effective is not None
    try:
        if effective is not None:
            workspace.occlusion_masks = effective
        if disable_second_identity_gate:
            workspace.metadata["reference_identity_verification_available"] = False
        yield applied, gated_pixels, low_detail_pixels
    finally:
        workspace.occlusion_masks = old_masks
        if old_identity is None:
            workspace.metadata.pop("reference_identity_verification_available", None)
        else:
            workspace.metadata["reference_identity_verification_available"] = old_identity


def install_partial_reference_runtime(executor) -> None:
    """Make blocks 7/8 respect observed footprint and original-detail reliability."""
    for kind in (BlockKind.REGION_SELECT, BlockKind.INPAINT):
        original = executor._handlers.get(kind)
        if original is None:
            continue

        def make_handler(block_kind: BlockKind, wrapped):
            @wraps(wrapped)
            def handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
                frozen_hint_pixels = (
                    _merge_frozen_primary_hint(executor.workspace)
                    if block_kind is BlockKind.INPAINT
                    else 0
                )
                hint_diagnostics = (
                    _expand_verified_full_reference_hint(executor.workspace)
                    if block_kind is BlockKind.INPAINT
                    else {"eligible": False, "reason": "not_inpaint", "added_pixels": 0}
                )
                with _temporary_partial_gate(
                    executor.workspace,
                    disable_second_identity_gate=block_kind is BlockKind.INPAINT,
                ) as (applied, gated_pixels, low_detail_pixels):
                    result = wrapped(block, parameters)
                details = dict(result.details)
                details.update(
                    {
                        "partial_reference_support_gate": bool(applied),
                        "unsupported_reference_pixels_blocked": int(gated_pixels),
                        "low_detail_reference_pixels_blocked": int(low_detail_pixels),
                        "detail_reliability_threshold": int(executor.workspace.metadata.get("detail_reliability_threshold", 40)),
                        "detail_reliability_source": str(executor.workspace.metadata.get("detail_reliability_source", "unknown")),
                        "partial_identity_gate_stage": "alignment" if block_kind is BlockKind.INPAINT else None,
                        "frozen_primary_hint_added_pixels": int(frozen_hint_pixels),
                        "verified_single_reference_hint": hint_diagnostics,
                    }
                )
                return ExecutionResult(result.block, result.image, details)
            return handler

        executor._handlers[kind] = make_handler(kind, original)
