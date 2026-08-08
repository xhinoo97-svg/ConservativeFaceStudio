from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Iterator

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.pipeline import BlockKind, BlockSpec
from app.restoration import detail_reliability_map


def _binary(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    item = np.asarray(mask)
    if item.ndim == 3:
        item = cv2.cvtColor(item, cv2.COLOR_BGR2GRAY)
    if item.shape != shape:
        raise ValueError("Support/occlusion mask non compatibile")
    return np.where(item > 0, 255, 0).astype(np.uint8)


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

    # Keep blur/low-detail independent from occlusion.  A blurred region is not
    # labelled as a sticker, but it is prevented from donating identity-critical
    # texture when it has almost no observed local structure.
    reliability_threshold = int(np.clip(workspace.metadata.get("detail_reliability_threshold", 40), 0, 255))
    reliability_maps: list[np.ndarray] = []
    for reference, existing in zip(references, ref_masks):
        reliability_maps.append(detail_reliability_map(reference, existing))
    workspace.metadata["aligned_reference_detail_reliability_maps"] = [item.copy() for item in reliability_maps]
    workspace.metadata["detail_reliability_threshold"] = reliability_threshold

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
    """Make blocks 7/8 respect observed footprint and local detail reliability."""
    for kind in (BlockKind.REGION_SELECT, BlockKind.INPAINT):
        original = executor._handlers.get(kind)
        if original is None:
            continue

        def make_handler(block_kind: BlockKind, wrapped):
            @wraps(wrapped)
            def handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
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
                        "partial_identity_gate_stage": "alignment" if block_kind is BlockKind.INPAINT else None,
                    }
                )
                return ExecutionResult(result.block, result.image, details)
            return handler

        executor._handlers[kind] = make_handler(kind, original)
