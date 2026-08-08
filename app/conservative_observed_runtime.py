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


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def verify_same_canvas_observed_source(
    workspace,
    reference: np.ndarray,
    runtime_reference_index: int,
    *,
    minimum_observed_fraction: float = 0.02,
    maximum_median_lab_delta: float = 0.035,
    maximum_p90_lab_delta: float = 0.12,
    maximum_edge_median_delta: float = 8.0,
    maximum_edge_p90_delta: float = 42.0,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    """Verify that a reference already uses the primary image coordinate system.

    Equal dimensions are never enough. The identity transform is accepted only when
    non-damaged, actually observed face pixels agree in colour and in edge geometry.
    This avoids an unnecessary affine warp for exact full-image, complementary-crop,
    and component references while rejecting merely same-sized unrelated images.
    """
    primary = workspace.primary
    if reference.shape != primary.shape or reference.ndim != 3 or reference.shape[2] != 3:
        return None
    shape = primary.shape[:2]
    observed = np.max(reference, axis=2) > 2
    observed_fraction = float(np.mean(observed))
    if observed_fraction < float(minimum_observed_fraction):
        return None

    bbox_raw = workspace.metadata.get("primary_bbox")
    if bbox_raw is None:
        return None
    try:
        bbox = tuple(int(v) for v in bbox_raw)
    except (TypeError, ValueError):
        return None
    face = face_support_mask(shape, bbox) > 0

    frozen = workspace.metadata.get("preflight_original_occlusion_masks")
    primary_occ = np.zeros(shape, dtype=np.uint8)
    reference_occ = np.zeros(shape, dtype=np.uint8)
    if isinstance(frozen, list):
        try:
            if frozen:
                primary_occ = _binary(np.asarray(frozen[0]), shape)
            if runtime_reference_index + 1 < len(frozen):
                reference_occ = _binary(np.asarray(frozen[runtime_reference_index + 1]), shape)
        except (TypeError, ValueError):
            return None

    # Sobel support extends beyond an occlusion by a few pixels. Exclude a narrow
    # verification-only ring so a real sticker edge does not look like a geometric
    # misalignment. The original support mask is returned unchanged for later repair.
    boundary_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    primary_blocked = cv2.dilate(primary_occ, boundary_kernel) > 0
    reference_blocked = cv2.dilate(reference_occ, boundary_kernel) > 0

    comparable = (
        observed
        & face
        & ~primary_blocked
        & ~reference_blocked
        & (np.max(primary, axis=2) > 2)
    )
    observed_face_pixels = int(np.count_nonzero(observed & face & ~reference_blocked))
    comparable_pixels = int(np.count_nonzero(comparable))
    minimum_comparable = max(96, int(round(observed_face_pixels * 0.10)))
    if observed_face_pixels <= 0 or comparable_pixels < minimum_comparable:
        return None

    base_lab = cv2.cvtColor(primary, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    colour_delta = np.mean(np.abs(base_lab - ref_lab), axis=2)[comparable]
    median_lab = float(np.median(colour_delta))
    p90_lab = float(np.percentile(colour_delta, 90.0))
    if median_lab > float(maximum_median_lab_delta) or p90_lab > float(maximum_p90_lab_delta):
        return None

    base_grad = _gradient_magnitude(primary)
    ref_grad = _gradient_magnitude(reference)
    edge_pixels = comparable & ((base_grad >= 12.0) | (ref_grad >= 12.0))
    edge_count = int(np.count_nonzero(edge_pixels))
    edge_median = 0.0
    edge_p90 = 0.0
    if edge_count >= 48:
        edge_delta = np.abs(base_grad[edge_pixels] - ref_grad[edge_pixels])
        edge_median = float(np.median(edge_delta))
        edge_p90 = float(np.percentile(edge_delta, 90.0))
        if edge_median > float(maximum_edge_median_delta) or edge_p90 > float(maximum_edge_p90_delta):
            return None

    support = observed.astype(np.uint8) * 255
    return support, {
        "runtime_reference_index": int(runtime_reference_index),
        "method": "verified-same-canvas-observed",
        "observed_fraction": observed_fraction,
        "observed_face_pixels": observed_face_pixels,
        "comparable_pixels": comparable_pixels,
        "median_lab_delta": median_lab,
        "p90_lab_delta": p90_lab,
        "edge_pixels": edge_count,
        "edge_median_delta": edge_median,
        "edge_p90_delta": edge_p90,
    }


def _restore_exact_same_canvas_references(workspace) -> list[dict[str, Any]]:
    references = list(workspace.references)
    if not references:
        return []

    aligned = list(workspace.aligned_references)
    runtime_indices_raw = workspace.metadata.get("aligned_reference_source_indices")
    runtime_indices = (
        [int(value) for value in runtime_indices_raw]
        if isinstance(runtime_indices_raw, list) and len(runtime_indices_raw) == len(aligned)
        else list(range(len(aligned)))
    )
    supports_raw = workspace.metadata.get("aligned_reference_support_masks")
    supports = (
        [np.asarray(value).copy() for value in supports_raw]
        if isinstance(supports_raw, list) and len(supports_raw) == len(aligned)
        else [np.full(workspace.primary.shape[:2], 255, dtype=np.uint8) for _ in aligned]
    )
    reliability_raw = workspace.metadata.get("aligned_reference_detail_reliability_maps")
    reliability = (
        [np.asarray(value).copy() for value in reliability_raw]
        if isinstance(reliability_raw, list) and len(reliability_raw) == len(aligned)
        else [np.full(workspace.primary.shape[:2], 255, dtype=np.uint8) for _ in aligned]
    )
    original_indices_raw = workspace.metadata.get("aligned_reference_original_source_indices")
    original_indices = (
        [int(value) for value in original_indices_raw]
        if isinstance(original_indices_raw, list) and len(original_indices_raw) == len(aligned)
        else [value + 1 for value in runtime_indices]
    )
    identity_scores_raw = workspace.metadata.get("aligned_reference_identity_scores")
    identity_scores = (
        list(identity_scores_raw)
        if isinstance(identity_scores_raw, list) and len(identity_scores_raw) == len(aligned)
        else [None] * len(aligned)
    )
    identity_verified_raw = workspace.metadata.get("aligned_reference_identity_verified")
    identity_verified = (
        [bool(value) for value in identity_verified_raw]
        if isinstance(identity_verified_raw, list) and len(identity_verified_raw) == len(aligned)
        else [False] * len(aligned)
    )
    partial_verified_raw = workspace.metadata.get("aligned_reference_partial_geometry_verified")
    partial_verified = (
        [bool(value) for value in partial_verified_raw]
        if isinstance(partial_verified_raw, list) and len(partial_verified_raw) == len(aligned)
        else [False] * len(aligned)
    )
    runtime_order_raw = workspace.metadata.get("runtime_source_order")
    runtime_order = (
        [int(value) for value in runtime_order_raw]
        if isinstance(runtime_order_raw, list) and len(runtime_order_raw) == len(references) + 1
        else list(range(len(references) + 1))
    )
    frozen_reliability = workspace.metadata.get("preflight_detail_reliability_maps")

    slot_by_runtime = {runtime_index: slot for slot, runtime_index in enumerate(runtime_indices)}
    diagnostics: list[dict[str, Any]] = []
    for runtime_index, reference in enumerate(references):
        verified = verify_same_canvas_observed_source(workspace, reference, runtime_index)
        if verified is None:
            continue
        support, details = verified
        source_reliability = np.full(workspace.primary.shape[:2], 255, dtype=np.uint8)
        if isinstance(frozen_reliability, list) and runtime_index + 1 < len(frozen_reliability):
            candidate = np.asarray(frozen_reliability[runtime_index + 1])
            if candidate.shape == workspace.primary.shape[:2]:
                source_reliability = candidate.astype(np.uint8, copy=True)
        source_reliability[support == 0] = 0

        slot = slot_by_runtime.get(runtime_index)
        if slot is None:
            slot = len(aligned)
            slot_by_runtime[runtime_index] = slot
            aligned.append(reference.copy())
            supports.append(support)
            reliability.append(source_reliability)
            runtime_indices.append(runtime_index)
            original_indices.append(int(runtime_order[runtime_index + 1]))
            identity_scores.append(None)
            identity_verified.append(False)
            partial_verified.append(True)
            details["action"] = "supplemented"
        else:
            aligned[slot] = reference.copy()
            supports[slot] = support
            reliability[slot] = source_reliability
            partial_verified[slot] = True
            details["action"] = "restored-exact-identity-transform"
        diagnostics.append(details)

    if diagnostics:
        workspace.aligned_references = aligned
        workspace.metadata["aligned_reference_support_masks"] = supports
        workspace.metadata["aligned_reference_detail_reliability_maps"] = reliability
        workspace.metadata["aligned_reference_source_indices"] = runtime_indices
        workspace.metadata["aligned_reference_original_source_indices"] = original_indices
        workspace.metadata["aligned_reference_identity_scores"] = identity_scores
        workspace.metadata["aligned_reference_identity_verified"] = identity_verified
        workspace.metadata["aligned_reference_partial_geometry_verified"] = partial_verified
    workspace.metadata["verified_same_canvas_alignment"] = diagnostics
    return diagnostics


def install_conservative_observed_runtime(executor) -> None:
    """Keep observed geometry exact and prevent automatic clean-pixel replacement."""
    original_align = executor._handlers.get(BlockKind.ALIGN)
    if original_align is not None:
        @wraps(original_align)
        def align_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
            result = original_align(block, parameters)
            diagnostics = _restore_exact_same_canvas_references(executor.workspace)
            details = dict(result.details)
            details["verified_same_canvas_alignment"] = diagnostics
            details["verified_same_canvas_count"] = len(diagnostics)
            return ExecutionResult(result.block, result.image, details)
        executor._handlers[BlockKind.ALIGN] = align_handler

    original_region = executor._handlers.get(BlockKind.REGION_SELECT)
    if original_region is not None:
        @wraps(original_region)
        def region_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
            before = executor.workspace.copy_primary()
            result = original_region(block, parameters)
            masks = executor.workspace.occlusion_masks
            if not isinstance(masks, list) or not masks:
                return result
            try:
                damaged = _binary(np.asarray(masks[0]), before.shape[:2]) > 0
            except ValueError:
                return result

            output = before.copy()
            output[damaged] = result.image[damaged]
            suppressed = int(np.count_nonzero(np.any(result.image != before, axis=2) & ~damaged))
            provenance = executor.workspace.provenance_map
            if isinstance(provenance, np.ndarray) and provenance.shape == before.shape[:2]:
                provenance = provenance.copy()
                provenance[~damaged] = 0
                executor.workspace.provenance_map = provenance
            details = dict(result.details)
            details["preserve_visible_primary"] = True
            details["suppressed_visible_primary_pixels"] = suppressed
            details["damage_mask_pixels"] = int(np.count_nonzero(damaged))
            return ExecutionResult(result.block, output, details)
        executor._handlers[BlockKind.REGION_SELECT] = region_handler
