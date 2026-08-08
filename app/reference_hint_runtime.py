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


def _local_observed_quality(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    sharpness = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F)), (0, 0), 2.0)
    exposure = 1.0 - np.clip(np.abs(gray - 0.5) / 0.5, 0.0, 1.0)
    return sharpness + 0.02 * exposure


def _connected_to_seed(candidate: np.ndarray, seed: np.ndarray) -> np.ndarray:
    binary = np.where(candidate, 255, 0).astype(np.uint8)
    if not np.any(binary) or not np.any(seed):
        return np.zeros(candidate.shape, dtype=bool)
    expanded_seed = cv2.dilate(
        np.where(seed, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        iterations=1,
    ) > 0
    count, labels = cv2.connectedComponents(binary, connectivity=8)
    keep = np.zeros(candidate.shape, dtype=bool)
    for label in range(1, count):
        component = labels == label
        if np.any(component & expanded_seed):
            keep |= component
    return keep


def expand_verified_single_reference_hint(
    primary: np.ndarray,
    reference: np.ndarray,
    reference_mask: np.ndarray,
    face_mask: np.ndarray,
    existing_hint: np.ndarray,
    *,
    minimum_face_coverage: float = 0.70,
    strong_difference_threshold: float = 0.10,
    maximum_face_fraction: float = 0.25,
    minimum_donor_quality_advantage: float = 0.001,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Expand a primary damage seed using a verified nearly-complete real reference.

    The expansion is directional. A large primary/reference difference is not enough:
    the primary must already contain a damage seed in that connected region and the
    observed donor must have at least slightly better local quality. This prevents a
    damaged reference from being copied back into a cleaner frontal base selected by
    preflight, while still growing a small detected core across an opaque sticker.
    """
    if primary.shape != reference.shape or primary.ndim != 3 or primary.shape[2] != 3:
        raise ValueError("Primary/reference non compatibili")
    shape = primary.shape[:2]
    blocked = _binary(reference_mask, shape) > 0
    face = _binary(face_mask, shape) > 0
    raw_hint = _binary(existing_hint, shape) > 0
    observed = (~blocked) & (np.max(reference, axis=2) > 2) & face
    face_pixels = max(1, int(np.count_nonzero(face)))
    coverage = float(np.count_nonzero(observed) / face_pixels)
    details: dict[str, Any] = {
        "eligible": False,
        "face_coverage": coverage,
        "added_pixels": 0,
        "reason": "insufficient_reference_coverage",
    }
    if coverage < float(minimum_face_coverage):
        return raw_hint.astype(np.uint8) * 255, details

    base_lab = cv2.cvtColor(primary, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    difference = np.mean(np.abs(base_lab - ref_lab), axis=2)
    base_quality = _local_observed_quality(primary)
    ref_quality = _local_observed_quality(reference)
    donor_advantage = ref_quality >= (base_quality + float(minimum_donor_quality_advantage))
    changed = observed & (difference >= float(strong_difference_threshold))
    directional = changed & donor_advantage

    # The seed itself is also directional: a heuristic false positive on an otherwise
    # clean primary must not authorize copying a worse/damaged reference into it.
    seed = raw_hint & directional
    if not np.any(seed):
        details.update({
            "eligible": True,
            "reason": "no_directional_primary_damage_seed",
            "directional_seed_pixels": 0,
            "strong_difference_pixels": int(np.count_nonzero(changed)),
        })
        return np.zeros(shape, dtype=np.uint8), details

    strong_mask = directional.astype(np.uint8) * 255
    strong_mask = cv2.morphologyEx(strong_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    strong_mask = cv2.morphologyEx(strong_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    connected = _connected_to_seed((strong_mask > 0) & observed, seed)
    proposed = (seed | connected) & face & observed

    proposed_pixels = int(np.count_nonzero(proposed))
    maximum_pixels = max(1, int(round(face_pixels * float(maximum_face_fraction))))
    if proposed_pixels > maximum_pixels:
        details.update({
            "eligible": True,
            "reason": "proposal_too_large",
            "proposed_pixels": proposed_pixels,
            "maximum_pixels": maximum_pixels,
            "directional_seed_pixels": int(np.count_nonzero(seed)),
        })
        return seed.astype(np.uint8) * 255, details

    expanded = proposed.astype(np.uint8) * 255
    added = int(np.count_nonzero(connected & ~seed))
    details.update({
        "eligible": True,
        "reason": "expanded" if added else "no_additional_directional_difference",
        "added_pixels": added,
        "proposed_pixels": proposed_pixels,
        "maximum_pixels": maximum_pixels,
        "directional_seed_pixels": int(np.count_nonzero(seed)),
        "strong_difference_pixels": int(np.count_nonzero(changed)),
    })
    return expanded, details


def install_reference_hint_runtime(executor) -> None:
    """Improve recall for opaque covers when one verified full reference exists."""
    original = executor._handlers.get(BlockKind.INPAINT)
    if original is None:
        return

    @wraps(original)
    def handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        workspace = executor.workspace
        references = list(workspace.aligned_references)
        diagnostics: dict[str, Any] = {"eligible": False, "reason": "not_single_reference"}
        if len(references) == 1 and bool(workspace.metadata.get("reference_identity_verification_available", False)):
            shape = workspace.primary.shape[:2]
            masks = workspace.occlusion_masks
            reference_mask = (
                np.asarray(masks[1])
                if isinstance(masks, list) and len(masks) == 2
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
                minimum_face_coverage=float(parameters.get("full_reference_minimum_face_coverage", 0.70)),
                strong_difference_threshold=float(parameters.get("full_reference_strong_difference_threshold", 0.10)),
                maximum_face_fraction=float(parameters.get("maximum_occlusion_fraction", 0.25)),
                minimum_donor_quality_advantage=float(parameters.get("minimum_donor_quality_advantage", 0.001)),
            )
            workspace.metadata["reference_consensus_occlusion"] = expanded

        result = original(block, parameters)
        details = dict(result.details)
        details["verified_single_reference_hint"] = diagnostics
        return ExecutionResult(result.block, result.image, details)

    executor._handlers[BlockKind.INPAINT] = handler
