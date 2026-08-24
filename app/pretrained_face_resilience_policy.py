from __future__ import annotations

from functools import wraps
from typing import Any

import cv2
import numpy as np

from app.alignment import align_to_reference
from app.execution import BlockExecutionError, ExecutionResult
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_values import FACE_MODEL_DEFAULTS


_INSTALLED = False


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.transform(values, np.asarray(matrix, dtype=np.float32))[0]


def _transform_bbox(
    bbox: tuple[int, int, int, int],
    matrix: np.ndarray,
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = (float(value) for value in bbox)
    corners = np.asarray(
        [[[x, y], [x + w, y], [x, y + h], [x + w, y + h]]],
        dtype=np.float32,
    )
    warped = cv2.transform(corners, np.asarray(matrix, dtype=np.float32))[0]
    height, width = shape
    x1 = int(np.clip(np.floor(np.min(warped[:, 0])), 0, max(0, width - 1)))
    y1 = int(np.clip(np.floor(np.min(warped[:, 1])), 0, max(0, height - 1)))
    x2 = int(np.clip(np.ceil(np.max(warped[:, 0])), x1 + 1, width))
    y2 = int(np.clip(np.ceil(np.max(warped[:, 1])), y1 + 1, height))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def _preflight_accepted_original_indices(workspace) -> set[int]:
    accepted: set[int] = set()
    candidates = workspace.metadata.get("preflight_candidates")
    if not isinstance(candidates, list):
        return accepted
    for item in candidates:
        if not isinstance(item, dict) or not bool(item.get("accepted_identity", False)):
            continue
        try:
            accepted.add(int(item.get("source_index")))
        except (TypeError, ValueError):
            continue
    return accepted


def _landmark_abstention(executor, block: BlockSpec, failure: Exception) -> ExecutionResult:
    workspace = executor.workspace
    workspace.metadata.update(
        {
            "primary_landmarks5": None,
            "primary_bbox": None,
            "primary_landmark_confidence": 0.0,
            "reference_landmarks5": [],
            "reference_bboxes": [],
            "reference_landmark_confidence": [],
            "reference_identity_scores": [],
            "reference_identity_verified": [],
            "reference_identity_verification_available": False,
            "reference_partial_candidates": [],
            "face_backend": "landmark-unavailable-conservative-abstain",
            "primary_landmarks_reference_derived": False,
        }
    )
    return ExecutionResult(
        block.key,
        workspace.copy_primary(),
        {
            "backend": "landmark-unavailable-conservative-abstain",
            "pretrained": True,
            "primary_detector_failed": True,
            "pretrained_fallback_reason": str(failure),
            "landmark_count": 0,
            "landmark_confidence": 0.0,
            "reference_faces": 0,
            "verified_reference_geometry": False,
            "generated_landmarks": 0,
            "abstained": True,
            "reason": "no_verified_reference_geometry_available",
        },
    )


def _reference_derived_landmarks(executor, block: BlockSpec, failure: Exception) -> ExecutionResult:
    workspace = executor.workspace
    backend = workspace.metadata.get("_identity_backend")
    if backend is None or not hasattr(backend, "analyze"):
        return _landmark_abstention(executor, block, failure)
    if not workspace.references:
        return _landmark_abstention(executor, block, failure)

    runtime_order = workspace.metadata.get("runtime_source_order")
    if not isinstance(runtime_order, list) or len(runtime_order) != len(workspace.references) + 1:
        runtime_order = list(range(len(workspace.references) + 1))
    accepted_original = _preflight_accepted_original_indices(workspace)

    reference_observations: list[Any | None] = []
    candidates: list[tuple[float, int, Any, Any]] = []
    for index, image in enumerate(workspace.references):
        try:
            observation = backend.analyze(image)
        except Exception:
            observation = None
        reference_observations.append(observation)
        if observation is None:
            continue
        original_index = int(runtime_order[index + 1])
        if accepted_original and original_index not in accepted_original:
            continue
        try:
            aligned = align_to_reference(image, workspace.primary)
        except Exception:
            continue
        geometry_ok = (
            int(aligned.matches) >= 6
            and float(aligned.inlier_ratio) >= 0.50
            and float(aligned.reprojection_error) <= 4.0
        )
        if not geometry_ok:
            continue
        score = (
            2.0 * float(aligned.inlier_ratio)
            - 0.10 * float(aligned.reprojection_error)
            + 0.002 * min(200, int(aligned.matches))
        )
        candidates.append((score, index, observation, aligned))

    if not candidates:
        return _landmark_abstention(executor, block, failure)

    candidates.sort(key=lambda item: item[0], reverse=True)
    _, donor_index, donor, alignment = candidates[0]
    primary_points = _transform_points(donor.landmarks5, alignment.matrix)
    primary_bbox = _transform_bbox(donor.bbox, alignment.matrix, workspace.primary.shape[:2])

    reference_landmarks = [None if item is None else item.landmarks5 for item in reference_observations]
    reference_bboxes = [None if item is None else item.bbox for item in reference_observations]
    verified_flags = [False for _ in reference_observations]
    identity_scores: list[float | None] = [None for _ in reference_observations]

    workspace.metadata.update(
        {
            "primary_landmarks5": primary_points.astype(np.float32),
            "primary_bbox": primary_bbox,
            "primary_landmark_confidence": 0.70,
            "reference_landmarks5": reference_landmarks,
            "reference_bboxes": reference_bboxes,
            "reference_landmark_confidence": [0.0 if item is None else float(item.score) for item in reference_observations],
            "reference_identity_scores": identity_scores,
            "reference_identity_verified": verified_flags,
            "reference_identity_verification_available": False,
            "reference_partial_candidates": [item is None for item in reference_observations],
            "face_backend": "opencv-zoo-yunet-reference-ransac",
            "primary_landmarks_reference_derived": True,
            "primary_landmarks_donor_runtime_index": int(donor_index),
            "primary_landmarks_donor_original_index": int(runtime_order[donor_index + 1]),
        }
    )
    return ExecutionResult(
        block.key,
        workspace.copy_primary(),
        {
            "backend": "opencv-zoo-yunet-reference-ransac",
            "pretrained": True,
            "primary_detector_failed": True,
            "pretrained_fallback_reason": str(failure),
            "landmark_count": 5,
            "landmark_confidence": 0.70,
            "reference_faces": int(sum(item is not None for item in reference_observations)),
            "verified_reference_geometry": True,
            "geometry_matches": int(alignment.matches),
            "geometry_inlier_ratio": float(alignment.inlier_ratio),
            "geometry_reprojection_error": float(alignment.reprojection_error),
            "donor_runtime_index": int(donor_index),
            "donor_original_index": int(runtime_order[donor_index + 1]),
            "generated_landmarks": 0,
        },
    )


def _has_full_verified_identity_reference(workspace) -> bool:
    scores = workspace.metadata.get("reference_identity_scores")
    flags = workspace.metadata.get("reference_identity_verified")
    if not isinstance(scores, list) or not isinstance(flags, list):
        return False
    for index, score in enumerate(scores):
        if index >= len(flags) or not bool(flags[index]) or score is None:
            continue
        try:
            if float(score) >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _observed_footprint(image: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove only border-connected extreme canvas padding from donor support.

    A partial reference may be stored on a full-size black/white canvas. Geometric warp
    support alone then incorrectly marks that blank canvas as photographed evidence.
    We remove only large near-black/near-white components that touch the outer image
    border. Interior dark anatomy (hair, pupil, shadow) remains observed because it is
    not connected to the canvas boundary through an extreme-valued component.
    """
    shape = image.shape[:2]
    mask = np.asarray(support)
    if mask.shape != shape:
        return mask.astype(np.uint8, copy=True), 0
    if image.ndim != 3 or image.shape[2] < 3:
        return mask.astype(np.uint8, copy=True), 0

    rgb = np.asarray(image[:, :, :3], dtype=np.uint8)
    near_black = np.max(rgb, axis=2) <= 4
    near_white = np.min(rgb, axis=2) >= 251
    candidate = (near_black | near_white).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    padding = np.zeros(shape, dtype=bool)
    minimum_area = max(64, int(round(shape[0] * shape[1] * 0.005)))
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        touches_border = x <= 0 or y <= 0 or x + w >= shape[1] or y + h >= shape[0]
        if touches_border and area >= minimum_area:
            padding |= labels == label
    refined = np.where((mask > 0) & ~padding, 255, 0).astype(np.uint8)
    removed = int(np.count_nonzero((mask > 0) & padding))
    return refined, removed


def install_pretrained_face_resilience_policy() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    import app.pretrained_face_handlers as module

    original_install = module.install_pretrained_face_handlers

    @wraps(original_install)
    def patched_install(executor, model_paths: dict[str, Any]) -> None:
        original_install(executor, model_paths)

        landmarks = executor._handlers.get(BlockKind.LANDMARKS)
        if landmarks is not None:
            @wraps(landmarks)
            def resilient_landmarks(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
                try:
                    return landmarks(block, parameters)
                except (BlockExecutionError, ValueError, RuntimeError, cv2.error, AttributeError) as exc:
                    return _reference_derived_landmarks(executor, block, exc)

            executor._handlers[BlockKind.LANDMARKS] = resilient_landmarks

        align = executor._handlers.get(BlockKind.ALIGN)
        if align is not None:
            @wraps(align)
            def resilient_align(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
                result = align(block, parameters)
                supports = executor.workspace.metadata.get("aligned_reference_support_masks")
                refs = list(executor.workspace.aligned_references)
                if not isinstance(supports, list) or len(supports) != len(refs):
                    return result
                refined: list[np.ndarray] = []
                removed_by_slot: list[int] = []
                for reference, support in zip(refs, supports):
                    value, removed = _observed_footprint(reference, np.asarray(support))
                    refined.append(value)
                    removed_by_slot.append(removed)
                executor.workspace.metadata["aligned_reference_support_masks"] = refined
                reliability = executor.workspace.metadata.get("aligned_reference_detail_reliability_maps")
                if isinstance(reliability, list) and len(reliability) == len(refined):
                    for index, support in enumerate(refined):
                        item = np.asarray(reliability[index]).copy()
                        if item.shape == support.shape:
                            item[support == 0] = 0
                            reliability[index] = item.astype(np.uint8)
                    executor.workspace.metadata["aligned_reference_detail_reliability_maps"] = reliability
                details = dict(result.details)
                details["border_padding_support_excluded_pixels"] = removed_by_slot
                details["observed_support_excludes_border_padding"] = True
                return ExecutionResult(result.block, result.image, details)

            executor._handlers[BlockKind.ALIGN] = resilient_align

        identity = executor._handlers.get(BlockKind.IDENTITY_CHECK)
        if identity is not None:
            @wraps(identity)
            def resilient_identity(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
                try:
                    return identity(block, parameters)
                except BlockExecutionError as exc:
                    if _has_full_verified_identity_reference(executor.workspace):
                        raise
                    return ExecutionResult(
                        block.key,
                        executor.workspace.copy_primary(),
                        {
                            "engine": "per-block-identity-retention-guardrails",
                            "pretrained": True,
                            "absolute_sface_gate_applied": False,
                            "reason": "no_full_verified_identity_reference",
                            "partial_references_not_used_as_negative_identity_evidence": True,
                            "original_sface_rejection": str(exc),
                        },
                    )

            executor._handlers[BlockKind.IDENTITY_CHECK] = resilient_identity

    module.install_pretrained_face_handlers = patched_install
    _INSTALLED = True
