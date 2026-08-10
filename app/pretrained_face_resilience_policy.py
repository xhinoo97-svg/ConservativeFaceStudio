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
    # Missing pretrained geometry is not a reason to revive a synthetic/obsolete Haar
    # path. The block completes as an explicit conservative abstention; later reference
    # alignment can still use observed SIFT/ORB/RANSAC geometry where available.
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
    verified_flags: list[bool] = []
    identity_scores: list[float | None] = []
    try:
        primary_embedding = backend.analyze(workspace.primary).embedding
    except Exception:
        primary_embedding = None
    for index, item in enumerate(reference_observations):
        original_index = int(runtime_order[index + 1])
        # Preflight membership alone is insufficient to call a component-only image a
        # full identity reference. Keep the flag false unless a real SFace comparison
        # can be produced later by the normal handler.
        verified_flags.append(False)
        identity_scores.append(None)

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


def install_pretrained_face_resilience_policy() -> None:
    """Patch pretrained face installation with conservative occlusion resilience.

    The normal YuNet/SFace path remains untouched. If YuNet cannot detect a heavily
    occluded primary, five-point geometry may be transferred only from a real preflight
    same-identity reference whose feature alignment to the primary passes RANSAC gates.
    If no verified geometry exists, LANDMARKS completes in explicit abstention mode.
    Partial/component references are prevented from becoming negative SFace evidence.
    """
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
