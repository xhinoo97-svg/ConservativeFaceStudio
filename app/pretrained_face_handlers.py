from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.alignment import align_from_points, align_to_reference
from app.execution import BlockExecutionError, ExecutionResult
from app.face_analysis import cosine_similarity
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_values import FACE_MODEL_DEFAULTS


def install_pretrained_face_handlers(executor, model_paths: dict[str, str | Path]) -> None:
    """Install verified YuNet/SFace handlers with safe CPU/OpenCL fallback.

    Full-face references use SFace identity verification. A partial crop that cannot
    produce an embedding is not automatically treated as a mismatch: it may still be
    retained if strict local feature alignment succeeds. A real low SFace score is a
    hard rejection.
    """
    yunet = model_paths.get("opencv_yunet")
    sface = model_paths.get("opencv_sface")
    if yunet is None:
        return

    hardware = executor.workspace.metadata.get("hardware_policy")
    dnn_target = "cpu"
    if isinstance(hardware, dict):
        candidate = str(hardware.get("dnn_target", "cpu")).lower()
        if candidate in {"cpu", "opencl"}:
            dnn_target = candidate

    original_landmarks = executor._handlers.get(BlockKind.LANDMARKS)
    original_identity = executor._handlers.get(BlockKind.IDENTITY_CHECK)

    def make_engine(with_identity: bool) -> OpenCVZooFaceEngine:
        return OpenCVZooFaceEngine(
            yunet,
            sface if with_identity and sface is not None else None,
            score_threshold=FACE_MODEL_DEFAULTS.yunet_score_threshold,
            nms_threshold=FACE_MODEL_DEFAULTS.yunet_nms_threshold,
            top_k=FACE_MODEL_DEFAULTS.yunet_top_k,
            dnn_target=dnn_target,
        )

    try:
        landmark_engine = make_engine(with_identity=True)
    except Exception:
        return

    if landmark_engine.recognizer is not None:
        executor.workspace.metadata["_identity_backend"] = landmark_engine

    def landmarks_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        try:
            primary = landmark_engine.analyze(executor.workspace.primary)
            refs = []
            for image in executor.workspace.references:
                try:
                    refs.append(landmark_engine.analyze(image))
                except ValueError:
                    refs.append(None)

            identity_scores: list[float | None] = []
            if primary.embedding is not None:
                for item in refs:
                    if item is None or item.embedding is None:
                        identity_scores.append(None)
                    else:
                        identity_scores.append(cosine_similarity(primary.embedding, item.embedding))
            else:
                identity_scores = [None for _ in refs]

            identity_verified = [
                score is not None and score >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine
                for score in identity_scores
            ]
            backend = (
                f"opencv-zoo-yunet-sface-{landmark_engine.target_name}"
                if primary.embedding is not None
                else f"opencv-zoo-yunet-{landmark_engine.target_name}"
            )
            executor.workspace.metadata.update(
                {
                    "primary_landmarks5": primary.landmarks5,
                    "primary_bbox": primary.bbox,
                    "primary_landmark_confidence": primary.score,
                    "reference_landmarks5": [None if item is None else item.landmarks5 for item in refs],
                    "reference_bboxes": [None if item is None else item.bbox for item in refs],
                    "reference_landmark_confidence": [0.0 if item is None else item.score for item in refs],
                    "reference_identity_scores": identity_scores,
                    "reference_identity_verified": identity_verified,
                    "reference_identity_verification_available": primary.embedding is not None,
                    "reference_partial_candidates": [item is None for item in refs],
                    "face_backend": backend,
                }
            )
            return ExecutionResult(
                block.key,
                executor.workspace.copy_primary(),
                {
                    "backend": backend,
                    "pretrained": True,
                    "dnn_target": landmark_engine.target_name,
                    "bbox": list(primary.bbox),
                    "landmark_count": 5,
                    "landmark_confidence": float(primary.score),
                    "yunet_score_threshold": FACE_MODEL_DEFAULTS.yunet_score_threshold,
                    "reference_faces": int(sum(item is not None for item in refs)),
                    "partial_reference_candidates": int(sum(item is None for item in refs)),
                    "reference_identity_scores": identity_scores,
                    "reference_identity_verified": int(sum(identity_verified)),
                    "reference_bbox_count": int(sum(item is not None for item in executor.workspace.metadata["reference_bboxes"])),
                    "sface_reference_threshold": FACE_MODEL_DEFAULTS.sface_same_identity_cosine,
                },
            )
        except Exception as exc:
            if original_landmarks is None:
                raise BlockExecutionError(str(exc)) from exc
            fallback = original_landmarks(block, parameters)
            details = dict(fallback.details)
            details["pretrained_fallback_reason"] = str(exc)
            return ExecutionResult(block.key, fallback.image, details)

    executor._handlers[BlockKind.LANDMARKS] = landmarks_handler

    def partial_aware_align_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        aligned: list[np.ndarray] = []
        diagnostics: list[dict[str, Any]] = []
        source_indices: list[int] = []
        aligned_scores: list[float | None] = []
        aligned_verified: list[bool] = []
        partial_geometry_verified: list[bool] = []

        primary_points = executor.workspace.metadata.get("primary_landmarks5")
        ref_points = executor.workspace.metadata.get("reference_landmarks5", [])
        identity_scores = executor.workspace.metadata.get("reference_identity_scores", [])
        identity_available = bool(executor.workspace.metadata.get("reference_identity_verification_available", False))
        rejected_identity = 0
        rejected_geometry = 0

        for index, reference in enumerate(executor.workspace.references):
            score = identity_scores[index] if isinstance(identity_scores, list) and index < len(identity_scores) else None
            # A real SFace score below threshold is a mismatch. None means the crop was
            # too partial for a full-face embedding and is allowed to try geometry.
            if identity_available and score is not None and float(score) < FACE_MODEL_DEFAULTS.sface_same_identity_cosine:
                rejected_identity += 1
                diagnostics.append({
                    "source_index": index,
                    "rejected": True,
                    "reason": "sface_identity_mismatch",
                    "identity_score": float(score),
                })
                continue

            points = ref_points[index] if isinstance(ref_points, list) and index < len(ref_points) else None
            method = ""
            geometry_verified = False
            try:
                if primary_points is not None and points is not None:
                    result = align_from_points(reference, points, primary_points, executor.workspace.primary.shape[:2])
                    method = "landmarks5-ransac"
                    geometry_verified = True
                else:
                    result = align_to_reference(reference, executor.workspace.primary)
                    method = "partial-sift-orb-ransac"
                    geometry_verified = (
                        result.matches >= 6
                        and result.inlier_ratio >= 0.50
                        and result.reprojection_error <= 4.0
                    )
                    if not geometry_verified:
                        raise ValueError("reference parziale senza supporto geometrico sufficiente")
            except Exception as exc:
                rejected_geometry += 1
                diagnostics.append({
                    "source_index": index,
                    "rejected": True,
                    "reason": "partial_geometry_rejected" if score is None else "alignment_failed",
                    "error": str(exc),
                    "identity_score": None if score is None else float(score),
                })
                continue

            aligned.append(result.image)
            source_indices.append(index)
            aligned_scores.append(None if score is None else float(score))
            full_identity_verified = score is not None and float(score) >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine
            aligned_verified.append(bool(full_identity_verified))
            partial_geometry_verified.append(bool(score is None and geometry_verified))
            diagnostics.append({
                "source_index": index,
                "method": method,
                "matches": result.matches,
                "inlier_ratio": result.inlier_ratio,
                "reprojection_error": result.reprojection_error,
                "identity_score": None if score is None else float(score),
                "identity_status": "sface_verified" if full_identity_verified else "partial_geometry_verified",
            })

        executor.workspace.aligned_references = aligned
        executor.workspace.metadata["aligned_reference_source_indices"] = source_indices
        executor.workspace.metadata["aligned_reference_identity_scores"] = aligned_scores
        executor.workspace.metadata["aligned_reference_identity_verified"] = aligned_verified
        executor.workspace.metadata["aligned_reference_partial_geometry_verified"] = partial_geometry_verified
        return ExecutionResult(
            block.key,
            executor.workspace.copy_primary(),
            {
                "aligned": len(aligned),
                "rejected_identity": rejected_identity,
                "rejected_geometry": rejected_geometry,
                "partial_geometry_verified": int(sum(partial_geometry_verified)),
                "identity_filter_applied": identity_available,
                "source_indices": source_indices,
                "diagnostics": diagnostics,
            },
        )

    executor._handlers[BlockKind.ALIGN] = partial_aware_align_handler

    if sface is None or original_identity is None:
        return

    try:
        identity_engine = make_engine(with_identity=True)
        executor.workspace.metadata["_identity_backend"] = identity_engine
    except Exception:
        return

    def identity_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        minimum = max(
            float(parameters.get("minimum", FACE_MODEL_DEFAULTS.sface_same_identity_cosine)),
            FACE_MODEL_DEFAULTS.sface_same_identity_cosine,
        )
        try:
            primary = identity_engine.analyze(executor.workspace.primary)
            if primary.embedding is None:
                raise ValueError("SFace non ha prodotto embedding")
            scores: list[float] = []
            for image in executor.workspace.references:
                try:
                    reference = identity_engine.analyze(image)
                except ValueError:
                    # Partial references are not evidence against the reconstructed
                    # identity; they were already gated by strict local geometry.
                    continue
                if reference.embedding is not None:
                    scores.append(cosine_similarity(primary.embedding, reference.embedding))
            best = max(scores, default=1.0)
            if scores and best < minimum:
                raise BlockExecutionError(
                    f"Controllo identità SFace sotto soglia: {best:.3f} < {minimum:.3f}"
                )
            return ExecutionResult(
                block.key,
                executor.workspace.copy_primary(),
                {
                    "engine": f"opencv-zoo-sface-{identity_engine.target_name}",
                    "pretrained": True,
                    "dnn_target": identity_engine.target_name,
                    "scores": scores,
                    "best": float(best),
                    "minimum": minimum,
                    "official_reference_threshold": FACE_MODEL_DEFAULTS.sface_same_identity_cosine,
                    "partial_references_not_used_as_negative_identity_evidence": True,
                },
            )
        except BlockExecutionError:
            raise
        except Exception as exc:
            fallback = original_identity(block, parameters)
            details = dict(fallback.details)
            details["pretrained_fallback_reason"] = str(exc)
            return ExecutionResult(block.key, fallback.image, details)

    executor._handlers[BlockKind.IDENTITY_CHECK] = identity_handler
