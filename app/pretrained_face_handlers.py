from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.alignment import align_from_points, align_to_reference
from app.component_bank import build_component_bank, warped_support_mask
from app.execution import BlockExecutionError, ExecutionResult
from app.face_analysis import cosine_similarity
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_values import FACE_MODEL_DEFAULTS


def install_pretrained_face_handlers(executor, model_paths: dict[str, str | Path]) -> None:
    """Install verified YuNet/SFace handlers with partial-reference support."""
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
        support_masks: list[np.ndarray] = []
        reliability_maps: list[np.ndarray] = []
        diagnostics: list[dict[str, Any]] = []
        runtime_source_indices: list[int] = []
        original_source_indices: list[int] = []
        aligned_scores: list[float | None] = []
        aligned_verified: list[bool] = []
        partial_geometry_verified: list[bool] = []

        primary_points = executor.workspace.metadata.get("primary_landmarks5")
        ref_points = executor.workspace.metadata.get("reference_landmarks5", [])
        identity_scores = executor.workspace.metadata.get("reference_identity_scores", [])
        identity_available = bool(executor.workspace.metadata.get("reference_identity_verification_available", False))
        runtime_order = executor.workspace.metadata.get("runtime_source_order")
        if not isinstance(runtime_order, list) or len(runtime_order) != len(executor.workspace.references) + 1:
            runtime_order = list(range(len(executor.workspace.references) + 1))
        preflight_reliability = executor.workspace.metadata.get("preflight_detail_reliability_maps")
        reliability_threshold = int(np.clip(executor.workspace.metadata.get("detail_reliability_threshold", 40), 0, 255))
        rejected_identity = 0
        rejected_geometry = 0

        for index, reference in enumerate(executor.workspace.references):
            score = identity_scores[index] if isinstance(identity_scores, list) and index < len(identity_scores) else None
            if identity_available and score is not None and float(score) < FACE_MODEL_DEFAULTS.sface_same_identity_cosine:
                rejected_identity += 1
                diagnostics.append({
                    "source_index": index,
                    "original_source_index": int(runtime_order[index + 1]),
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
                    "original_source_index": int(runtime_order[index + 1]),
                    "rejected": True,
                    "reason": "partial_geometry_rejected" if score is None else "alignment_failed",
                    "error": str(exc),
                    "identity_score": None if score is None else float(score),
                })
                continue

            target_h, target_w = executor.workspace.primary.shape[:2]
            support = warped_support_mask(reference.shape[:2], result.matrix, (target_h, target_w))
            source_reliability = None
            if isinstance(preflight_reliability, list) and index + 1 < len(preflight_reliability):
                candidate = np.asarray(preflight_reliability[index + 1])
                if candidate.shape == reference.shape[:2]:
                    source_reliability = candidate.astype(np.uint8, copy=False)
            if source_reliability is None:
                # Fail closed for evidence quality: unknown pre-deblur reliability may
                # still support geometry, but cannot donate identity-critical texture.
                aligned_reliability = np.zeros((target_h, target_w), dtype=np.uint8)
            else:
                aligned_reliability = cv2.warpAffine(
                    source_reliability,
                    result.matrix,
                    (target_w, target_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(np.uint8)
                aligned_reliability[support == 0] = 0

            original_index = int(runtime_order[index + 1])
            aligned.append(result.image)
            support_masks.append(support)
            reliability_maps.append(aligned_reliability)
            runtime_source_indices.append(index)
            original_source_indices.append(original_index)
            aligned_scores.append(None if score is None else float(score))
            full_identity_verified = score is not None and float(score) >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine
            aligned_verified.append(bool(full_identity_verified))
            partial_geometry_verified.append(bool(score is None and geometry_verified))
            reliable_fraction = float(np.mean((support > 0) & (aligned_reliability >= reliability_threshold)))
            diagnostics.append({
                "source_index": index,
                "original_source_index": original_index,
                "method": method,
                "matches": result.matches,
                "inlier_ratio": result.inlier_ratio,
                "reprojection_error": result.reprojection_error,
                "support_fraction": float(np.mean(support > 0)),
                "reliable_support_fraction": reliable_fraction,
                "identity_score": None if score is None else float(score),
                "identity_status": "sface_verified" if full_identity_verified else "partial_geometry_verified",
            })

        executor.workspace.aligned_references = aligned
        executor.workspace.metadata["aligned_reference_source_indices"] = runtime_source_indices
        executor.workspace.metadata["aligned_reference_original_source_indices"] = original_source_indices
        executor.workspace.metadata["aligned_reference_support_masks"] = support_masks
        executor.workspace.metadata["aligned_reference_detail_reliability_maps"] = reliability_maps
        executor.workspace.metadata["aligned_reference_identity_scores"] = aligned_scores
        executor.workspace.metadata["aligned_reference_identity_verified"] = aligned_verified
        executor.workspace.metadata["aligned_reference_partial_geometry_verified"] = partial_geometry_verified

        bank_summary: dict[str, list[dict[str, Any]]] = {}
        bbox = executor.workspace.metadata.get("primary_bbox")
        if support_masks and primary_points is not None and bbox is not None:
            evidence_support_masks = [
                cv2.bitwise_and(
                    support,
                    np.where(reliability >= reliability_threshold, 255, 0).astype(np.uint8),
                )
                for support, reliability in zip(support_masks, reliability_maps)
            ]
            bank = build_component_bank(
                evidence_support_masks,
                np.asarray(primary_points, dtype=np.float32),
                tuple(int(v) for v in bbox),
                source_indices=original_source_indices,
                minimum_coverage=float(parameters.get("component_minimum_coverage", 0.18)),
            )
            bank_summary = {name: [asdict(item) for item in items] for name, items in bank.items()}
            executor.workspace.metadata["component_reference_bank"] = bank_summary

        return ExecutionResult(
            block.key,
            executor.workspace.copy_primary(),
            {
                "aligned": len(aligned),
                "rejected_identity": rejected_identity,
                "rejected_geometry": rejected_geometry,
                "partial_geometry_verified": int(sum(partial_geometry_verified)),
                "identity_filter_applied": identity_available,
                "runtime_source_indices": runtime_source_indices,
                "original_source_indices": original_source_indices,
                "support_masks_created": len(support_masks),
                "pre_deblur_reliability_maps_propagated": len(reliability_maps),
                "detail_reliability_threshold": reliability_threshold,
                "component_bank_regions": {name: len(items) for name, items in bank_summary.items()},
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
