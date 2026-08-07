from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.execution import BlockExecutionError, ExecutionResult
from app.face_analysis import cosine_similarity
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_values import FACE_MODEL_DEFAULTS


def install_pretrained_face_handlers(executor, model_paths: dict[str, str | Path]) -> None:
    """Install verified YuNet/SFace handlers with safe CPU/OpenCL fallback."""
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

    # Private runtime object: it is intentionally not included in project JSON.
    # AutomaticPipelineRunner can reuse it for post-block identity guardrails.
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
                    "reference_landmark_confidence": [0.0 if item is None else item.score for item in refs],
                    "reference_identity_scores": identity_scores,
                    "reference_identity_verified": identity_verified,
                    "reference_identity_verification_available": primary.embedding is not None,
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
                    "reference_identity_scores": identity_scores,
                    "reference_identity_verified": int(sum(identity_verified)),
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

    if sface is None or original_identity is None:
        return

    try:
        identity_engine = make_engine(with_identity=True)
        executor.workspace.metadata["_identity_backend"] = identity_engine
    except Exception:
        return

    def identity_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        # 0.363 is the OpenCV Zoo SFace cosine reference threshold. A caller may
        # request a stricter value but cannot silently weaken the pretrained gate.
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
