from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.execution import BlockExecutionError, ExecutionResult
from app.face_analysis import cosine_similarity
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pipeline import BlockKind, BlockSpec


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

    try:
        landmark_engine = OpenCVZooFaceEngine(
            yunet,
            sface if sface is not None else None,
            dnn_target=dnn_target,
        )
    except Exception:
        return

    def landmarks_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        try:
            primary = landmark_engine.analyze(executor.workspace.primary)
            refs = []
            for image in executor.workspace.references:
                try:
                    refs.append(landmark_engine.analyze(image))
                except ValueError:
                    refs.append(None)
            backend = f"opencv-zoo-yunet-sface-{landmark_engine.target_name}" if primary.embedding is not None else f"opencv-zoo-yunet-{landmark_engine.target_name}"
            executor.workspace.metadata.update(
                {
                    "primary_landmarks5": primary.landmarks5,
                    "primary_bbox": primary.bbox,
                    "primary_landmark_confidence": primary.score,
                    "reference_landmarks5": [None if item is None else item.landmarks5 for item in refs],
                    "reference_landmark_confidence": [0.0 if item is None else item.score for item in refs],
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
                    "reference_faces": int(sum(item is not None for item in refs)),
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
        identity_engine = OpenCVZooFaceEngine(yunet, sface, dnn_target=dnn_target)
    except Exception:
        return

    def identity_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
        minimum = float(parameters.get("minimum", 0.35))
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
