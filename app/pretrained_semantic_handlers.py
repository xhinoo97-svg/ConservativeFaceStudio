from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.frontalization import conservative_mild_frontal_affine, warp_auxiliary_map
from app.opencv_semantic_models import FaceParsingEngine, HeadPoseEngine
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_values import HEAD_POSE_DEFAULTS


def _hardware_target(workspace) -> str:
    policy = workspace.metadata.get("hardware_policy")
    if isinstance(policy, dict) and str(policy.get("dnn_target", "cpu")).lower() == "opencl":
        return "opencl"
    return "cpu"


def _face_crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = (int(value) for value in bbox)
    height, width = image.shape[:2]
    margin_x = int(round(w * 0.12))
    margin_y = int(round(h * 0.12))
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)
    return image[y1:y2, x1:x2]


def install_pretrained_semantic_handlers(executor, model_paths: dict[str, str | Path]) -> None:
    """Attach non-generative pretrained semantic/pose models to strict blocks."""
    target = _hardware_target(executor.workspace)

    parsing_path = model_paths.get("face_parsing_resnet18_onnx")
    original_occlusion = executor._handlers.get(BlockKind.OCCLUSION_MASK)
    parsing_engines: dict[str, FaceParsingEngine] = {}

    def parsing_engine(requested: str) -> FaceParsingEngine:
        engine = parsing_engines.get(requested)
        if engine is None:
            engine = FaceParsingEngine(Path(parsing_path), target=requested)
            parsing_engines[requested] = engine
        return engine

    if parsing_path is not None and Path(parsing_path).is_file() and original_occlusion is not None:
        def occlusion_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
            actual_target = target
            try:
                try:
                    engine = parsing_engine(actual_target)
                    labels = engine.predict(executor.workspace.primary)
                except Exception:
                    if actual_target != "opencl":
                        raise
                    actual_target = "cpu"
                    engine = parsing_engine("cpu")
                    labels = engine.predict(executor.workspace.primary)

                support = engine.support_mask(labels)
                accessories = engine.accessory_mask(labels)
                executor.workspace.metadata["face_parsing_labels"] = labels
                executor.workspace.metadata["face_semantic_support"] = support
                executor.workspace.metadata["face_accessory_candidates"] = accessories

                result = original_occlusion(block, parameters)
                stored = executor.workspace.metadata.get("reference_consensus_occlusion")
                if isinstance(stored, np.ndarray) and stored.shape == support.shape:
                    constrained = cv2.bitwise_and(stored.astype(np.uint8), support)
                    executor.workspace.metadata["reference_consensus_occlusion"] = constrained
                details = dict(result.details)
                details.update({
                    "face_parsing": "resnet18-celebamaskhq-onnx",
                    "face_parsing_pretrained": True,
                    "face_parsing_backend": actual_target,
                    "semantic_face_coverage": float(np.mean(support > 0)),
                    "accessory_candidate_coverage": float(np.mean(accessories > 0)),
                })
                return ExecutionResult(block.key, result.image, details)
            except Exception as exc:
                fallback = original_occlusion(block, parameters)
                details = dict(fallback.details)
                details["face_parsing_pretrained"] = False
                details["face_parsing_fallback_reason"] = str(exc)
                return ExecutionResult(block.key, fallback.image, details)

        executor._handlers[BlockKind.OCCLUSION_MASK] = occlusion_handler

    pose_path = model_paths.get("head_pose_mobilenetv2_onnx")
    original_frontalize = executor._handlers.get(BlockKind.FRONTALIZE)
    pose_engines: dict[str, HeadPoseEngine] = {}

    def pose_engine(requested: str) -> HeadPoseEngine:
        engine = pose_engines.get(requested)
        if engine is None:
            engine = HeadPoseEngine(Path(pose_path), target=requested)
            pose_engines[requested] = engine
        return engine

    if pose_path is not None and Path(pose_path).is_file() and original_frontalize is not None:
        def frontalize_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
            bbox = executor.workspace.metadata.get("primary_bbox")
            if bbox is None:
                return original_frontalize(block, parameters)
            actual_target = target
            try:
                crop = _face_crop(executor.workspace.primary, tuple(int(v) for v in bbox))
                try:
                    pitch, yaw, roll = pose_engine(actual_target).estimate(crop)
                except Exception:
                    if actual_target != "opencl":
                        raise
                    actual_target = "cpu"
                    pitch, yaw, roll = pose_engine("cpu").estimate(crop)

                pose = {"pitch": pitch, "yaw": yaw, "roll": roll}
                executor.workspace.metadata["pretrained_head_pose"] = pose
                safe = (
                    abs(yaw) <= HEAD_POSE_DEFAULTS.max_abs_yaw_strict
                    and abs(pitch) <= HEAD_POSE_DEFAULTS.max_abs_pitch_strict
                )
                if not safe:
                    return ExecutionResult(
                        block.key,
                        executor.workspace.copy_primary(),
                        {
                            "engine": "mobilenetv2-head-pose-gated-abstain",
                            "pretrained": True,
                            "backend": actual_target,
                            "pose_degrees": pose,
                            "applied": False,
                            "yaw_synthesized": False,
                            "reason": "posa fuori dalla zona sicura: nessuna texture non osservata viene sintetizzata",
                        },
                    )

                result = original_frontalize(block, parameters)
                details = dict(result.details)
                details.update({
                    "head_pose_model": "mobilenetv2-6d-onnx",
                    "head_pose_pretrained": True,
                    "head_pose_backend": actual_target,
                    "pose_degrees": pose,
                    "pose_gate_passed": True,
                })

                landmarks = executor.workspace.metadata.get("primary_landmarks5")
                mild = None
                if landmarks is not None:
                    mild = conservative_mild_frontal_affine(
                        result.image,
                        np.asarray(landmarks, dtype=np.float32),
                        tuple(int(v) for v in bbox),
                        float(yaw),
                        maximum_abs_yaw=min(12.0, HEAD_POSE_DEFAULTS.max_abs_yaw_strict),
                    )

                if mild is not None:
                    details["mild_yaw_frontalization"] = {
                        "applied": mild.applied,
                        "yaw_degrees": mild.yaw_degrees,
                        "strength": mild.strength,
                        "max_landmark_displacement": mild.max_landmark_displacement,
                        "supported_fraction": mild.supported_fraction,
                        "reason": mild.reason,
                        "synthesized_pixels": 0,
                    }
                    if mild.applied:
                        provenance = executor.workspace.provenance_map
                        if isinstance(provenance, np.ndarray) and provenance.shape == mild.changed_mask.shape:
                            executor.workspace.provenance_map = warp_auxiliary_map(
                                provenance,
                                mild.matrix,
                                mild.changed_mask,
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(np.uint16)
                        confidence = executor.workspace.metadata.get("specific_reference_confidence")
                        if isinstance(confidence, np.ndarray) and confidence.shape == mild.changed_mask.shape:
                            executor.workspace.metadata["specific_reference_confidence"] = warp_auxiliary_map(
                                confidence,
                                mild.matrix,
                                mild.changed_mask,
                                interpolation=cv2.INTER_NEAREST,
                            ).astype(np.uint8)
                        executor.workspace.metadata["primary_landmarks5"] = mild.transformed_landmarks.copy()
                        details["engine"] = "observed-roll-plus-mild-yaw-frontalization"
                        details["yaw_synthesized"] = False
                        details["provenance_geometry_updated"] = executor.workspace.provenance_map is not None
                        return ExecutionResult(block.key, mild.image, details)

                return ExecutionResult(block.key, result.image, details)
            except Exception as exc:
                fallback = original_frontalize(block, parameters)
                details = dict(fallback.details)
                details["head_pose_pretrained"] = False
                details["head_pose_fallback_reason"] = str(exc)
                return ExecutionResult(block.key, fallback.image, details)

        executor._handlers[BlockKind.FRONTALIZE] = frontalize_handler
