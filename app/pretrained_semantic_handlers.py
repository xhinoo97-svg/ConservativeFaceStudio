from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.execution import ExecutionResult
from app.frontalization import (
    conservative_mild_frontal_affine,
    select_more_frontal_reference,
    warp_auxiliary_map,
)
from app.opencv_semantic_models import FaceParsingEngine, HeadPoseEngine
from app.pipeline import BlockKind, BlockSpec
from app.pretrained_values import FACE_MODEL_DEFAULTS, HEAD_POSE_DEFAULTS


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


def _aligned_identity_scores(workspace, count: int) -> list[float | None] | None:
    values = workspace.metadata.get("aligned_reference_identity_scores")
    if isinstance(values, list) and len(values) == count:
        return [None if item is None else float(item) for item in values]
    return None


def _aligned_source_indices(workspace, count: int) -> list[int]:
    values = workspace.metadata.get("aligned_reference_source_indices")
    if isinstance(values, list) and len(values) == count:
        try:
            parsed = [int(item) for item in values]
        except (TypeError, ValueError):
            parsed = []
        if len(parsed) == count and all(0 <= item < len(workspace.references) for item in parsed):
            return parsed
    return list(range(count))


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

    def estimate_with_fallback(image: np.ndarray, bbox: tuple[int, int, int, int], requested: str):
        crop = _face_crop(image, bbox)
        try:
            return pose_engine(requested).estimate(crop), requested
        except Exception:
            if requested != "opencl":
                raise
            return pose_engine("cpu").estimate(crop), "cpu"

    if pose_path is not None and Path(pose_path).is_file() and original_frontalize is not None:
        def frontalize_handler(block: BlockSpec, parameters: dict[str, Any]) -> ExecutionResult:
            bbox = executor.workspace.metadata.get("primary_bbox")
            if bbox is None:
                return original_frontalize(block, parameters)
            box = tuple(int(v) for v in bbox)
            actual_target = target
            try:
                primary_pose, actual_target = estimate_with_fallback(executor.workspace.primary, box, actual_target)
                pitch, yaw, roll = (float(v) for v in primary_pose)
                pose = {"pitch": pitch, "yaw": yaw, "roll": roll}
                executor.workspace.metadata["pretrained_head_pose"] = pose

                # Estimate each aligned reference sequentially: one small ONNX model stays
                # resident, so this remains practical on the 16-GB CPU-first target laptop.
                references = list(executor.workspace.aligned_references)
                reference_poses: list[tuple[float, float, float] | None] = []
                reference_backends: list[str | None] = []
                for reference in references:
                    try:
                        estimated, backend = estimate_with_fallback(reference, box, actual_target)
                        reference_poses.append(tuple(float(v) for v in estimated))
                        reference_backends.append(backend)
                    except Exception:
                        reference_poses.append(None)
                        reference_backends.append(None)

                identity_available = bool(
                    executor.workspace.metadata.get("reference_identity_verification_available", False)
                )
                identity_scores = _aligned_identity_scores(executor.workspace, len(references))
                evidence = select_more_frontal_reference(
                    (pitch, yaw, roll),
                    reference_poses,
                    identity_scores=identity_scores,
                    identity_verification_available=identity_available,
                    identity_threshold=FACE_MODEL_DEFAULTS.sface_same_identity_cosine,
                    minimum_gain=float(parameters.get("minimum_reference_frontal_gain", 1.5)),
                )
                source_indices = _aligned_source_indices(executor.workspace, len(references))
                selected_original = (
                    None
                    if evidence.selected_index is None
                    else source_indices[evidence.selected_index]
                )
                executor.workspace.metadata["frontal_reference_evidence"] = {
                    "accepted": evidence.accepted,
                    "selected_aligned_index": evidence.selected_index,
                    "selected_original_reference_index": selected_original,
                    "primary_frontalness": evidence.primary_frontalness,
                    "reference_frontalness": evidence.reference_frontalness,
                    "gain": evidence.gain,
                    "selected_pose": evidence.selected_pose,
                    "reason": evidence.reason,
                }

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
                            "reference_poses": reference_poses,
                            "frontal_reference_evidence": executor.workspace.metadata["frontal_reference_evidence"],
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
                    "reference_poses": reference_poses,
                    "reference_pose_backends": reference_backends,
                    "frontal_reference_evidence": executor.workspace.metadata["frontal_reference_evidence"],
                })

                landmarks = executor.workspace.metadata.get("primary_landmarks5")
                mild = None
                if landmarks is not None:
                    # With an independently observed, same-identity frontal reference we
                    # allow the normal conservative strength. Without that evidence the
                    # transform is intentionally capped lower rather than pretending that
                    # bilateral symmetry alone proves the hidden geometry.
                    max_strength = 0.45 if evidence.accepted else 0.30
                    mild = conservative_mild_frontal_affine(
                        result.image,
                        np.asarray(landmarks, dtype=np.float32),
                        box,
                        float(yaw),
                        maximum_abs_yaw=min(12.0, HEAD_POSE_DEFAULTS.max_abs_yaw_strict),
                        maximum_strength=max_strength,
                    )

                if mild is not None:
                    details["mild_yaw_frontalization"] = {
                        "applied": mild.applied,
                        "yaw_degrees": mild.yaw_degrees,
                        "strength": mild.strength,
                        "max_landmark_displacement": mild.max_landmark_displacement,
                        "supported_fraction": mild.supported_fraction,
                        "reference_evidence_strength_cap": 0.45 if evidence.accepted else 0.30,
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
                        details["engine"] = "observed-roll-plus-reference-gated-mild-yaw-frontalization"
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