from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FaceModelDefaults:
    # OpenCV Zoo demos/reference implementation.
    yunet_score_threshold: float = 0.75
    yunet_nms_threshold: float = 0.30
    yunet_top_k: int = 5000
    sface_same_identity_cosine: float = 0.363


@dataclass(frozen=True)
class ParsingDefaults:
    # yakhyo/face-parsing ONNX inference preprocessing.
    input_size: int = 512
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class HeadPoseDefaults:
    # Same ImageNet normalization used by the upstream pretrained model.
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    # Project safety gates, NOT learned weights.
    max_abs_yaw_strict: float = 20.0
    max_abs_pitch_strict: float = 15.0
    max_abs_roll_correction: float = 12.0


@dataclass(frozen=True)
class RestorationSafetyDefaults:
    # Project-calibrated conservative runtime values, NOT pretrained parameters.
    nafnet_observed_blend: float = 0.60
    identity_minimum_retention: float = 0.95
    identity_max_drop: float = 0.05
    tile_overlap: int = 32


FACE_MODEL_DEFAULTS = FaceModelDefaults()
PARSING_DEFAULTS = ParsingDefaults()
HEAD_POSE_DEFAULTS = HeadPoseDefaults()
RESTORATION_SAFETY_DEFAULTS = RestorationSafetyDefaults()


def validate_defaults() -> None:
    assert 0.0 < FACE_MODEL_DEFAULTS.yunet_score_threshold <= 1.0
    assert 0.0 < FACE_MODEL_DEFAULTS.sface_same_identity_cosine < 1.0
    assert PARSING_DEFAULTS.input_size == 512
    assert 0.0 < RESTORATION_SAFETY_DEFAULTS.identity_minimum_retention <= 1.0
    assert 0.0 <= RESTORATION_SAFETY_DEFAULTS.nafnet_observed_blend <= 1.0
