from __future__ import annotations

import numpy as np

from app.opencv_semantic_models import FaceParsingEngine, HeadPoseEngine
from app.pretrained_values import FACE_MODEL_DEFAULTS, PARSING_DEFAULTS, validate_defaults
from app.standard_pretrained import STANDARD_MODELS, STANDARD_GENERATIVE_KEYS, STANDARD_STRICT_KEYS, standard_manifest_by_key


def test_standard_pretrained_models_have_pinned_hashes() -> None:
    assert len(STANDARD_MODELS) == 4
    for manifest in STANDARD_MODELS:
        assert manifest.source_url is not None
        assert manifest.expected_sha256 is not None
        assert len(manifest.expected_sha256) == 64


def test_expected_production_models_are_registered() -> None:
    registry = standard_manifest_by_key()
    assert set(registry) == {
        "opencv_nafnet_deblur",
        "face_parsing_resnet18_onnx",
        "head_pose_mobilenetv2_onnx",
        "opencv_lama_inpaint",
    }
    assert set(STANDARD_STRICT_KEYS).isdisjoint(set(STANDARD_GENERATIVE_KEYS))
    assert registry["opencv_lama_inpaint"].conservative_default is False


def test_official_runtime_values_are_sane() -> None:
    validate_defaults()
    assert FACE_MODEL_DEFAULTS.sface_same_identity_cosine == 0.363
    assert PARSING_DEFAULTS.input_size == 512


def test_face_parsing_semantic_masks_are_disjoint_for_accessories() -> None:
    labels = np.array([[0, 1, 2, 3, 14, 15, 17]], dtype=np.uint8)
    support = FaceParsingEngine.support_mask(labels)
    accessories = FaceParsingEngine.accessory_mask(labels)
    assert support[0, 1] == 255
    assert support[0, 2] == 255
    assert support[0, 6] == 255
    assert accessories[0, 3] == 255
    assert accessories[0, 4] == 255
    assert not np.any((support > 0) & (accessories > 0))


def test_head_pose_rotation_identity_is_zero() -> None:
    pitch, yaw, roll = HeadPoseEngine.rotation_matrix_to_euler(np.eye(3, dtype=np.float32))
    assert abs(pitch) < 1e-6
    assert abs(yaw) < 1e-6
    assert abs(roll) < 1e-6
