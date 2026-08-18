from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if "app" not in sys.modules:
    package = types.ModuleType("app")
    package.__path__ = [str(APP)]
    sys.modules["app"] = package

spec = importlib.util.spec_from_file_location("app.face_parsing_runtime", APP / "face_parsing_runtime.py")
assert spec is not None and spec.loader is not None
runtime_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runtime_module
spec.loader.exec_module(runtime_module)

FaceParsingRuntime = runtime_module.FaceParsingRuntime
one_hot_parsing = runtime_module.one_hot_parsing


class _Input:
    name = "input"


class _Output:
    name = "output"


class FakeSession:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = np.asarray(logits)
        self.last_feed = None

    def get_inputs(self):
        return [_Input()]

    def get_outputs(self):
        return [_Output()]

    def run(self, output_names, input_feed):
        self.last_feed = input_feed
        return [self.logits.copy()]


def _logits(class_index: int = 1) -> np.ndarray:
    value = np.full((1, 19, 512, 512), -2.0, dtype=np.float32)
    value[:, class_index, :, :] = 3.0
    return value


def test_taxonomy_has_background_plus_eighteen_celebamask_attributes() -> None:
    assert len(runtime_module.CELEBAMASK_HQ_CLASSES) == 19
    assert runtime_module.CELEBAMASK_HQ_CLASSES[0] == "background"
    assert runtime_module.CELEBAMASK_HQ_CLASSES[1:] == (
        "skin", "l_brow", "r_brow", "l_eye", "r_eye", "eye_g",
        "l_ear", "r_ear", "ear_r", "nose", "mouth", "u_lip", "l_lip",
        "neck", "neck_l", "cloth", "hair", "hat",
    )


def test_preprocess_matches_upstream_bgr_rgb_imagenet_contract() -> None:
    image = np.zeros((100, 80, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # BGR red -> RGB [1,0,0]
    tensor = FaceParsingRuntime.preprocess(image)
    assert tensor.shape == (1, 3, 512, 512)
    assert tensor.dtype == np.float32
    expected = np.asarray([
        (1.0 - 0.485) / 0.229,
        (0.0 - 0.456) / 0.224,
        (0.0 - 0.406) / 0.225,
    ], dtype=np.float32)
    np.testing.assert_allclose(tensor[0, :, 10, 10], expected, atol=1e-6)


def test_predict_argmax_and_nearest_resize_preserve_integer_labels() -> None:
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    session = FakeSession(_logits(class_index=10))
    runtime = FaceParsingRuntime(session=session)
    labels = runtime.predict(image)
    assert labels.shape == (64, 96)
    assert labels.dtype == np.uint8
    assert np.all(labels == 10)
    assert session.last_feed is not None
    assert session.last_feed["input"].shape == (1, 3, 512, 512)


def test_one_hot_parsing_matches_ref_face_19_channel_contract() -> None:
    labels = np.asarray([[0, 1], [10, 18]], dtype=np.uint8)
    one_hot = one_hot_parsing(labels)
    assert one_hot.shape == (19, 2, 2)
    assert np.allclose(np.sum(one_hot, axis=0), 1.0)
    assert one_hot[0, 0, 0] == 1.0
    assert one_hot[1, 0, 1] == 1.0
    assert one_hot[10, 1, 0] == 1.0
    assert one_hot[18, 1, 1] == 1.0


def test_runtime_rejects_wrong_channel_count_and_nan() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    bad_shape = np.zeros((1, 18, 512, 512), dtype=np.float32)
    with pytest.raises(RuntimeError, match="logits shape invalid"):
        FaceParsingRuntime(session=FakeSession(bad_shape)).predict(image)

    bad_nan = _logits()
    bad_nan[0, 0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        FaceParsingRuntime(session=FakeSession(bad_nan)).predict(image)


def test_runtime_rejects_invalid_image_and_labels() -> None:
    with pytest.raises(ValueError, match="uint8 BGR"):
        FaceParsingRuntime.preprocess(np.zeros((64, 64, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="out-of-range"):
        one_hot_parsing(np.asarray([[19]], dtype=np.int32))
