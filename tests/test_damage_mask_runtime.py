from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"

# Load only the narrow runtime dependencies without executing the production app bootstrap.
if "app" not in sys.modules:
    package = types.ModuleType("app")
    package.__path__ = [str(APP)]
    sys.modules["app"] = package


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


taxonomy = _load("app.damage_taxonomy", APP / "damage_taxonomy.py")
component_bank = _load("app.component_bank", APP / "component_bank.py")
runtime_module = _load("app.damage_mask_runtime", APP / "damage_mask_runtime.py")

DamageMaskRuntime = runtime_module.DamageMaskRuntime


class _Input:
    name = "image"


class FakeSession:
    def __init__(self, logits: np.ndarray) -> None:
        self.logits = np.asarray(logits)
        self.last_feed = None

    def get_inputs(self):
        return [_Input()]

    def run(self, output_names, input_feed):
        self.last_feed = input_feed
        return [self.logits.copy()]


def _geometry(size: int = 64):
    landmarks = np.array(
        [[23.0, 24.0], [41.0, 24.0], [32.0, 33.0], [26.0, 44.0], [38.0, 44.0]],
        dtype=np.float32,
    )
    bbox = (14, 10, 36, 47)
    image = np.full((size, size, 3), 100, dtype=np.uint8)
    return image, landmarks, bbox


def _base_logits(size: int = 64) -> np.ndarray:
    logits = np.full((1, len(taxonomy.DAMAGE_CLASSES), size, size), -5.0, dtype=np.float32)
    logits[:, taxonomy.HEALTHY_INDEX, :, :] = 7.0
    return logits


def test_healthy_logits_produce_no_damage_authority() -> None:
    image, landmarks, bbox = _geometry()
    session = FakeSession(_base_logits())
    runtime = DamageMaskRuntime(session=session, input_size=64)
    result = runtime.infer(image, landmarks5=landmarks, bbox=bbox)

    assert result.dominant_damage_class == "HEALTHY"
    assert result.dominant_confidence == 0.0
    assert np.count_nonzero(result.binary_damage_mask) == 0
    assert result.affected_components == ()
    assert session.last_feed is not None
    tensor = session.last_feed["image"]
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor[0, :, 0, 0], np.asarray([100, 100, 100], np.float32) / 255.0)


def test_strong_scribble_over_left_eye_reports_component_and_class() -> None:
    image, landmarks, bbox = _geometry()
    logits = _base_logits()
    masks = component_bank.canonical_component_masks((64, 64), landmarks, bbox)
    eye = masks["left_eye"] > 0
    scribble_index = taxonomy.CLASS_TO_INDEX["SCRIBBLE"]
    healthy_plane = logits[0, taxonomy.HEALTHY_INDEX]
    scribble_plane = logits[0, scribble_index]
    healthy_plane[eye] = -4.0
    scribble_plane[eye] = 9.0

    result = DamageMaskRuntime(
        session=FakeSession(logits),
        input_size=64,
        damage_confidence_threshold=0.55,
        component_fraction_threshold=0.20,
    ).infer(image, landmarks5=landmarks, bbox=bbox)

    assert result.dominant_damage_class == "SCRIBBLE"
    assert result.dominant_confidence > 0.99
    assert np.count_nonzero(result.binary_damage_mask) == np.count_nonzero(eye)
    by_name = {item.component: item for item in result.affected_components}
    assert "left_eye" in by_name
    assert by_name["left_eye"].damage_class == "SCRIBBLE"
    assert by_name["left_eye"].affected_fraction > 0.99
    assert "right_eye" not in by_name
    assert "mouth" not in by_name
    assert "face_contour" not in by_name


def test_nonhealthy_argmax_below_confidence_threshold_has_zero_routing_authority() -> None:
    image, landmarks, bbox = _geometry()
    logits = np.zeros((1, len(taxonomy.DAMAGE_CLASSES), 64, 64), dtype=np.float32)
    logits[:, taxonomy.CLASS_TO_INDEX["STICKER"], :, :] = 0.10
    result = DamageMaskRuntime(
        session=FakeSession(logits),
        input_size=64,
        damage_confidence_threshold=0.55,
    ).infer(image, landmarks5=landmarks, bbox=bbox)

    assert np.all(result.class_map == taxonomy.CLASS_TO_INDEX["STICKER"])
    assert np.count_nonzero(result.binary_damage_mask) == 0
    assert result.dominant_damage_class == "HEALTHY"
    assert result.affected_components == ()


def test_runtime_rejects_wrong_number_of_output_classes() -> None:
    image, landmarks, bbox = _geometry()
    logits = np.zeros((1, len(taxonomy.DAMAGE_CLASSES) - 1, 64, 64), dtype=np.float32)
    runtime = DamageMaskRuntime(session=FakeSession(logits), input_size=64)
    with pytest.raises(RuntimeError, match="logits shape mismatch"):
        runtime.infer(image, landmarks5=landmarks, bbox=bbox)


def test_runtime_rejects_nonfinite_logits() -> None:
    image, landmarks, bbox = _geometry()
    logits = _base_logits()
    logits[0, 0, 3, 3] = np.nan
    runtime = DamageMaskRuntime(session=FakeSession(logits), input_size=64)
    with pytest.raises(RuntimeError, match="non-finite logits"):
        runtime.infer(image, landmarks5=landmarks, bbox=bbox)


def test_runtime_rejects_bad_input_contract() -> None:
    image, landmarks, bbox = _geometry()
    runtime = DamageMaskRuntime(session=FakeSession(_base_logits()), input_size=64)
    with pytest.raises(ValueError, match="uint8 BGR"):
        runtime.infer(image.astype(np.float32), landmarks5=landmarks, bbox=bbox)


class BadInputSession(FakeSession):
    def get_inputs(self):
        return [_Input(), _Input()]


def test_runtime_rejects_onnx_with_multiple_inputs() -> None:
    with pytest.raises(RuntimeError, match="exactly one named input"):
        DamageMaskRuntime(session=BadInputSession(_base_logits()), input_size=64)


def test_research_generator_imports_the_same_frozen_taxonomy() -> None:
    source = (ROOT / "research" / "damage_mask_dataset.py").read_text(encoding="utf-8")
    assert "from app.damage_taxonomy import CLASS_TO_INDEX, DAMAGE_CLASSES" in source
    assert "DAMAGE_CLASSES = (" not in source
