from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _module():
    sys.path.insert(0, str(RESEARCH))
    try:
        return importlib.import_module("build_phase04_controlface_bank")
    finally:
        if sys.path and sys.path[0] == str(RESEARCH):
            sys.path.pop(0)


def test_controlface_revision_is_pinned() -> None:
    module = _module()
    assert module.CONTROLFACE_REVISION == "a03589de1a9e028b2d16fa1eb0e019a6930e817c"
    assert module.CONTROLFACE_URL.startswith("https://huggingface.co/datasets/HuMInGameLab/ControlFace10K/")


def test_normalization_is_center_square_resize_only() -> None:
    module = _module()
    image = np.zeros((120, 200, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(200, dtype=np.uint8)[None, :]
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    result = module._decode_and_center_face(encoded.tobytes(), output_size=96)
    assert result.shape == (96, 96, 3)
    assert result.dtype == np.uint8
    # Center square from a 200-wide image starts at x=40; this proves we are not
    # padding or synthesizing pixels before the ordinary resize.
    assert int(result[48, 0, 0]) >= 35
    assert int(result[48, -1, 0]) <= 165


def test_normalization_rejects_too_small_images() -> None:
    module = _module()
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    with pytest.raises(RuntimeError, match="too small"):
        module._decode_and_center_face(encoded.tobytes(), output_size=128)


def test_output_size_is_fail_closed() -> None:
    module = _module()
    with pytest.raises(ValueError, match="output_size must be >= 64"):
        module.build(
            output_dir=Path("unused"),
            manifest_path=Path("unused.json"),
            output_size=32,
        )
