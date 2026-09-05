from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"


def _module():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(RESEARCH))
    try:
        return importlib.import_module("phase04_face_crop")
    finally:
        for value in (str(RESEARCH), str(ROOT)):
            if value in sys.path:
                sys.path.remove(value)


@dataclass(frozen=True)
class _Observation:
    bbox: tuple[int, int, int, int]
    score: float


class _Engine:
    def __init__(self, bbox=(60, 30, 80, 100), score=0.95):
        self.observation = _Observation(bbox=bbox, score=score)

    def analyze(self, image):
        return self.observation


class _ScaleSensitiveEngine:
    def analyze(self, image):
        h, w = image.shape[:2]
        if max(h, w) > 1280:
            raise ValueError("input too large for stable synthetic detector")
        return _Observation(
            bbox=(int(0.30 * w), int(0.20 * h), int(0.35 * w), int(0.45 * h)),
            score=0.91,
        )


def test_context_square_contains_face_and_stays_inside_image() -> None:
    module = _module()
    crop = module._context_square((0, 10, 80, 100), width=180, height=160, context_scale=1.35)
    x, y, w, h = crop
    assert w == h
    assert x >= 0 and y >= 0
    assert x + w <= 180 and y + h <= 160
    assert x <= 0 + 80 and y <= 10 + 100


def test_crop_main_face_returns_only_observed_source_pixels_resized() -> None:
    module = _module()
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    for row in range(image.shape[0]):
        image[row, :, :] = row % 255
    result = module.crop_main_face(image, _Engine(), output_size=128, context_scale=1.25)
    assert result.image.shape == (128, 128, 3)
    assert result.image.dtype == np.uint8
    assert result.detector_backend == "opencv_zoo_yunet_multiscale"
    assert result.detector_score == pytest.approx(0.95)
    assert result.detector_input_scale == pytest.approx(1.0)
    x, y, w, h = result.crop_bbox
    assert w == h and w > 0
    assert 0 <= x < 200 and 0 <= y < 160
    assert x + w <= 200 and y + h <= 160


def test_multiscale_detector_maps_bbox_back_to_high_resolution_source() -> None:
    module = _module()
    image = np.zeros((2400, 1800, 3), dtype=np.uint8)
    result = module.crop_main_face(
        image,
        _ScaleSensitiveEngine(),
        output_size=128,
        context_scale=1.25,
        detector_max_dimensions=(640, 960, 1280),
    )
    assert result.detector_score == pytest.approx(0.91)
    assert 0.0 < result.detector_input_scale < 1.0
    x, y, w, h = result.source_bbox
    assert 400 <= x <= 700
    assert 350 <= y <= 650
    assert 500 <= w <= 750
    assert 900 <= h <= 1200


def test_crop_main_face_rejects_low_confidence_detection() -> None:
    module = _module()
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="below required confidence"):
        module.crop_main_face(image, _Engine(score=0.50), minimum_detector_score=0.75)
