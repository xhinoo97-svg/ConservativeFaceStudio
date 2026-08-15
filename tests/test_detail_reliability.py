from __future__ import annotations

import cv2
import numpy as np

from app.restoration import detail_reliability_map


def _striped(size: int = 128) -> np.ndarray:
    yy, xx = np.indices((size, size))
    pattern = (((xx // 4) + (yy // 4)) % 2 * 180 + 35).astype(np.uint8)
    return cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)


def test_detail_reliability_separates_observed_detail_from_blur() -> None:
    sharp = _striped()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 8.0)
    sharp_map = detail_reliability_map(sharp)
    blur_map = detail_reliability_map(blurred)
    assert sharp_map.dtype == np.uint8
    assert blur_map.dtype == np.uint8
    assert float(np.mean(sharp_map)) > float(np.mean(blur_map)) + 45.0
    assert float(np.mean(blur_map)) < 80.0


def test_detail_reliability_keeps_occlusion_separate_and_zeroes_it() -> None:
    image = _striped()
    occlusion = np.zeros(image.shape[:2], dtype=np.uint8)
    occlusion[30:90, 40:100] = 255
    reliability = detail_reliability_map(image, occlusion)
    assert np.all(reliability[30:90, 40:100] == 0)
    assert int(np.count_nonzero(reliability[:20])) > 0
