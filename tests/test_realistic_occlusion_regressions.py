from __future__ import annotations

import cv2
import numpy as np

from app.restoration import detect_occlusion_candidates


def _smooth_partial_face() -> np.ndarray:
    image = np.full((180, 260, 3), 190, dtype=np.uint8)
    cv2.ellipse(image, (130, 95), (70, 88), 0, 0, 360, (170, 155, 145), -1)
    cv2.ellipse(image, (130, 120), (24, 8), 0, 0, 180, (110, 95, 95), 2)
    # Bright smooth hair/background regions deliberately mimic the real partial
    # photographs used during development; these must not become occlusions simply
    # because their local variance is low.
    cv2.rectangle(image, (0, 0), (55, 179), (232, 232, 232), -1)
    cv2.rectangle(image, (205, 0), (259, 179), (238, 238, 238), -1)
    return cv2.GaussianBlur(image, (0, 0), 1.6)


def test_smooth_face_and_bright_flat_regions_are_not_mass_occlusions() -> None:
    image = _smooth_partial_face()
    mask = detect_occlusion_candidates(image)
    assert float(np.mean(mask > 0)) < 0.03


def test_opaque_black_scribble_remains_detectable() -> None:
    image = _smooth_partial_face()
    cv2.rectangle(image, (82, 55), (178, 100), (0, 0, 0), -1)
    for y in range(58, 99, 8):
        cv2.line(image, (70, y), (190, y + 3), (0, 0, 0), 5)
    mask = detect_occlusion_candidates(image)
    scribble = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.rectangle(scribble, (70, 52), (190, 108), 255, -1)
    overlap = np.count_nonzero((mask > 0) & (scribble > 0)) / max(1, np.count_nonzero(scribble))
    assert overlap > 0.45


def test_unmarked_pixels_are_preserved_as_majority() -> None:
    image = _smooth_partial_face()
    cv2.rectangle(image, (90, 60), (150, 88), (0, 0, 0), -1)
    mask = detect_occlusion_candidates(image)
    assert float(np.mean(mask == 0)) > 0.80
