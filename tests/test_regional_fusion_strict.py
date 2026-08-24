from __future__ import annotations

import cv2
import numpy as np

from app.regional_fusion import regional_reference_fusion


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    primary = np.full((128, 128, 3), 120, dtype=np.uint8)
    reference = primary.copy()
    for y in range(24, 104, 4):
        for x in range(24, 104, 4):
            value = 55 if ((x + y) // 4) % 2 else 205
            reference[y : y + 4, x : x + 4] = value
    landmarks = np.asarray([[48, 50], [80, 50], [64, 67], [52, 85], [76, 85]], dtype=np.float32)
    bbox = (24, 20, 80, 92)
    return primary, reference, landmarks, bbox


def test_regional_fusion_never_rewrites_visible_primary_pixels() -> None:
    primary, reference, landmarks, bbox = _inputs()
    primary_mask = np.zeros(primary.shape[:2], dtype=np.uint8)
    cv2.rectangle(primary_mask, (43, 44), (54, 56), 255, -1)
    reference_mask = np.zeros_like(primary_mask)

    output, provenance, _ = regional_reference_fusion(
        [primary, reference],
        [primary_mask, reference_mask],
        landmarks,
        bbox,
        minimum_improvement=0.0,
        preserve_visible_primary=True,
    )

    visible = primary_mask == 0
    assert np.array_equal(output[visible], primary[visible])
    assert np.count_nonzero(provenance[visible]) == 0
    assert np.count_nonzero(provenance[primary_mask > 0]) > 0


def test_regional_fusion_does_not_copy_occluded_reference_pixels() -> None:
    primary, reference, landmarks, bbox = _inputs()
    primary_mask = np.zeros(primary.shape[:2], dtype=np.uint8)
    cv2.rectangle(primary_mask, (43, 44), (54, 56), 255, -1)
    reference_mask = np.zeros_like(primary_mask)
    cv2.rectangle(reference_mask, (43, 44), (48, 56), 255, -1)

    output, provenance, _ = regional_reference_fusion(
        [primary, reference],
        [primary_mask, reference_mask],
        landmarks,
        bbox,
        minimum_improvement=0.0,
        preserve_visible_primary=True,
    )

    unavailable = (primary_mask > 0) & (reference_mask > 0)
    assert np.array_equal(output[unavailable], primary[unavailable])
    assert np.count_nonzero(provenance[unavailable]) == 0
