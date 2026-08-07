from __future__ import annotations

import cv2
import numpy as np

from app.component_bank import (
    build_component_bank,
    canonical_component_masks,
    component_coverage,
    warped_support_mask,
)


def _geometry():
    shape = (160, 160)
    landmarks = np.array(
        [[58.0, 60.0], [102.0, 60.0], [80.0, 82.0], [64.0, 108.0], [96.0, 108.0]],
        dtype=np.float32,
    )
    bbox = (35, 24, 90, 116)
    return shape, landmarks, bbox


def test_warped_support_mask_tracks_only_observed_crop_pixels() -> None:
    matrix = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, 40.0]], dtype=np.float32)
    support = warped_support_mask((30, 40), matrix, (160, 160))
    assert support.shape == (160, 160)
    assert int(np.count_nonzero(support)) == 30 * 40
    assert np.all(support[40:70, 50:90] == 255)
    assert np.count_nonzero(support[:35]) == 0


def test_component_coverage_identifies_nose_only_reference() -> None:
    shape, landmarks, bbox = _geometry()
    masks = canonical_component_masks(shape, landmarks, bbox)
    support = np.zeros(shape, dtype=np.uint8)
    nose = masks["nose"] > 0
    support[nose] = 255
    result = {item.component: item for item in component_coverage(support, masks, source_index=3)}
    assert result["nose"].coverage > 0.95
    assert result["nose"].usable
    assert result["mouth"].coverage < 0.20
    assert not result["mouth"].usable


def test_component_bank_allows_different_sources_for_different_parts() -> None:
    shape, landmarks, bbox = _geometry()
    masks = canonical_component_masks(shape, landmarks, bbox)
    left_eye_support = cv2.dilate(masks["left_eye"], np.ones((5, 5), np.uint8), iterations=1)
    mouth_support = cv2.dilate(masks["mouth"], np.ones((5, 5), np.uint8), iterations=1)
    bank = build_component_bank(
        [left_eye_support, mouth_support],
        landmarks,
        bbox,
        source_indices=[11, 22],
        minimum_coverage=0.50,
    )
    assert bank["left_eye"]
    assert bank["left_eye"][0].source_index == 11
    assert bank["mouth"]
    assert bank["mouth"][0].source_index == 22


def test_component_bank_does_not_claim_unobserved_regions() -> None:
    shape, landmarks, bbox = _geometry()
    empty = np.zeros(shape, dtype=np.uint8)
    bank = build_component_bank([empty], landmarks, bbox, source_indices=[7])
    assert all(values == [] for values in bank.values())
