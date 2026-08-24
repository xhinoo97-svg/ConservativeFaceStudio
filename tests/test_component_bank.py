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
    assert result["face_contour"].coverage == 0.0


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


def test_brow_only_reference_does_not_claim_eye_or_forehead() -> None:
    shape, landmarks, bbox = _geometry()
    masks = canonical_component_masks(shape, landmarks, bbox)
    support = masks["left_brow"].copy()
    bank = build_component_bank(
        [support],
        landmarks,
        bbox,
        source_indices=[27],
        minimum_coverage=0.50,
    )
    assert bank["left_brow"] and bank["left_brow"][0].source_index == 27
    assert bank["left_eye"] == []
    assert bank["forehead"] == []
    assert bank["face_contour"] == []
    assert np.count_nonzero((masks["left_brow"] > 0) & (masks["left_eye"] > 0)) == 0


def test_lower_face_crop_can_supply_mouth_chin_without_claiming_eyes() -> None:
    shape, landmarks, bbox = _geometry()
    masks = canonical_component_masks(shape, landmarks, bbox)
    support = cv2.bitwise_or(masks["mouth"], masks["chin"])
    support = cv2.bitwise_or(support, masks["philtrum"])
    bank = build_component_bank(
        [support],
        landmarks,
        bbox,
        source_indices=[31],
        minimum_coverage=0.50,
    )
    assert bank["mouth"] and bank["mouth"][0].source_index == 31
    assert bank["chin"] and bank["chin"][0].source_index == 31
    assert bank["philtrum"] and bank["philtrum"][0].source_index == 31
    assert bank["left_eye"] == []
    assert bank["right_eye"] == []
    assert bank["face_contour"] == []


def test_chin_is_not_double_counted_as_broad_jaw_evidence() -> None:
    shape, landmarks, bbox = _geometry()
    masks = canonical_component_masks(shape, landmarks, bbox)
    overlap = (masks["chin"] > 0) & (masks["jaw"] > 0)
    assert np.count_nonzero(overlap) == 0


def test_face_contour_is_a_nonempty_disjoint_boundary_band() -> None:
    shape, landmarks, bbox = _geometry()
    masks = canonical_component_masks(shape, landmarks, bbox)
    contour = masks["face_contour"] > 0
    assert np.count_nonzero(contour) > 0
    for name, mask in masks.items():
        if name == "face_contour":
            continue
        assert np.count_nonzero(contour & (mask > 0)) == 0, name


def test_contour_only_reference_gets_contour_authority_without_central_components() -> None:
    shape, landmarks, bbox = _geometry()
    masks = canonical_component_masks(shape, landmarks, bbox)
    bank = build_component_bank(
        [masks["face_contour"].copy()],
        landmarks,
        bbox,
        source_indices=[41],
        minimum_coverage=0.50,
    )
    assert bank["face_contour"] and bank["face_contour"][0].source_index == 41
    for name in ("left_eye", "right_eye", "nose", "mouth", "left_cheek", "right_cheek", "chin", "jaw"):
        assert bank[name] == []


def test_component_bank_exposes_all_thirteen_required_components() -> None:
    shape, landmarks, bbox = _geometry()
    assert set(canonical_component_masks(shape, landmarks, bbox)) == {
        "left_eye",
        "right_eye",
        "left_brow",
        "right_brow",
        "nose",
        "philtrum",
        "mouth",
        "left_cheek",
        "right_cheek",
        "chin",
        "jaw",
        "forehead",
        "face_contour",
    }


def test_component_bank_does_not_claim_unobserved_regions() -> None:
    shape, landmarks, bbox = _geometry()
    empty = np.zeros(shape, dtype=np.uint8)
    bank = build_component_bank([empty], landmarks, bbox, source_indices=[7])
    assert all(values == [] for values in bank.values())
