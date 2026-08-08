from __future__ import annotations

import cv2
import numpy as np

from app.strict_repair import (
    conservative_roll_normalize,
    reference_consensus_occlusion_mask,
    repair_from_observed_references,
)


def clean_face() -> np.ndarray:
    image = np.full((128, 128, 3), 35, dtype=np.uint8)
    cv2.ellipse(image, (64, 66), (40, 52), 0, 0, 360, (135, 165, 195), -1)
    cv2.circle(image, (50, 54), 4, (30, 30, 30), -1)
    cv2.circle(image, (78, 54), 4, (30, 30, 30), -1)
    cv2.line(image, (64, 60), (64, 78), (70, 80, 90), 2)
    cv2.line(image, (52, 88), (76, 88), (55, 55, 75), 2)
    return image


def test_reference_consensus_repairs_only_supported_occlusion() -> None:
    clean = clean_face()
    primary = clean.copy()
    primary[48:78, 45:83] = 0
    hint = np.zeros((128, 128), dtype=np.uint8)
    hint[48:78, 45:83] = 255
    masks = [np.zeros((128, 128), dtype=np.uint8), np.zeros((128, 128), dtype=np.uint8)]

    target = reference_consensus_occlusion_mask(primary, [clean, clean.copy()], hint, masks)
    assert np.count_nonzero(target) > 0
    repaired = repair_from_observed_references(primary, [clean, clean.copy()], target, masks, feather_sigma=0)
    assert repaired.repaired_pixels == np.count_nonzero(target)
    assert repaired.unresolved_pixels == 0
    assert np.array_equal(repaired.image[target == 0], primary[target == 0])
    assert np.mean(np.abs(repaired.image.astype(np.int16) - clean.astype(np.int16))) < np.mean(
        np.abs(primary.astype(np.int16) - clean.astype(np.int16))
    )


def test_complementary_partial_references_can_confirm_hint_without_overlap() -> None:
    clean = clean_face()
    primary = clean.copy()
    primary[50:78, 42:86] = 0
    hint = np.zeros((128, 128), dtype=np.uint8)
    hint[50:78, 42:86] = 255

    left = np.zeros_like(clean)
    left[:, :68] = clean[:, :68]
    right = np.zeros_like(clean)
    right[:, 60:] = clean[:, 60:]
    left_mask = np.full((128, 128), 255, dtype=np.uint8)
    left_mask[:, :68] = 0
    right_mask = np.full((128, 128), 255, dtype=np.uint8)
    right_mask[:, 60:] = 0

    target = reference_consensus_occlusion_mask(
        primary,
        [left, right],
        hint,
        [left_mask, right_mask],
    )
    repaired = repair_from_observed_references(
        primary,
        [left, right],
        target,
        [left_mask, right_mask],
        feather_sigma=0,
    )

    hinted = hint > 0
    assert np.count_nonzero(target[hinted]) / np.count_nonzero(hinted) >= 0.90
    assert repaired.unresolved_pixels == 0
    assert np.mean(np.abs(repaired.image[hinted].astype(np.int16) - clean[hinted].astype(np.int16))) < 1.0


def test_reference_consensus_detects_local_blur_supported_by_two_sharp_references() -> None:
    clean = clean_face()
    primary = clean.copy()
    region = primary[42:90, 40:88].copy()
    primary[42:90, 40:88] = cv2.GaussianBlur(region, (0, 0), 5.0)
    no_hint = np.zeros((128, 128), dtype=np.uint8)

    target = reference_consensus_occlusion_mask(
        primary,
        [clean, clean.copy()],
        no_hint,
        maximum_fraction=0.35,
    )

    assert target.dtype == np.uint8
    assert set(np.unique(target)).issubset({0, 255})
    assert np.count_nonzero(target) > 0
    detected_inside = np.count_nonzero(target[42:90, 40:88])
    detected_total = np.count_nonzero(target)
    assert detected_inside / max(1, detected_total) >= 0.70
    repaired = repair_from_observed_references(
        primary, [clean, clean.copy()], target, feather_sigma=0
    )
    before_error = np.mean(np.abs(primary.astype(np.int16) - clean.astype(np.int16)))
    after_error = np.mean(np.abs(repaired.image.astype(np.int16) - clean.astype(np.int16)))
    assert after_error < before_error


def test_reference_consensus_does_not_mark_clean_face_as_blurred() -> None:
    clean = clean_face()
    no_hint = np.zeros((128, 128), dtype=np.uint8)
    target = reference_consensus_occlusion_mask(
        clean, [clean.copy(), clean.copy()], no_hint
    )
    assert np.count_nonzero(target) == 0


def test_reference_consensus_abstains_when_references_disagree() -> None:
    clean = clean_face()
    primary = clean.copy()
    primary[48:78, 45:83] = 0
    hint = np.zeros((128, 128), dtype=np.uint8)
    hint[48:78, 45:83] = 255
    opposite = np.full_like(clean, 245)
    target = reference_consensus_occlusion_mask(primary, [clean, opposite], hint)
    assert np.count_nonzero(target) == 0


def test_single_reference_requires_hint() -> None:
    clean = clean_face()
    primary = clean.copy()
    primary[48:78, 45:83] = 0
    no_hint = np.zeros((128, 128), dtype=np.uint8)
    target = reference_consensus_occlusion_mask(primary, [clean], no_hint)
    assert np.count_nonzero(target) == 0


def test_pose_normalization_never_synthesizes_large_roll() -> None:
    image = clean_face()
    landmarks = np.array([[45, 45], [80, 70], [64, 68], [53, 88], [75, 88]], dtype=np.float32)
    result = conservative_roll_normalize(image, landmarks, maximum_angle=12.0)
    assert not result.applied
    assert np.array_equal(result.image, image)


def test_pose_normalization_accepts_already_level_face_without_change() -> None:
    image = clean_face()
    landmarks = np.array([[50, 54], [78, 54], [64, 68], [53, 88], [75, 88]], dtype=np.float32)
    result = conservative_roll_normalize(image, landmarks)
    assert not result.applied
    assert result.supported_fraction == 1.0
    assert np.array_equal(result.image, image)
