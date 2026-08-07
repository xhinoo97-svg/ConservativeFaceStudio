from __future__ import annotations

import cv2
import numpy as np

from app.reference_inpainting import verified_reference_repair


def _face() -> np.ndarray:
    image = np.full((128, 128, 3), 28, dtype=np.uint8)
    cv2.ellipse(image, (64, 65), (41, 52), 0, 0, 360, (145, 170, 198), -1)
    cv2.circle(image, (50, 53), 4, (25, 25, 25), -1)
    cv2.circle(image, (78, 53), 4, (25, 25, 25), -1)
    cv2.line(image, (64, 60), (63, 76), (80, 92, 104), 2)
    cv2.line(image, (52, 87), (76, 87), (55, 55, 70), 2)
    return image


def _target() -> np.ndarray:
    mask = np.zeros((128, 128), dtype=np.uint8)
    cv2.rectangle(mask, (44, 47), (84, 78), 255, -1)
    return mask


def test_verified_reference_repair_recovers_sticker_from_matching_references() -> None:
    clean = _face()
    damaged = clean.copy()
    mask = _target()
    damaged[mask > 0] = (245, 30, 180)

    result = verified_reference_repair(
        damaged,
        [clean.copy(), clean.copy()],
        mask,
        identity_scores=[0.8, 0.82],
        identity_verification_available=True,
        minimum_context_score=0.20,
        agreement_threshold=5.0,
        feather_sigma=0.0,
    )

    assert result.requested_pixels > 0
    assert result.repaired_pixels == result.requested_pixels
    assert result.unresolved_pixels == 0
    assert set(np.unique(result.provenance_map)) <= {0, 1, 2}
    assert np.array_equal(result.image[mask == 0], damaged[mask == 0])
    assert np.mean(np.abs(result.image.astype(np.int16) - clean.astype(np.int16))) < np.mean(
        np.abs(damaged.astype(np.int16) - clean.astype(np.int16))
    )


def test_identity_filter_rejects_wrong_reference() -> None:
    clean = _face()
    damaged = clean.copy()
    mask = _target()
    damaged[mask > 0] = 0
    wrong = np.full_like(clean, 235)

    result = verified_reference_repair(
        damaged,
        [wrong],
        mask,
        identity_scores=[0.10],
        identity_threshold=0.363,
        identity_verification_available=True,
        minimum_context_score=0.0,
    )

    assert result.repaired_pixels == 0
    assert result.unresolved_pixels == result.requested_pixels
    assert np.array_equal(result.image, damaged)


def test_reference_disagreement_is_left_unresolved() -> None:
    clean = _face()
    damaged = clean.copy()
    mask = _target()
    damaged[mask > 0] = 0
    conflicting = clean.copy()
    conflicting[mask > 0] = (250, 250, 250)

    result = verified_reference_repair(
        damaged,
        [clean, conflicting],
        mask,
        identity_scores=[0.8, 0.8],
        identity_verification_available=True,
        minimum_context_score=0.0,
        agreement_threshold=3.0,
        feather_sigma=0.0,
    )

    assert result.agreement_rejected_pixels > 0
    assert result.unresolved_pixels > 0
    assert np.array_equal(result.image[result.unresolved_mask > 0], damaged[result.unresolved_mask > 0])


def test_local_alignment_corrects_small_reference_shift() -> None:
    clean = _face()
    damaged = clean.copy()
    mask = _target()
    damaged[mask > 0] = 0
    matrix = np.float32([[1, 0, 3], [0, 1, -2]])
    shifted = cv2.warpAffine(clean, matrix, (128, 128), borderMode=cv2.BORDER_REFLECT)

    result = verified_reference_repair(
        damaged,
        [shifted],
        mask,
        identity_scores=[0.8],
        identity_verification_available=True,
        max_local_shift=5,
        minimum_context_score=0.20,
        feather_sigma=0.0,
    )

    assert result.repaired_pixels > 0
    assert result.local_shifts
    dx, dy = result.local_shifts[0]
    assert abs(dx) <= 5 and abs(dy) <= 5


def test_local_alignment_accepts_same_face_under_exposure_change() -> None:
    clean = _face()
    damaged = clean.copy()
    mask = _target()
    damaged[mask > 0] = (245, 30, 180)

    brighter = cv2.convertScaleAbs(clean, alpha=1.08, beta=24)
    matrix = np.float32([[1, 0, 2], [0, 1, -1]])
    brighter_shifted = cv2.warpAffine(brighter, matrix, (128, 128), borderMode=cv2.BORDER_REFLECT)

    result = verified_reference_repair(
        damaged,
        [brighter_shifted],
        mask,
        identity_scores=[0.8],
        identity_verification_available=True,
        max_local_shift=5,
        minimum_context_score=0.42,
        feather_sigma=0.0,
    )

    assert result.repaired_pixels > 0
    assert result.context_scores and result.context_scores[0] >= 0.42
    assert result.local_shifts
    assert np.array_equal(result.image[mask == 0], damaged[mask == 0])


def test_local_photometric_matching_reduces_visible_exposure_seam() -> None:
    clean = _face()
    damaged = clean.copy()
    mask = _target()
    damaged[mask > 0] = (245, 30, 180)
    brighter = cv2.convertScaleAbs(clean, alpha=1.0, beta=16)

    result = verified_reference_repair(
        damaged,
        [brighter],
        mask,
        identity_scores=[0.8],
        identity_verification_available=True,
        minimum_context_score=0.20,
        feather_sigma=0.0,
    )

    assert result.repaired_pixels == result.requested_pixels
    assert result.photometric_offsets_lab
    assert abs(result.photometric_offsets_lab[0][0]) > 1.0
    repaired_error = float(np.mean(np.abs(result.image[mask > 0].astype(np.int16) - clean[mask > 0].astype(np.int16))))
    raw_reference_error = float(np.mean(np.abs(brighter[mask > 0].astype(np.int16) - clean[mask > 0].astype(np.int16))))
    assert repaired_error < raw_reference_error
    assert np.array_equal(result.image[mask == 0], damaged[mask == 0])
