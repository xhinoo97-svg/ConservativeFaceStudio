from __future__ import annotations

import cv2
import numpy as np

from app.reference_inpainting import verified_reference_repair


def _face() -> np.ndarray:
    image = np.full((160, 160, 3), 32, dtype=np.uint8)
    cv2.ellipse(image, (80, 82), (48, 61), 0, 0, 360, (142, 169, 197), -1)
    cv2.circle(image, (62, 68), 5, (24, 24, 24), -1)
    cv2.circle(image, (98, 68), 5, (24, 24, 24), -1)
    cv2.line(image, (80, 75), (79, 94), (78, 88, 100), 2)
    cv2.ellipse(image, (80, 109), (18, 7), 0, 0, 180, (50, 50, 72), 3)
    return image


def _mask() -> np.ndarray:
    mask = np.zeros((160, 160), dtype=np.uint8)
    cv2.rectangle(mask, (55, 58), (105, 98), 255, -1)
    return mask


def test_two_same_face_references_can_agree_despite_exposure_and_white_balance() -> None:
    clean = _face()
    target = _mask()
    damaged = clean.copy()
    damaged[target > 0] = (245, 35, 185)

    brighter = cv2.convertScaleAbs(clean, alpha=1.10, beta=20)
    warmer = clean.astype(np.int16)
    warmer[..., 0] -= 8
    warmer[..., 1] += 5
    warmer[..., 2] += 18
    warmer = np.clip(warmer, 0, 255).astype(np.uint8)

    result = verified_reference_repair(
        damaged,
        [brighter, warmer],
        target,
        identity_scores=[0.84, 0.86],
        identity_verification_available=True,
        minimum_context_score=0.20,
        agreement_threshold=16.0,
        feather_sigma=0.0,
    )

    assert result.repaired_pixels > int(result.requested_pixels * 0.90)
    assert result.unresolved_pixels < int(result.requested_pixels * 0.10)
    assert np.array_equal(result.image[target == 0], damaged[target == 0])
    assert set(np.unique(result.provenance_map)) <= {0, 1, 2}


def test_exposure_robust_consensus_still_rejects_structural_conflict() -> None:
    clean = _face()
    target = _mask()
    damaged = clean.copy()
    damaged[target > 0] = 0

    brighter = cv2.convertScaleAbs(clean, alpha=1.08, beta=18)
    conflicting = cv2.convertScaleAbs(clean, alpha=0.92, beta=8)
    cv2.rectangle(conflicting, (57, 60), (103, 96), (250, 250, 250), -1)

    result = verified_reference_repair(
        damaged,
        [brighter, conflicting],
        target,
        identity_scores=[0.85, 0.85],
        identity_verification_available=True,
        minimum_context_score=0.0,
        agreement_threshold=4.0,
        feather_sigma=0.0,
    )

    assert result.agreement_rejected_pixels > 0
    assert result.unresolved_pixels > 0
