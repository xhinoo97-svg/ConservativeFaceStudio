from __future__ import annotations

import cv2
import numpy as np

from app.frontalization import (
    conservative_mild_frontal_affine,
    select_more_frontal_reference,
    warp_auxiliary_map,
)


def _face() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image = np.full((160, 160, 3), 24, dtype=np.uint8)
    bbox = (36, 26, 88, 112)
    cv2.ellipse(image, (80, 82), (43, 55), 0, 0, 360, (150, 175, 205), -1)
    cv2.circle(image, (59, 62), 4, (24, 24, 24), -1)
    cv2.circle(image, (101, 64), 4, (24, 24, 24), -1)
    cv2.line(image, (86, 68), (88, 89), (75, 88, 102), 2)
    cv2.line(image, (64, 106), (99, 108), (55, 55, 70), 2)
    landmarks = np.array(
        [[59.0, 62.0], [101.0, 64.0], [88.0, 82.0], [64.0, 106.0], [99.0, 108.0]],
        dtype=np.float32,
    )
    return image, landmarks, bbox


def _horizontal_asymmetry(points: np.ndarray) -> float:
    eye_mid = float((points[0, 0] + points[1, 0]) * 0.5)
    nose_error = abs(float(points[2, 0]) - eye_mid)
    mouth_mid = float((points[3, 0] + points[4, 0]) * 0.5)
    mouth_error = abs(mouth_mid - eye_mid)
    level_error = abs(float(points[0, 1] - points[1, 1])) + abs(float(points[3, 1] - points[4, 1]))
    return nose_error + mouth_error + 0.25 * level_error


def test_mild_yaw_frontalization_reduces_landmark_asymmetry_without_synthesis() -> None:
    image, landmarks, bbox = _face()
    before = _horizontal_asymmetry(landmarks)

    result = conservative_mild_frontal_affine(image, landmarks, bbox, yaw_degrees=8.0)

    assert result.applied
    assert result.supported_fraction >= 0.985
    assert result.max_landmark_displacement <= max(bbox[2], bbox[3]) * 0.065
    assert _horizontal_asymmetry(result.transformed_landmarks) < before
    assert np.count_nonzero(result.changed_mask) > 0
    assert np.array_equal(result.image[result.changed_mask == 0], image[result.changed_mask == 0])


def test_frontalization_abstains_on_large_yaw() -> None:
    image, landmarks, bbox = _face()
    result = conservative_mild_frontal_affine(image, landmarks, bbox, yaw_degrees=19.0)
    assert not result.applied
    assert np.array_equal(result.image, image)
    assert np.count_nonzero(result.changed_mask) == 0


def test_auxiliary_map_changes_only_where_image_transform_is_active() -> None:
    image, landmarks, bbox = _face()
    result = conservative_mild_frontal_affine(image, landmarks, bbox, yaw_degrees=8.0)
    assert result.applied
    provenance = np.zeros(image.shape[:2], dtype=np.uint16)
    provenance[50:115, 48:112] = 2
    warped = warp_auxiliary_map(provenance, result.matrix, result.changed_mask)
    assert warped.dtype == provenance.dtype
    assert np.array_equal(warped[result.changed_mask == 0], provenance[result.changed_mask == 0])


def test_reference_evidence_selects_verified_more_frontal_photo() -> None:
    evidence = select_more_frontal_reference(
        (3.0, 9.0, 1.0),
        [(2.0, 7.0, 1.0), (0.8, 1.5, 0.5), (1.0, 0.8, 0.2)],
        identity_scores=[0.81, 0.76, 0.72],
        identity_verification_available=True,
        identity_threshold=0.363,
    )
    assert evidence.accepted
    assert evidence.selected_index == 2
    assert evidence.reference_frontalness is not None
    assert evidence.reference_frontalness < evidence.primary_frontalness
    assert evidence.gain >= 1.5


def test_reference_evidence_rejects_frontal_wrong_identity() -> None:
    evidence = select_more_frontal_reference(
        (2.0, 8.0, 0.5),
        [(0.2, 0.4, 0.1), (1.0, 6.5, 0.5)],
        identity_scores=[0.12, 0.80],
        identity_verification_available=True,
        identity_threshold=0.363,
    )
    assert evidence.accepted
    assert evidence.selected_index == 1
    assert evidence.selected_pose != (0.2, 0.4, 0.1)


def test_reference_evidence_abstains_when_no_meaningful_pose_gain() -> None:
    evidence = select_more_frontal_reference(
        (1.0, 3.0, 0.5),
        [(0.8, 2.8, 0.4), (1.0, 3.1, 0.2)],
        identity_scores=[0.8, 0.8],
        identity_verification_available=True,
        minimum_gain=1.5,
    )
    assert not evidence.accepted
    assert evidence.gain < 1.5
