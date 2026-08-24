from __future__ import annotations

import cv2
import numpy as np

from app.regional_fusion import facial_region_masks, regional_reference_fusion
from app.validation import evaluate_identity_guardrail, synthetic_degradations, validation_metrics


def face_like() -> np.ndarray:
    image = np.zeros((160, 160, 3), np.uint8)
    cv2.ellipse(image, (80, 82), (50, 65), 0, 0, 360, (135, 165, 195), -1)
    cv2.circle(image, (62, 67), 6, (25, 25, 25), -1)
    cv2.circle(image, (98, 67), 6, (25, 25, 25), -1)
    cv2.line(image, (80, 74), (80, 96), (70, 80, 90), 3)
    cv2.ellipse(image, (80, 112), (20, 8), 0, 0, 180, (40, 40, 80), 3)
    for x in range(35, 126, 8):
        cv2.line(image, (x, 35), (x + 3, 45), (90, 120, 150), 1)
    return image


def landmarks() -> np.ndarray:
    return np.array([[62, 67], [98, 67], [80, 88], [68, 112], [92, 112]], np.float32)


def test_guardrail_accepts_identical_image() -> None:
    image = face_like()
    decision = evaluate_identity_guardrail(image, image.copy(), [image])
    assert decision.accepted
    assert decision.score_drop <= 1e-6


def test_guardrail_rejects_extreme_identity_regression() -> None:
    image = face_like()
    candidate = np.full_like(image, 255)
    decision = evaluate_identity_guardrail(image, candidate, [image], max_drop=0.05, absolute_minimum=0.8)
    assert not decision.accepted


def test_synthetic_validation_degradations_are_deterministic() -> None:
    image = face_like()
    first = synthetic_degradations(image)
    second = synthetic_degradations(image)
    assert set(first) == {"blur", "noise", "jpeg", "occlusion"}
    assert all(np.array_equal(first[key], second[key]) for key in first)
    metrics = validation_metrics(first["blur"], image)
    assert metrics.psnr > 0
    assert 0 <= metrics.identity_score <= 1


def test_region_masks_use_landmarks_and_bbox() -> None:
    masks = facial_region_masks((160, 160), landmarks(), (30, 20, 100, 130))
    assert set(masks) == {"left_eye", "right_eye", "nose", "mouth", "face"}
    assert all(mask.shape == (160, 160) for mask in masks.values())
    assert all(np.count_nonzero(mask) > 0 for mask in masks.values())


def test_regional_fusion_prefers_sharper_observed_reference() -> None:
    reference = face_like()
    primary = cv2.GaussianBlur(reference, (11, 11), 3.0)
    masks = [np.zeros((160, 160), np.uint8), np.zeros((160, 160), np.uint8)]
    fused, provenance, decisions = regional_reference_fusion(
        [primary, reference], masks, landmarks(), (30, 20, 100, 130), minimum_improvement=0.01
    )
    assert fused.shape == primary.shape
    assert provenance.shape == primary.shape[:2]
    assert any(decision.source_index == 1 for decision in decisions)
    assert np.count_nonzero(provenance == 1) > 0


def test_regional_fusion_does_not_use_occluded_reference_region() -> None:
    reference = face_like()
    primary = cv2.GaussianBlur(reference, (11, 11), 3.0)
    blocked = np.full((160, 160), 255, np.uint8)
    fused, provenance, decisions = regional_reference_fusion(
        [primary, reference], [np.zeros((160, 160), np.uint8), blocked], landmarks(), (30, 20, 100, 130), minimum_improvement=0.01
    )
    assert np.count_nonzero(provenance) == 0
    assert all(decision.source_index == 0 for decision in decisions)


def test_regional_fusion_never_changes_pixels_without_provenance() -> None:
    """Feathering must not leak into observed pixels outside selected facial regions."""
    reference = face_like()
    # Make the primary sufficiently blurred so at least one observed reference region wins.
    primary = cv2.GaussianBlur(reference, (15, 15), 4.0)
    masks = [np.zeros((160, 160), np.uint8), np.zeros((160, 160), np.uint8)]

    fused, provenance, decisions = regional_reference_fusion(
        [primary, reference], masks, landmarks(), (30, 20, 100, 130), minimum_improvement=0.005
    )

    assert any(decision.source_index == 1 for decision in decisions)
    changed = np.any(fused != primary, axis=2)
    # This is the strict provenance invariant: no pixel may be modified unless its
    # source is recorded. It catches Gaussian feather halos outside region masks.
    assert not np.any(changed & (provenance == 0))
