from __future__ import annotations

import cv2
import numpy as np

from app.reference_hint_runtime import expand_verified_single_reference_hint


def _face() -> np.ndarray:
    image = np.full((128, 128, 3), 24, dtype=np.uint8)
    cv2.ellipse(image, (64, 66), (42, 52), 0, 0, 360, (145, 170, 198), -1)
    cv2.circle(image, (50, 54), 4, (22, 22, 22), -1)
    cv2.circle(image, (78, 54), 4, (22, 22, 22), -1)
    return image


def test_verified_full_reference_expands_dark_sticker_core() -> None:
    clean = _face()
    primary = clean.copy()
    sticker = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.ellipse(sticker, (64, 64), (18, 12), 0, 0, 360, 255, -1)
    primary[sticker > 0] = (15, 15, 15)
    existing = np.zeros_like(sticker)
    cv2.circle(existing, (64, 64), 5, 255, -1)
    face = np.zeros_like(sticker)
    cv2.ellipse(face, (64, 66), (42, 52), 0, 0, 360, 255, -1)

    expanded, details = expand_verified_single_reference_hint(
        primary,
        clean,
        np.zeros_like(sticker),
        face,
        existing,
        strong_difference_threshold=0.08,
    )

    assert details["eligible"] is True
    assert details["added_pixels"] > 0
    assert np.count_nonzero(expanded & sticker) > np.count_nonzero(existing & sticker)
    assert np.count_nonzero(expanded & cv2.bitwise_not(face)) == 0


def test_hint_expansion_does_not_copy_damaged_reference_into_clean_primary() -> None:
    clean = _face()
    damaged_reference = clean.copy()
    sticker = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.ellipse(sticker, (64, 64), (18, 12), 0, 0, 360, 255, -1)
    damaged_reference[sticker > 0] = (15, 15, 15)
    false_positive_seed = np.zeros_like(sticker)
    cv2.circle(false_positive_seed, (64, 64), 5, 255, -1)
    face = np.zeros_like(sticker)
    cv2.ellipse(face, (64, 66), (42, 52), 0, 0, 360, 255, -1)

    expanded, details = expand_verified_single_reference_hint(
        clean,
        damaged_reference,
        np.zeros_like(sticker),
        face,
        false_positive_seed,
        strong_difference_threshold=0.08,
    )

    assert details["eligible"] is True
    assert details["reason"] == "no_directional_primary_damage_seed"
    assert np.count_nonzero(expanded) == 0


def test_hint_expansion_abstains_for_partial_reference() -> None:
    clean = _face()
    primary = clean.copy()
    reference = clean.copy()
    reference[:, :70] = 0
    blocked = np.zeros(clean.shape[:2], dtype=np.uint8)
    blocked[:, :70] = 255
    face = np.zeros_like(blocked)
    cv2.ellipse(face, (64, 66), (42, 52), 0, 0, 360, 255, -1)
    existing = np.zeros_like(blocked)

    expanded, details = expand_verified_single_reference_hint(
        primary,
        reference,
        blocked,
        face,
        existing,
        minimum_face_coverage=0.70,
    )

    assert details["eligible"] is False
    assert details["reason"] == "insufficient_reference_coverage"
    assert np.array_equal(expanded, existing)


def test_hint_expansion_abstains_when_directional_proposal_is_too_large() -> None:
    clean = _face()
    primary = np.full_like(clean, 250)
    face = np.full(clean.shape[:2], 255, dtype=np.uint8)
    existing = np.zeros_like(face)
    cv2.circle(existing, (64, 64), 5, 255, -1)

    expanded, details = expand_verified_single_reference_hint(
        primary,
        clean,
        np.zeros_like(face),
        face,
        existing,
        strong_difference_threshold=0.01,
        maximum_face_fraction=0.10,
    )

    assert details["eligible"] is True
    assert details["reason"] == "proposal_too_large"
    assert np.count_nonzero(expanded) <= np.count_nonzero(existing)
