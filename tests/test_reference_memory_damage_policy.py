from __future__ import annotations

import cv2
import numpy as np

from app.reference_memory import specific_reference_memory_fusion


def _face(size: int = 160) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image = np.full((size, size, 3), 28, dtype=np.uint8)
    bbox = (32, 22, 96, 120)
    cv2.ellipse(image, (80, 82), (46, 58), 0, 0, 360, (142, 168, 194), -1)
    cv2.circle(image, (61, 66), 6, (24, 28, 34), -1)
    cv2.circle(image, (99, 66), 6, (24, 28, 34), -1)
    cv2.line(image, (80, 70), (77, 91), (80, 95, 112), 3)
    cv2.line(image, (65, 107), (95, 107), (52, 55, 80), 3)
    landmarks = np.asarray([[61, 66], [99, 66], [80, 84], [68, 107], [92, 107]], dtype=np.float32)
    return image, landmarks, bbox


def _damage_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(mask, (80, 82), (25, 19), 0, 0, 360, 255, -1)
    return mask


def test_clean_donor_repairs_damage_without_touching_visible_primary() -> None:
    clean, landmarks, bbox = _face()
    damage = _damage_mask(clean.shape[:2])
    primary = clean.copy()
    primary[damage > 0] = (8, 8, 8)
    zero = np.zeros(clean.shape[:2], dtype=np.uint8)
    full_support = np.full(clean.shape[:2], 255, dtype=np.uint8)

    result = specific_reference_memory_fusion(
        [primary, clean.copy()],
        [damage, zero],
        landmarks,
        bbox,
        reference_support_masks=[full_support],
        minimum_region_confidence=0.35,
        minimum_quality_gain=10.0,
    )

    active = damage > 0
    repaired = np.all(result.image == clean, axis=2) & active
    assert np.count_nonzero(repaired) / np.count_nonzero(active) > 0.90
    outside = ~active
    assert np.array_equal(result.image[outside], primary[outside])
    assert not np.any((result.provenance_map > 0) & outside)


def test_complementary_donors_union_covers_damaged_region() -> None:
    clean, landmarks, bbox = _face()
    damage = _damage_mask(clean.shape[:2])
    primary = clean.copy()
    primary[damage > 0] = (8, 8, 8)
    zero = np.zeros(clean.shape[:2], dtype=np.uint8)
    left = np.zeros(clean.shape[:2], dtype=np.uint8)
    right = np.zeros(clean.shape[:2], dtype=np.uint8)
    left[:, :81] = 255
    right[:, 80:] = 255

    result = specific_reference_memory_fusion(
        [primary, clean.copy(), clean.copy()],
        [damage, zero, zero],
        landmarks,
        bbox,
        reference_support_masks=[left, right],
        top_k=2,
        minimum_region_confidence=0.30,
        minimum_quality_gain=10.0,
    )

    active = damage > 0
    donor_used = (result.provenance_map > 0) & active
    assert np.count_nonzero(donor_used) / np.count_nonzero(active) > 0.90
    assert np.all(result.image[donor_used] == clean[donor_used])
    assert np.any(result.provenance_map == 1)
    assert np.any(result.provenance_map == 2)
    assert not np.any((result.provenance_map > 0) & ~active)
