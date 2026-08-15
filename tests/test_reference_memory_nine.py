from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.reference_memory import specific_reference_memory_fusion


def _face(size: int = 160) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image = np.full((size, size, 3), 28, dtype=np.uint8)
    bbox = (32, 22, 96, 120)
    cv2.ellipse(image, (80, 82), (46, 58), 0, 0, 360, (142, 168, 194), -1)
    cv2.circle(image, (61, 66), 6, (24, 28, 34), -1)
    cv2.circle(image, (99, 66), 6, (24, 28, 34), -1)
    cv2.line(image, (80, 70), (77, 91), (80, 95, 112), 3)
    cv2.line(image, (65, 107), (95, 107), (52, 55, 80), 3)
    landmarks = np.array(
        [[61, 66], [99, 66], [80, 84], [68, 107], [92, 107]],
        dtype=np.float32,
    )
    return image, landmarks, bbox


def test_reference_memory_can_use_ninth_reference_for_missing_region() -> None:
    clean, landmarks, bbox = _face()
    damage = np.zeros(clean.shape[:2], dtype=np.uint8)
    damage[94:118, 62:100] = 255
    primary = clean.copy()
    primary[damage > 0] = 0

    refs: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    masks: list[np.ndarray] = [damage.copy()]
    for _ in range(8):
        refs.append(clean.copy())
        supports.append(np.zeros(clean.shape[:2], dtype=np.uint8))
        masks.append(np.zeros(clean.shape[:2], dtype=np.uint8))

    refs.append(clean.copy())
    supports.append(damage.copy())
    masks.append(np.zeros(clean.shape[:2], dtype=np.uint8))

    result = specific_reference_memory_fusion(
        [primary, *refs],
        masks,
        landmarks,
        bbox,
        reference_support_masks=supports,
        minimum_region_confidence=0.40,
        minimum_quality_gain=0.0,
    )

    repaired = (result.provenance_map == 9) & (damage > 0)
    assert np.any(repaired)
    assert np.array_equal(result.image[repaired], clean[repaired])
    assert np.max(result.provenance_map) == 9


def test_reference_memory_rejects_more_than_nine_references() -> None:
    clean, landmarks, bbox = _face()
    images = [clean.copy() for _ in range(11)]
    masks = [np.zeros(clean.shape[:2], dtype=np.uint8) for _ in images]

    with pytest.raises(ValueError, match="massimo|limite"):
        specific_reference_memory_fusion(images, masks, landmarks, bbox)
