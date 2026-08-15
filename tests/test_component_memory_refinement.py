from __future__ import annotations

import cv2
import numpy as np

from app.reference_memory import specific_reference_memory_fusion


def _ground_truth_face() -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image = np.full((160, 160, 3), 145, dtype=np.uint8)
    cv2.ellipse(image, (80, 82), (48, 62), 0, 0, 360, (176, 154, 140), -1)
    cv2.circle(image, (60, 64), 6, (25, 25, 25), -1)
    cv2.circle(image, (100, 64), 6, (25, 25, 25), -1)
    cv2.line(image, (80, 70), (80, 92), (95, 82, 78), 3)
    cv2.ellipse(image, (80, 112), (17, 6), 0, 0, 180, (70, 60, 65), 3)
    # Add deterministic local texture so phase correlation has enough evidence.
    for x in range(45, 116, 7):
        cv2.circle(image, (x, 88 + ((x // 7) % 3) * 4), 1, (135, 115, 105), -1)
    landmarks = np.asarray(
        [[60.0, 64.0], [100.0, 64.0], [80.0, 84.0], [65.0, 111.0], [95.0, 111.0]],
        dtype=np.float32,
    )
    return image, landmarks, (32, 20, 96, 124)


def test_region_memory_corrects_small_component_shift_before_transfer() -> None:
    truth, landmarks, bbox = _ground_truth_face()
    primary = cv2.GaussianBlur(truth, (0, 0), 2.2)
    matrix = np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float32)
    donor = cv2.warpAffine(truth, matrix, (160, 160), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT_101)
    masks = [np.zeros((160, 160), dtype=np.uint8), np.zeros((160, 160), dtype=np.uint8)]
    support = [np.full((160, 160), 255, dtype=np.uint8)]

    result = specific_reference_memory_fusion(
        [primary, donor],
        masks,
        landmarks,
        bbox,
        reference_support_masks=support,
        top_k=1,
        minimum_region_confidence=0.50,
        minimum_quality_gain=0.0005,
        maximum_replace_fraction=0.45,
        local_refinement_max_shift=4.0,
        local_refinement_min_response=0.04,
    )

    before = float(np.mean((primary.astype(np.float32) - truth.astype(np.float32)) ** 2))
    after = float(np.mean((result.image.astype(np.float32) - truth.astype(np.float32)) ** 2))
    assert result.transferred_pixels > 0
    assert after < before
