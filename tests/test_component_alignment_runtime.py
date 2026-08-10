from __future__ import annotations

import cv2
import numpy as np

from app.component_alignment import refine_component_translation


def test_component_micro_refinement_corrects_small_shift_only() -> None:
    primary = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(primary, (48, 48), 12, (220, 220, 220), -1)
    cv2.line(primary, (40, 48), (56, 48), (30, 30, 30), 2)
    cv2.line(primary, (48, 40), (48, 56), (30, 30, 30), 2)

    matrix = np.asarray([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]], dtype=np.float32)
    shifted = cv2.warpAffine(primary, matrix, (96, 96), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    support = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(support, (51, 46), 18, 255, -1)
    component = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(component, (48, 48), 20, 255, -1)

    result = refine_component_translation(
        shifted,
        primary,
        support,
        component,
        maximum_shift=5.0,
        minimum_response=0.02,
    )

    assert result.accepted
    assert abs(result.dx) <= 5.0
    assert abs(result.dy) <= 5.0
    before = float(np.mean(np.abs(shifted.astype(np.float32) - primary.astype(np.float32))))
    after = float(np.mean(np.abs(result.image.astype(np.float32) - primary.astype(np.float32))))
    assert after < before


def test_strict_component_refinement_uses_integer_nearest_transfer() -> None:
    primary = np.zeros((96, 96, 3), dtype=np.uint8)
    # Deliberately discrete colours make interpolation-created values easy to detect.
    primary[34:62, 36:44] = (17, 73, 191)
    primary[42:50, 44:68] = (211, 31, 89)
    shifted = np.zeros_like(primary)
    shifted[36:64, 39:47] = (17, 73, 191)
    shifted[44:52, 47:71] = (211, 31, 89)

    support = np.zeros((96, 96), dtype=np.uint8)
    support[30:70, 30:78] = 255
    component = support.copy()

    result = refine_component_translation(
        shifted,
        primary,
        support,
        component,
        maximum_shift=5.0,
        minimum_response=0.01,
        minimum_similarity_gain=0.001,
        preserve_observed_pixels=True,
    )

    assert result.accepted
    assert float(result.dx).is_integer()
    assert float(result.dy).is_integer()
    source_colours = {tuple(value) for value in np.unique(shifted.reshape(-1, 3), axis=0)}
    output_colours = {tuple(value) for value in np.unique(result.image.reshape(-1, 3), axis=0)}
    assert output_colours.issubset(source_colours)


def test_component_micro_refinement_abstains_on_large_shift() -> None:
    primary = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.circle(primary, (48, 48), 10, (220, 220, 220), -1)
    shifted = np.roll(primary, 14, axis=1)
    support = np.full((96, 96), 255, dtype=np.uint8)
    component = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(component, (48, 48), 20, 255, -1)

    result = refine_component_translation(
        shifted,
        primary,
        support,
        component,
        maximum_shift=5.0,
        minimum_response=0.02,
    )
    assert not result.accepted


def test_component_micro_refinement_abstains_on_blur_induced_subpixel_jitter() -> None:
    """Same anatomy at the same coordinates must not be resampled just because it is blurred."""
    sharp = np.full((128, 128, 3), 28, dtype=np.uint8)
    cv2.ellipse(sharp, (64, 64), (34, 44), 0, 0, 360, (142, 168, 194), -1)
    cv2.circle(sharp, (50, 55), 5, (24, 28, 34), -1)
    cv2.circle(sharp, (78, 55), 5, (24, 28, 34), -1)
    cv2.line(sharp, (64, 58), (62, 77), (80, 95, 112), 3)
    blurred = cv2.GaussianBlur(sharp, (9, 9), 2.2)
    support = np.full((128, 128), 255, dtype=np.uint8)
    component = np.zeros((128, 128), dtype=np.uint8)
    cv2.ellipse(component, (64, 64), (34, 44), 0, 0, 360, 255, -1)

    result = refine_component_translation(
        sharp,
        blurred,
        support,
        component,
        maximum_shift=5.0,
        minimum_response=0.02,
    )

    assert not result.accepted
    assert result.dx == 0.0
    assert result.dy == 0.0
    assert np.array_equal(result.image, sharp)
