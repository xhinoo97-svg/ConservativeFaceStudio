from __future__ import annotations

import cv2
import numpy as np

from app.execution import Workspace
from app.pipeline import BlockKind, default_pipeline
from app.reference_memory import specific_reference_memory_fusion
from app.strict_execution import StrictBlockExecutor


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


def _zeros(shape: tuple[int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def _region_block():
    return next(block for block in default_pipeline() if block.kind is BlockKind.REGION_SELECT)


def test_specific_memory_uses_observed_pixels_and_keeps_provenance_exact() -> None:
    clean, landmarks, bbox = _face()
    primary = cv2.GaussianBlur(clean, (9, 9), 2.2)
    ref_a = clean.copy()
    ref_b = clean.copy()
    masks = [_zeros(clean.shape[:2]) for _ in range(3)]

    result = specific_reference_memory_fusion(
        [primary, ref_a, ref_b],
        masks,
        landmarks,
        bbox,
        minimum_region_confidence=0.55,
        minimum_quality_gain=0.001,
        maximum_replace_fraction=0.30,
    )

    assert result.transferred_pixels > 0
    assert np.max(result.provenance_map) <= 2
    used = result.provenance_map > 0
    assert np.any(used)
    for source_index in (1, 2):
        source_pixels = result.provenance_map == source_index
        if np.any(source_pixels):
            assert np.array_equal(result.image[source_pixels], [ref_a, ref_b][source_index - 1][source_pixels])
    assert np.count_nonzero(result.confidence_map) == np.count_nonzero(used)


def test_specific_memory_fuses_non_overlapping_complementary_support() -> None:
    clean, landmarks, bbox = _face()
    primary = cv2.GaussianBlur(clean, (11, 11), 2.8)
    ref_a = clean.copy()
    ref_b = clean.copy()
    masks = [_zeros(clean.shape[:2]) for _ in range(3)]

    support_a = _zeros(clean.shape[:2])
    support_b = _zeros(clean.shape[:2])
    support_a[52:82, 45:76] = 255
    support_b[94:120, 64:98] = 255
    assert not np.any((support_a > 0) & (support_b > 0))

    result = specific_reference_memory_fusion(
        [primary, ref_a, ref_b],
        masks,
        landmarks,
        bbox,
        reference_support_masks=[support_a, support_b],
        minimum_region_confidence=0.45,
        minimum_quality_gain=0.001,
        maximum_replace_fraction=0.50,
    )

    used = result.provenance_map > 0
    allowed = (support_a > 0) | (support_b > 0)
    assert np.any(result.provenance_map == 1)
    assert np.any(result.provenance_map == 2)
    assert not np.any(used & ~allowed)
    assert np.all((result.provenance_map == 1) <= (support_a > 0))
    assert np.all((result.provenance_map == 2) <= (support_b > 0))


def test_specific_memory_abstains_when_references_disagree() -> None:
    clean, landmarks, bbox = _face()
    primary = clean.copy()
    ref_a = clean.copy()
    ref_b = clean.copy()
    ref_a[52:82, 48:73] = (10, 10, 240)
    ref_b[52:82, 48:73] = (240, 10, 10)
    masks = [_zeros(clean.shape[:2]) for _ in range(3)]

    result = specific_reference_memory_fusion(
        [primary, ref_a, ref_b],
        masks,
        landmarks,
        bbox,
        minimum_region_confidence=0.70,
        minimum_quality_gain=0.01,
        agreement_colour_threshold=3.0,
    )

    changed = np.any(result.image != primary, axis=2)
    disputed = np.zeros(primary.shape[:2], dtype=bool)
    disputed[52:82, 48:73] = True
    assert not np.any(changed & disputed)


def test_strict_executor_routes_region_select_through_specific_memory() -> None:
    clean, landmarks, bbox = _face()
    primary = cv2.GaussianBlur(clean, (9, 9), 2.2)
    masks = [_zeros(clean.shape[:2]) for _ in range(3)]
    workspace = Workspace(
        primary=primary,
        references=[clean.copy(), clean.copy()],
        aligned_references=[clean.copy(), clean.copy()],
        occlusion_masks=masks,
        metadata={"primary_landmarks5": landmarks, "primary_bbox": bbox},
    )
    executor = StrictBlockExecutor(workspace)
    result = executor.execute(
        _region_block(),
        minimum_region_confidence=0.55,
        minimum_quality_gain=0.001,
    )

    assert result.details["engine"] == "dmd-inspired-specific-memory"
    assert result.details["generic_dictionary_used"] is False
    assert result.details["reference_count"] == 2
    assert result.details["transferred_pixels"] > 0
    assert executor.workspace.provenance_map is not None
    assert "specific_reference_memory" in executor.workspace.metadata
