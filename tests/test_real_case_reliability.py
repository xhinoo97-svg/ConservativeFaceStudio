from __future__ import annotations

import cv2
import numpy as np

from app.execution import Workspace
from app.partial_reference_runtime import _effective_masks
from app.restoration import detect_occlusion_candidates, detail_reliability_map


def _textured_patch(size: int = 192) -> np.ndarray:
    y, x = np.indices((size, size))
    base = 128.0 + 24.0 * np.sin(x / 3.7) + 18.0 * np.cos(y / 5.1)
    texture = 7.0 * np.sin((x + y) / 1.9)
    gray = np.clip(base + texture, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_strong_blur_cannot_become_high_confidence_identity_donor() -> None:
    observed = _textured_patch()
    blurred = cv2.GaussianBlur(observed, (0, 0), 9.0)
    sharp_map = detail_reliability_map(observed)
    blur_map = detail_reliability_map(blurred)
    roi = np.s_[24:-24, 24:-24]
    assert float(np.mean(sharp_map[roi])) > float(np.mean(blur_map[roi])) + 35.0
    assert float(np.mean(blur_map[roi])) < 80.0


def test_marker_and_colour_mosaic_produce_occlusion_proposals() -> None:
    image = np.full((180, 220, 3), (145, 165, 185), dtype=np.uint8)
    cv2.rectangle(image, (25, 55), (195, 92), (4, 4, 4), -1)

    # Piecewise-flat saturated polygons model geometric/pixelated obscuration without
    # embedding any personal photograph in the test suite.
    polygon_colours = [(35, 55, 175), (190, 115, 45), (50, 165, 190), (165, 55, 145)]
    polygons = [
        np.array([[48, 112], [82, 98], [102, 132], [68, 148]], np.int32),
        np.array([[84, 100], [118, 104], [108, 142], [100, 132]], np.int32),
        np.array([[120, 104], [154, 98], [172, 132], [140, 150]], np.int32),
        np.array([[108, 142], [140, 150], [124, 170], [92, 166]], np.int32),
    ]
    for colour, polygon in zip(polygon_colours, polygons):
        cv2.fillConvexPoly(image, polygon, colour)

    proposal = detect_occlusion_candidates(image)
    marker_fraction = float(np.mean(proposal[58:90, 30:190] > 0))
    mosaic_fraction = float(np.mean(proposal[100:170, 45:175] > 0))
    assert marker_fraction > 0.80
    assert mosaic_fraction > 0.08


def test_occlusion_is_never_counted_as_reliable_detail() -> None:
    image = _textured_patch(160)
    mask = np.zeros((160, 160), dtype=np.uint8)
    mask[45:110, 50:120] = 255
    reliability = detail_reliability_map(image, mask)
    assert np.count_nonzero(reliability[45:110, 50:120]) == 0
    assert float(np.mean(reliability[:35, :35])) > 0.0


def test_runtime_prefers_frozen_pre_deblur_reliability_over_sharpened_reference() -> None:
    primary = _textured_patch(128)
    reference = primary.copy()
    support = np.full((128, 128), 255, dtype=np.uint8)

    workspace = Workspace(primary=primary, references=[])
    workspace.aligned_references = [reference]
    workspace.occlusion_masks = [
        np.zeros((128, 128), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.uint8),
    ]
    workspace.metadata["aligned_reference_support_masks"] = [support]
    # Simulate an original source that was strongly blurred before a learned deblur
    # made the aligned RGB image look sharp. Evidence must remain low confidence.
    workspace.metadata["aligned_reference_detail_reliability_maps"] = [
        np.zeros((128, 128), dtype=np.uint8)
    ]
    workspace.metadata["detail_reliability_threshold"] = 40

    effective, _, low_detail = _effective_masks(workspace)
    assert effective is not None
    assert low_detail == 128 * 128
    assert np.all(effective[1] == 255)
    assert workspace.metadata["detail_reliability_source"] == "pre-deblur-aligned"
