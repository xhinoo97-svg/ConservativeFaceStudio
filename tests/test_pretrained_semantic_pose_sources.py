from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.pretrained_semantic_handlers import _aligned_original_pose_inputs, _bbox_from_landmarks


def test_pose_inputs_map_aligned_slots_back_to_original_reference_photos() -> None:
    refs = [
        np.full((80, 90, 3), 11, dtype=np.uint8),
        np.full((80, 90, 3), 22, dtype=np.uint8),
        np.full((80, 90, 3), 33, dtype=np.uint8),
    ]
    workspace = SimpleNamespace(
        references=refs,
        metadata={
            "aligned_reference_source_indices": [2, 0],
            "reference_bboxes": [(10, 12, 30, 40), (11, 13, 31, 41), (12, 14, 32, 42)],
            "reference_landmarks5": [None, None, None],
        },
    )

    items = _aligned_original_pose_inputs(workspace, 2)

    assert len(items) == 2
    assert items[0] is not None and items[1] is not None
    image0, bbox0, source0 = items[0]
    image1, bbox1, source1 = items[1]
    assert source0 == 2 and source1 == 0
    assert bbox0 == (12, 14, 32, 42)
    assert bbox1 == (10, 12, 30, 40)
    assert int(image0[0, 0, 0]) == 33
    assert int(image1[0, 0, 0]) == 11


def test_pose_bbox_falls_back_to_original_landmarks_without_using_aligned_geometry() -> None:
    refs = [np.zeros((120, 140, 3), dtype=np.uint8)]
    landmarks = np.array(
        [[45.0, 42.0], [88.0, 43.0], [67.0, 62.0], [50.0, 83.0], [84.0, 84.0]],
        dtype=np.float32,
    )
    workspace = SimpleNamespace(
        references=refs,
        metadata={
            "aligned_reference_source_indices": [0],
            "reference_bboxes": [None],
            "reference_landmarks5": [landmarks],
        },
    )

    items = _aligned_original_pose_inputs(workspace, 1)

    assert items[0] is not None
    image, bbox, source = items[0]
    assert image is refs[0]
    assert source == 0
    x, y, w, h = bbox
    assert 0 <= x < 140 and 0 <= y < 120
    assert w >= 8 and h >= 8
    assert x <= int(np.min(landmarks[:, 0]))
    assert y <= int(np.min(landmarks[:, 1]))


def test_bbox_from_landmarks_rejects_non_finite_points() -> None:
    bad = np.zeros((5, 2), dtype=np.float32)
    bad[2, 0] = np.nan
    assert _bbox_from_landmarks(bad, (100, 100)) is None
