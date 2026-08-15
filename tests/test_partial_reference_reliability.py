from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from app.partial_reference_runtime import _effective_masks


def test_blurred_reference_is_gated_separately_from_occlusion() -> None:
    sharp = np.zeros((96, 96, 3), dtype=np.uint8)
    for x in range(12, 84, 8):
        cv2.line(sharp, (x, 12), (x, 84), (220, 220, 220), 2)
    blurred = cv2.GaussianBlur(sharp, (0, 0), 9.0)
    zero = np.zeros((96, 96), dtype=np.uint8)
    support = np.full((96, 96), 255, dtype=np.uint8)
    workspace = SimpleNamespace(
        primary=sharp,
        aligned_references=[blurred],
        occlusion_masks=[zero.copy(), zero.copy()],
        metadata={
            "aligned_reference_support_masks": [support],
            "detail_reliability_threshold": 40,
        },
    )
    masks, support_blocked, low_detail_blocked = _effective_masks(workspace)
    assert masks is not None
    assert support_blocked == 0
    assert low_detail_blocked > 0
    assert int(np.count_nonzero(masks[1])) == low_detail_blocked
    assert "aligned_reference_detail_reliability_maps" in workspace.metadata
