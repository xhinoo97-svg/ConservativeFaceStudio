from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.pipeline import BlockKind, default_pipeline
from app.strict_execution import StrictBlockExecutor


def test_strict_region_select_abstains_without_landmarks() -> None:
    primary = np.full((64, 64, 3), 100, dtype=np.uint8)
    reference = np.full((64, 64, 3), 120, dtype=np.uint8)
    zero = np.zeros((64, 64), dtype=np.uint8)
    workspace = Workspace(
        primary=primary.copy(),
        references=[reference.copy()],
        aligned_references=[reference.copy()],
        occlusion_masks=[zero.copy(), zero.copy()],
    )
    block = next(item for item in default_pipeline() if item.kind is BlockKind.REGION_SELECT)
    result = StrictBlockExecutor(workspace).execute(block)
    assert result.details["engine"] == "specific-memory-abstain"
    assert result.details["transferred_pixels"] == 0
    assert np.array_equal(result.image, primary)
