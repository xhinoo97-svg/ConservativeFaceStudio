from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.automatic import AutomaticPipelineRunner
from app.pipeline import BlockKind


def _runner_with_workspace(workspace):
    runner = AutomaticPipelineRunner.__new__(AutomaticPipelineRunner)
    runner.executor = SimpleNamespace(workspace=workspace)
    return runner


def test_frontalize_is_skipped_when_damage_target_exists():
    target = np.zeros((24, 24), dtype=np.uint8)
    target[8:16, 8:16] = 255
    workspace = SimpleNamespace(
        references=[np.zeros((24, 24, 3), dtype=np.uint8)],
        occlusion_masks=[],
        metadata={"inpaint_target_mask": target},
    )
    runner = _runner_with_workspace(workspace)

    reason = runner._skip_reason(BlockKind.FRONTALIZE)

    assert reason is not None
    assert "geometria originale" in reason


def test_frontalize_is_skipped_when_occlusion_mask_exists_before_inpaint_target():
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:10, 5:11] = 255
    workspace = SimpleNamespace(
        references=[np.zeros((24, 24, 3), dtype=np.uint8)],
        occlusion_masks=[mask],
        metadata={},
    )
    runner = _runner_with_workspace(workspace)

    assert runner._skip_reason(BlockKind.FRONTALIZE) is not None


def test_frontalize_remains_available_without_any_damage_target():
    workspace = SimpleNamespace(
        references=[],
        occlusion_masks=[],
        metadata={},
    )
    runner = _runner_with_workspace(workspace)

    assert runner._skip_reason(BlockKind.FRONTALIZE) is None
