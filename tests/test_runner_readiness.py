from __future__ import annotations

import numpy as np

from app.automatic import AutomaticPipelineRunner
from app.execution import Workspace
from app.pipeline import BlockKind


def test_automatic_runner_has_real_handler_for_every_block_kind() -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    runner = AutomaticPipelineRunner(Workspace(image, references=[]))
    missing = [kind.value for kind in BlockKind if kind not in runner.executor._handlers]
    assert missing == []


def test_case_aware_runtime_is_installed_after_pretrained_handlers() -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    runner = AutomaticPipelineRunner(Workspace(image, references=[]))
    assert runner._skip_reason(BlockKind.INPAINT) is None
    assert runner.executor.workspace.metadata.get("restoration_case") is not None
