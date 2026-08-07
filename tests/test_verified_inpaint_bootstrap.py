from __future__ import annotations

import numpy as np

from app.automatic import AutomaticPipelineRunner
from app.execution import Workspace
from app.pipeline import BlockKind


def test_verified_reference_inpaint_is_installed_without_model_pack() -> None:
    primary = np.zeros((64, 64, 3), dtype=np.uint8)
    reference = primary.copy()
    workspace = Workspace(primary=primary, references=[reference])

    runner = AutomaticPipelineRunner(workspace)
    handler = runner.executor._handlers[BlockKind.INPAINT]

    assert handler.__module__ == "app.pretrained_inpaint_handler"
    assert "core_model_paths" not in workspace.metadata
