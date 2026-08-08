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

    # The final runner is intentionally wrapped by the case-aware runtime. The
    # wrapper must remain installed even without a model pack so it can route
    # single-image and multi-reference cases while delegating reference repair to
    # the verified pretrained handler underneath.
    assert handler.__module__ == "app.case_aware_runtime"
    assert "core_model_paths" not in workspace.metadata
