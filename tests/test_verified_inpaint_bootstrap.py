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

    # The final production handler is the adaptive cascade. Case-aware/reference repair
    # remains underneath it through the final autorun installer binding. This must also
    # be true without a downloaded model pack so observed reference transfer still works
    # and generation remains an optional severe-stage fallback only.
    assert handler.__module__ == "app.adaptive_restoration_cascade"
    assert getattr(handler, "_adaptive_restoration_cascade", False) is True
    assert "core_model_paths" not in workspace.metadata
