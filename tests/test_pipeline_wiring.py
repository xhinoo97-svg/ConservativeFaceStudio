from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.pipeline import BlockKind, default_pipeline
from app.strict_execution import StrictBlockExecutor


def test_strict_executor_wires_every_default_pipeline_block() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    executor = StrictBlockExecutor(Workspace(primary=image))

    declared = {block.kind for block in default_pipeline()}
    wired = set(executor._handlers)

    assert declared == wired
    assert len(declared) == 13
    assert BlockKind.INPAINT in wired
    assert BlockKind.FRONTALIZE in wired


def test_pipeline_keeps_residual_generation_inside_mandatory_inpaint_block() -> None:
    blocks = {block.kind: block for block in default_pipeline()}

    assert blocks[BlockKind.INPAINT].optional is False
    assert blocks[BlockKind.FRONTALIZE].optional is True
    assert blocks[BlockKind.IDENTITY_CHECK].depends_on == ("fusion",)
    assert blocks[BlockKind.EXPORT].depends_on == ("identity_check",)
