from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.pipeline import BlockKind, default_pipeline
from app.strict_execution import StrictBlockExecutor


def test_one_unalignable_reference_does_not_abort_mandatory_align_block() -> None:
    primary = np.full((96, 96, 3), 120, dtype=np.uint8)
    reference = np.zeros_like(primary)
    workspace = Workspace(primary=primary, references=[reference])
    workspace.metadata["primary_landmarks5"] = np.asarray(
        [[30, 34], [66, 34], [48, 49], [35, 68], [61, 68]], dtype=np.float32
    )
    executor = StrictBlockExecutor(workspace)
    block = next(item for item in default_pipeline() if item.kind is BlockKind.ALIGN)

    result = executor.execute(block)

    assert result.details["aligned"] == 0
    assert result.details["rejected_geometry"] == 1
    assert len(result.details["diagnostics"]) == 1
    assert result.details["diagnostics"][0]["rejected"] is True
