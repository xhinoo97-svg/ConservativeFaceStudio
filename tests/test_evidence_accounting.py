from __future__ import annotations

import numpy as np
import pytest

from app.evidence_accounting import reconcile_evidence_accounting
from app.execution import Workspace


def test_damaged_unrepaired_pixels_are_forced_to_remain_unresolved() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    target = np.zeros((12, 16), dtype=np.uint8)
    target[3:9, 4:12] = 255
    workspace = Workspace(primary=image, metadata={"inpaint_target_mask": target})
    report = reconcile_evidence_accounting(workspace)
    assert report["target_pixels"] == 48
    assert report["unresolved_pixels"] == 48
    assert np.array_equal(workspace.metadata["inpaint_unresolved_mask"], target)


def test_mismatched_final_mask_is_a_hard_accounting_failure() -> None:
    workspace = Workspace(
        primary=np.zeros((12, 16, 3), dtype=np.uint8),
        metadata={"inpaint_target_mask": np.zeros((6, 8), dtype=np.uint8)},
    )
    with pytest.raises(AssertionError, match="does not match final MAIN"):
        reconcile_evidence_accounting(workspace)
