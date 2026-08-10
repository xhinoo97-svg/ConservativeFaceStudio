from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.partial_reference_runtime import _merge_frozen_primary_hint


def test_frozen_preflight_occlusion_does_not_authorize_repair_without_donor_evidence() -> None:
    shape = (64, 64)
    primary = np.full((64, 64, 3), 140, dtype=np.uint8)
    frozen = np.zeros(shape, dtype=np.uint8)
    frozen[20:36, 18:46] = 255
    current = np.zeros(shape, dtype=np.uint8)
    current[24:32, 24:40] = 255
    workspace = SimpleNamespace(
        primary=primary,
        aligned_references=[],
        metadata={
            "preflight_original_occlusion_masks": [frozen.copy()],
            "reference_consensus_occlusion": current.copy(),
        },
    )

    added = _merge_frozen_primary_hint(workspace)
    merged = workspace.metadata["reference_consensus_occlusion"]

    assert added == 0
    assert np.array_equal(merged, current)
    diagnostics = workspace.metadata["reference_guided_seed_diagnostics"]
    assert diagnostics["trusted_donors"] == 0


def test_frozen_hint_merge_is_noop_without_preflight_evidence() -> None:
    primary = np.full((32, 32, 3), 140, dtype=np.uint8)
    workspace = SimpleNamespace(primary=primary, aligned_references=[], metadata={})
    assert _merge_frozen_primary_hint(workspace) == 0
    assert "reference_consensus_occlusion" not in workspace.metadata
