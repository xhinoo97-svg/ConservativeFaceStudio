from __future__ import annotations

import numpy as np

from app.cross_reference_preclean import preclean_aligned_references
from app.execution import Workspace


def _workspace(*, conflict: bool = False) -> Workspace:
    shape = (40, 40)
    primary = np.full((*shape, 3), 120, dtype=np.uint8)
    left = np.zeros((*shape, 3), dtype=np.uint8)
    right = np.zeros((*shape, 3), dtype=np.uint8)

    left[10:30, 8:22] = (90, 110, 130)
    right[10:30, 18:32] = (90, 110, 130)
    if conflict:
        right[10:30, 18:22] = (180, 195, 210)

    support_left = np.zeros(shape, dtype=np.uint8)
    support_right = np.zeros(shape, dtype=np.uint8)
    support_left[10:30, 8:22] = 255
    support_right[10:30, 18:32] = 255

    # Simulate the current generic proposal failure mode: every useful pixel is marked
    # suspect even though these coordinate-preserving partial references are clean.
    ref_damage_left = support_left.copy()
    ref_damage_right = support_right.copy()

    workspace = Workspace(primary=primary, references=[left.copy(), right.copy()])
    workspace.aligned_references = [left.copy(), right.copy()]
    workspace.occlusion_masks = [np.zeros(shape, dtype=np.uint8), ref_damage_left, ref_damage_right]
    workspace.metadata["aligned_reference_support_masks"] = [support_left, support_right]
    workspace.metadata["aligned_reference_source_indices"] = [0, 1]
    workspace.metadata["aligned_reference_original_source_indices"] = [1, 2]
    workspace.metadata["same_canvas_partial_alignment_diagnostics"] = [
        {
            "runtime_reference_index": 0,
            "method": "verified-same-canvas-partial",
            "global_transform_required": False,
            "local_identity_transform": True,
        },
        {
            "runtime_reference_index": 1,
            "method": "verified-same-canvas-partial",
            "global_transform_required": False,
            "local_identity_transform": True,
        },
    ]
    return workspace


def test_agreeing_coordinate_partials_preserve_unconfirmed_observed_evidence() -> None:
    workspace = _workspace(conflict=False)

    cleaned, evidence, stats = preclean_aligned_references(workspace)

    assert len(cleaned) == 2
    assert workspace.metadata["coordinate_reference_consensus_trusted_slots"] == [0, 1]
    assert int(workspace.metadata["coordinate_reference_evidence_recovered_pixels"]) > 0

    # Non-overlap pixels have no donor that could repair them. They must remain attributed
    # to the original clean coordinate-preserving reference rather than becoming zero.
    assert np.all(np.asarray(evidence[0])[12:28, 9:17] == 1)
    assert np.all(np.asarray(evidence[1])[12:28, 23:31] == 2)
    assert stats[0].unresolved_pixels == 0
    assert stats[1].unresolved_pixels == 0


def test_conflicting_coordinate_partials_do_not_promote_heuristic_suspect_pixels() -> None:
    workspace = _workspace(conflict=True)

    _, evidence, stats = preclean_aligned_references(workspace)

    assert workspace.metadata["coordinate_reference_consensus_trusted_slots"] == []
    assert workspace.metadata["coordinate_reference_evidence_recovered_pixels"] == 0
    assert np.all(np.asarray(evidence[0])[12:28, 9:17] == 0)
    assert np.all(np.asarray(evidence[1])[12:28, 23:31] == 0)
    assert stats[0].unresolved_pixels > 0
    assert stats[1].unresolved_pixels > 0
