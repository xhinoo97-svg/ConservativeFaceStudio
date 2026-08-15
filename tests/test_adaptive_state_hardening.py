from __future__ import annotations

import numpy as np

from app.adaptive_restoration_cascade import build_severity_map
from app.automatic import AutomaticPipelineRunner
from app.execution import Workspace


def test_block7_observed_reference_provenance_is_preprotected_before_block8() -> None:
    image = np.full((64, 64, 3), 120, dtype=np.uint8)
    workspace = Workspace(image.copy())
    workspace.metadata["primary_bbox"] = (8, 8, 48, 48)

    damage = np.zeros((64, 64), dtype=np.uint8)
    damage[18:30, 18:30] = 255
    workspace.metadata["preflight_original_occlusion_masks"] = [damage]
    workspace.metadata["preflight_detail_reliability_maps"] = [np.full((64, 64), 255, dtype=np.uint8)]

    provenance = np.zeros((64, 64), dtype=np.uint16)
    provenance[20:24, 21:25] = np.uint16(2)  # observed reference source
    provenance[25:27, 25:27] = np.uint16(65534)  # symmetry must not be promoted
    provenance[27:29, 27:29] = np.uint16(65535)  # generated must not be promoted
    workspace.provenance_map = provenance

    build_severity_map(workspace)

    protected = np.asarray(workspace.metadata["protected_region_mask"]) > 0
    assert np.all(protected[20:24, 21:25])
    assert not np.any(protected[25:27, 25:27])
    assert not np.any(protected[27:29, 27:29])
    assert workspace.metadata["adaptive_preexisting_observed_protected_pixels"] == 16


def test_outer_guardrail_tracks_all_adaptive_cascade_state() -> None:
    keys = set(AutomaticPipelineRunner._GUARDRAIL_METADATA_KEYS)
    required = {
        "protected_region_mask",
        "adaptive_blur_classification",
        "adaptive_severity_map",
        "adaptive_severity_counts",
        "adaptive_restoration_stage",
        "adaptive_restoration_stage_mask",
        "adaptive_restoration_reports",
        "adaptive_restoration_remaining_mask",
        "adaptive_preexisting_observed_protected_pixels",
        "adaptive_preexisting_observed_protection",
    }
    assert required <= keys


def test_outer_guardrail_restore_removes_stale_adaptive_state_after_rejection() -> None:
    image = np.full((48, 48, 3), 90, dtype=np.uint8)
    workspace = Workspace(image.copy())
    runner = AutomaticPipelineRunner(workspace)

    original_protected = np.zeros((48, 48), dtype=np.uint8)
    original_protected[5:9, 5:9] = 255
    workspace.metadata["protected_region_mask"] = original_protected.copy()
    snapshot = runner._snapshot_guardrail_state()

    workspace.metadata["protected_region_mask"] = np.full((48, 48), 255, dtype=np.uint8)
    workspace.metadata["adaptive_restoration_stage"] = "severe"
    workspace.metadata["adaptive_restoration_reports"] = [{"accepted": True}]
    workspace.metadata["adaptive_restoration_remaining_mask"] = np.zeros((48, 48), dtype=np.uint8)

    runner._restore_guardrail_state(snapshot)

    assert np.array_equal(workspace.metadata["protected_region_mask"], original_protected)
    assert "adaptive_restoration_stage" not in workspace.metadata
    assert "adaptive_restoration_reports" not in workspace.metadata
    assert "adaptive_restoration_remaining_mask" not in workspace.metadata
