from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.tiny_observed_evidence_policy import _complete_observed_pixels, _source_codes


def _base_workspace(count: int = 1) -> tuple[Workspace, np.ndarray]:
    primary = np.full((20, 20, 3), 120, dtype=np.uint8)
    refs = [primary.copy() for _ in range(count)]
    workspace = Workspace(primary=primary.copy(), references=[item.copy() for item in refs])
    workspace.aligned_references = [item.copy() for item in refs]

    target = np.zeros((20, 20), dtype=np.uint8)
    target[9, 10] = 255
    workspace.metadata["inpaint_target_mask"] = target
    workspace.metadata["aligned_reference_support_masks"] = [target.copy() for _ in range(count)]
    workspace.metadata["aligned_reference_detail_reliability_maps"] = [
        np.full((20, 20), 100 - index, dtype=np.uint8) for index in range(count)
    ]
    workspace.metadata["aligned_reference_identity_verified"] = [False] * count
    workspace.metadata["aligned_reference_partial_geometry_verified"] = [True] * count
    workspace.metadata["aligned_reference_original_source_indices"] = list(range(1, count + 1))
    # Deliberately mark the reference pixel as heuristically occluded. The evidence map
    # must be authoritative once cross-reference preclean has established provenance.
    workspace.occlusion_masks = [target.copy(), *[target.copy() for _ in range(count)]]
    return workspace, target


def test_authoritative_evidence_map_overrides_heuristic_reference_occlusion_veto() -> None:
    workspace, _ = _base_workspace(1)
    workspace.aligned_references[0][9, 10] = (35, 55, 75)
    evidence = np.zeros((20, 20), dtype=np.uint16)
    evidence[9, 10] = np.uint16(1)
    workspace.metadata["preclean_reference_evidence_maps"] = [evidence]

    output, details = _complete_observed_pixels(workspace, workspace.primary.copy())

    assert tuple(int(v) for v in output[9, 10]) == (35, 55, 75)
    assert int(workspace.provenance_map[9, 10]) == 1
    assert details["tiny_observed_pixels"] == 1
    assert details["preclean_evidence_authoritative"] is True


def test_cleaned_working_reference_preserves_true_donor_source_code() -> None:
    workspace, _ = _base_workspace(2)
    # Slot 0 is a cleaned working reference, but this pixel actually came from source 2.
    workspace.aligned_references[0][9, 10] = (44, 66, 88)
    workspace.aligned_references[1][9, 10] = (10, 20, 30)
    evidence0 = np.zeros((20, 20), dtype=np.uint16)
    evidence1 = np.zeros((20, 20), dtype=np.uint16)
    evidence0[9, 10] = np.uint16(2)
    # Slot 1 is deliberately not eligible at this pixel, so slot 0 wins unambiguously.
    workspace.metadata["preclean_reference_evidence_maps"] = [evidence0, evidence1]

    output, details = _complete_observed_pixels(workspace, workspace.primary.copy())

    assert tuple(int(v) for v in output[9, 10]) == (44, 66, 88)
    assert int(workspace.provenance_map[9, 10]) == 2
    assert details["tiny_observed_sources"] == [2]
    assert details["true_source_provenance_preserved"] is True


def test_zero_evidence_remains_blocked_even_when_heuristic_mask_is_clean() -> None:
    workspace, target = _base_workspace(1)
    workspace.aligned_references[0][9, 10] = (35, 55, 75)
    workspace.occlusion_masks = [target.copy(), np.zeros((20, 20), dtype=np.uint8)]
    workspace.metadata["preclean_reference_evidence_maps"] = [np.zeros((20, 20), dtype=np.uint16)]

    output, details = _complete_observed_pixels(workspace, workspace.primary.copy())

    assert np.array_equal(output, workspace.primary)
    assert details["tiny_observed_pixels"] == 0
    assert details["preclean_evidence_authoritative"] is True


def test_original_source_indices_are_already_provenance_codes() -> None:
    workspace, _ = _base_workspace(2)
    workspace.metadata["aligned_reference_original_source_indices"] = [1, 9]

    assert _source_codes(workspace, 2) == [1, 9]
