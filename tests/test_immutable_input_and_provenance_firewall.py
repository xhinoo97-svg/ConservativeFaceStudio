from __future__ import annotations

import cv2
import numpy as np

from app.execution import Workspace
from app.immutable_input_store import ensure_immutable_input_store
from app.provenance_firewall_policy import install_provenance_firewall_policy
from app.observed_target_repair_runtime import repair_observed_target
from app.tiny_observed_evidence_policy import _complete_observed_pixels


def test_immutable_input_store_survives_working_mutation() -> None:
    main = np.full((32, 32, 3), 40, dtype=np.uint8)
    ref = np.full((32, 32, 3), 90, dtype=np.uint8)
    workspace = Workspace(primary=main.copy(), references=[ref.copy()])

    store = ensure_immutable_input_store(workspace)
    expected_main = store.copy_main()
    expected_ref = store.copy_reference(0)

    workspace.primary[:] = 220
    workspace.references[0][:] = 10

    assert np.array_equal(store.copy_main(), expected_main)
    assert np.array_equal(store.copy_reference(0), expected_ref)
    assert store.main.flags.writeable is False
    assert store.references[0].flags.writeable is False
    assert workspace.metadata["immutable_input_manifest"]["source_count"] == 2


def _trusted_workspace_for_firewall() -> tuple[Workspace, np.ndarray]:
    primary = np.full((64, 64, 3), 80, dtype=np.uint8)
    reference = primary.copy()
    damage = np.zeros(primary.shape[:2], dtype=np.uint8)
    cv2.rectangle(damage, (28, 28), (30, 30), 255, -1)
    reference[damage > 0] = (10, 20, 30)

    workspace = Workspace(primary=primary.copy(), references=[reference.copy()])
    workspace.aligned_references = [reference.copy()]
    workspace.occlusion_masks = [damage.copy(), np.zeros_like(damage)]
    workspace.metadata["primary_bbox"] = (8, 8, 48, 48)
    workspace.metadata["aligned_reference_original_source_indices"] = [1]
    workspace.metadata["aligned_reference_source_indices"] = [0]
    workspace.metadata["aligned_reference_identity_verified"] = [True]
    workspace.metadata["aligned_reference_partial_geometry_verified"] = [True]
    workspace.metadata["aligned_reference_support_masks"] = [np.full_like(damage, 255)]
    workspace.metadata["aligned_reference_detail_reliability_maps"] = [np.full_like(damage, 255)]
    workspace.metadata["preflight_original_occlusion_masks"] = [damage.copy(), np.zeros_like(damage)]
    return workspace, damage


def test_observed_repair_rejects_working_pixel_without_evidence_source() -> None:
    install_provenance_firewall_policy()
    workspace, damage = _trusted_workspace_for_firewall()
    evidence = np.zeros(damage.shape, dtype=np.uint16)
    # The cleaned working reference contains RGB data here, but the evidence map says
    # that these pixels are not backed by an observed photograph.
    workspace.metadata["preclean_reference_evidence_maps"] = [evidence]

    repaired, provenance, details = repair_observed_target(workspace, workspace.primary)

    assert details["provenance_firewall"] is True
    assert details["firewall_rejected_non_evidence_pixels"] > 0
    assert not np.any(provenance[damage > 0])
    assert np.array_equal(repaired, workspace.primary)


def test_observed_repair_reassigns_cross_cleaned_pixel_to_true_source() -> None:
    install_provenance_firewall_policy()
    workspace, damage = _trusted_workspace_for_firewall()
    # Simulate REF1 working copy having been repaired using an actually observed pixel
    # from imported REF2.  Provenance must follow REF2, never the container REF1.
    evidence = np.zeros(damage.shape, dtype=np.uint16)
    evidence[damage > 0] = np.uint16(2)
    workspace.metadata["preclean_reference_evidence_maps"] = [evidence]

    repaired, provenance, details = repair_observed_target(workspace, workspace.primary)

    assert details["provenance_firewall"] is True
    assert details["firewall_reassigned_true_source_pixels"] > 0
    assert np.all(provenance[damage > 0] == 2)
    assert np.array_equal(repaired[damage > 0], workspace.aligned_references[0][damage > 0])


def test_tiny_completion_cannot_promote_ineligible_working_pixel() -> None:
    install_provenance_firewall_policy()
    workspace, damage = _trusted_workspace_for_firewall()
    workspace.metadata["inpaint_target_mask"] = damage.copy()
    workspace.metadata["preclean_reference_evidence_maps"] = [np.zeros(damage.shape, dtype=np.uint16)]
    workspace.provenance_map = np.zeros(damage.shape, dtype=np.uint16)

    completed, details = _complete_observed_pixels(workspace, workspace.primary)

    assert details["provenance_firewall"] is True
    assert details["firewall_rejected_non_evidence_pixels"] > 0
    assert np.array_equal(completed, workspace.primary)
    assert not np.any(workspace.provenance_map[damage > 0])
