from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.reference_guided_seed_policy import _trusted_reference_disagreement


def test_broad_frozen_mask_is_reduced_to_real_donor_disagreement() -> None:
    primary = np.full((96, 96, 3), 120, dtype=np.uint8)
    clean = primary.copy()
    primary[40:56, 30:66] = 10

    workspace = Workspace(primary=primary.copy(), references=[clean.copy()])
    workspace.aligned_references = [clean.copy()]
    workspace.metadata["aligned_reference_support_masks"] = [np.full((96, 96), 255, np.uint8)]
    workspace.metadata["same_canvas_imported_primary"] = primary.copy()

    frozen = np.zeros((96, 96), dtype=np.uint8)
    frozen[12:84, 12:84] = 255  # intentionally huge recall-oriented proposal
    refined, details = _trusted_reference_disagreement(workspace, frozen)

    assert int(np.count_nonzero(refined)) > 0
    assert int(np.count_nonzero(refined)) < int(np.count_nonzero(frozen)) * 0.20
    assert np.all(refined[40:56, 30:66] > 0)
    assert details["trusted_donors"] == 1
    assert details["baseline_proven_donors"] == 1


def test_non_same_canvas_reference_cannot_authorize_broad_seed() -> None:
    primary = np.full((96, 96, 3), 120, dtype=np.uint8)
    donor = np.full((96, 96, 3), 210, dtype=np.uint8)
    workspace = Workspace(primary=primary.copy(), references=[donor.copy()])
    workspace.aligned_references = [donor.copy()]
    workspace.metadata["aligned_reference_support_masks"] = [np.full((96, 96), 255, np.uint8)]
    workspace.metadata["same_canvas_imported_primary"] = primary.copy()
    frozen = np.zeros((96, 96), dtype=np.uint8)
    frozen[30:66, 30:66] = 255

    refined, details = _trusted_reference_disagreement(workspace, frozen)
    assert int(np.count_nonzero(refined)) == 0
    assert details["trusted_donors"] == 0


def test_reference_guided_seed_mostly_outside_face_fails_closed() -> None:
    primary = np.full((96, 96, 3), 120, dtype=np.uint8)
    donor = primary.copy()
    donor[4:24, 4:40] = 20

    workspace = Workspace(primary=primary.copy(), references=[donor.copy()])
    workspace.aligned_references = [donor.copy()]
    workspace.metadata["aligned_reference_support_masks"] = [np.full((96, 96), 255, np.uint8)]
    workspace.metadata["same_canvas_imported_primary"] = primary.copy()
    workspace.metadata["primary_bbox"] = (32, 30, 32, 42)

    frozen = np.zeros((96, 96), dtype=np.uint8)
    frozen[4:24, 4:40] = 255
    refined, details = _trusted_reference_disagreement(workspace, frozen)

    assert not np.any(refined)
    assert details["reason"] == "reference_guided_seed_outside_face_domain"
    assert details["rejected_refined_pixels"] > 0
    assert details["refined_face_fraction"] < 0.5


def test_trusted_component_only_reference_preserves_only_supported_existing_seed() -> None:
    primary = np.full((96, 96, 3), 120, dtype=np.uint8)
    donor = np.zeros_like(primary)
    support = np.zeros((96, 96), dtype=np.uint8)
    support[40:54, 42:58] = 255
    donor[support > 0] = (80, 105, 135)

    frozen = np.zeros((96, 96), dtype=np.uint8)
    frozen[30:70, 30:70] = 255
    workspace = Workspace(primary=primary.copy(), references=[donor.copy()])
    workspace.aligned_references = [donor.copy()]
    workspace.metadata["aligned_reference_support_masks"] = [support]
    workspace.metadata["aligned_reference_identity_verified"] = [False]
    workspace.metadata["aligned_reference_partial_geometry_verified"] = [True]
    workspace.metadata["same_canvas_imported_primary"] = primary.copy()

    refined, details = _trusted_reference_disagreement(workspace, frozen)
    expected = (support > 0) & (frozen > 0)
    assert np.array_equal(refined > 0, expected)
    assert details["trusted_partial_seed_only_pixels"] == int(np.count_nonzero(expected))
    assert details["seed_expansion_from_partial_reference"] is False


def test_untrusted_component_only_reference_without_baseline_is_rejected() -> None:
    primary = np.full((96, 96, 3), 120, dtype=np.uint8)
    donor = np.zeros_like(primary)
    support = np.zeros((96, 96), dtype=np.uint8)
    support[40:54, 42:58] = 255
    donor[support > 0] = (80, 105, 135)
    frozen = np.zeros((96, 96), dtype=np.uint8)
    frozen[30:70, 30:70] = 255

    workspace = Workspace(primary=primary.copy(), references=[donor.copy()])
    workspace.aligned_references = [donor.copy()]
    workspace.metadata["aligned_reference_support_masks"] = [support]
    workspace.metadata["aligned_reference_identity_verified"] = [False]
    workspace.metadata["aligned_reference_partial_geometry_verified"] = [False]
    workspace.metadata["same_canvas_imported_primary"] = primary.copy()

    refined, details = _trusted_reference_disagreement(workspace, frozen)
    assert not np.any(refined)
    assert details["trusted_donors"] == 0
