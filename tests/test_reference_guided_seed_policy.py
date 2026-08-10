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
