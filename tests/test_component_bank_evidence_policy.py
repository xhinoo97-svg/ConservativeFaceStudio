from __future__ import annotations

import numpy as np

from app.component_bank_evidence_policy import _original_eligibility_masks
from app.execution import Workspace


def test_component_bank_eligibility_excludes_reference_damage_even_after_preclean() -> None:
    shape = (48, 48)
    primary = np.full((*shape, 3), 80, dtype=np.uint8)
    reference = np.full_like(primary, 120)
    workspace = Workspace(primary=primary, references=[reference.copy()])
    workspace.aligned_references = [reference.copy()]

    support = np.full(shape, 255, dtype=np.uint8)
    damage = np.zeros(shape, dtype=np.uint8)
    damage[20:25, 20:25] = 255
    workspace.metadata["aligned_reference_support_masks"] = [support]
    workspace.occlusion_masks = [np.zeros(shape, np.uint8), damage.copy()]

    # The working reference may have been visually repaired in this region, but Block 7
    # must not relabel those pixels as original pixels of this container reference.
    workspace.metadata["preclean_reference_evidence_maps"] = [
        np.where(damage > 0, 2, 1).astype(np.uint16)
    ]

    eligibility = _original_eligibility_masks(workspace)

    assert len(eligibility) == 1
    assert np.all(eligibility[0][damage > 0] == 0)
    assert np.all(eligibility[0][damage == 0] == 255)
    assert workspace.metadata["component_bank_source_eligibility_pixels"][0] == int(np.count_nonzero(damage == 0))


def test_component_bank_eligibility_respects_geometric_support() -> None:
    shape = (40, 40)
    primary = np.full((*shape, 3), 60, dtype=np.uint8)
    reference = np.full_like(primary, 100)
    workspace = Workspace(primary=primary, references=[reference.copy()])
    workspace.aligned_references = [reference.copy()]

    support = np.zeros(shape, dtype=np.uint8)
    support[8:32, 10:30] = 255
    workspace.metadata["aligned_reference_support_masks"] = [support]
    workspace.occlusion_masks = [np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)]

    eligibility = _original_eligibility_masks(workspace)[0]

    assert np.array_equal(eligibility > 0, support > 0)
