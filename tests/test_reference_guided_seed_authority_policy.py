from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.reference_guided_seed_authority_policy import (
    constrain_verified_reference_guided_target,
    constrain_verified_reference_guided_transfer,
    prefer_verified_reference_guided_seed,
)


def _mask(shape: tuple[int, int], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    result[y0:y1, x0:x1] = 255
    return result


def _verified_workspace() -> Workspace:
    shape = (64, 64)
    workspace = Workspace(primary=np.zeros((*shape, 3), dtype=np.uint8), references=[])
    authority = _mask(shape, 20, 40, 12, 24)
    broad_consensus = _mask(shape, 8, 56, 6, 58)
    workspace.metadata["reference_guided_authority_mask"] = authority
    workspace.metadata["reference_consensus_occlusion"] = broad_consensus
    workspace.metadata["reference_guided_seed_diagnostics"] = {
        "reason": "reference_guided_frozen_seed",
        "trusted_donors": 2,
        "refined_pixels": int(np.count_nonzero(authority)),
        "seed_expansion_from_partial_reference": False,
    }
    return workspace


def test_verified_reference_guided_authority_blocks_later_broad_inpaint_expansion() -> None:
    workspace = _verified_workspace()
    shape = workspace.primary.shape[:2]
    authority = workspace.metadata["reference_guided_authority_mask"]
    broad = workspace.metadata["reference_consensus_occlusion"]
    workspace.metadata["inpaint_target_mask"] = broad

    chosen = prefer_verified_reference_guided_seed(workspace, shape, broad)

    assert np.array_equal(chosen, authority)
    assert int(np.count_nonzero(chosen)) < int(np.count_nonzero(broad))


def test_verified_reference_guided_authority_keeps_narrower_inpaint_validation() -> None:
    workspace = _verified_workspace()
    shape = workspace.primary.shape[:2]
    narrower = _mask(shape, 25, 35, 14, 20)
    workspace.metadata["inpaint_target_mask"] = narrower

    chosen = prefer_verified_reference_guided_seed(workspace, shape, narrower)

    assert np.array_equal(chosen, narrower)


def test_verified_authority_clamps_proposed_inpaint_target_before_transfer() -> None:
    workspace = _verified_workspace()
    authority = workspace.metadata["reference_guided_authority_mask"]
    broad = workspace.metadata["reference_consensus_occlusion"]

    constrained = constrain_verified_reference_guided_target(workspace, broad)

    assert np.array_equal(constrained, authority)


def test_unverified_authority_preserves_historical_inpaint_target() -> None:
    workspace = _verified_workspace()
    broad = workspace.metadata["reference_consensus_occlusion"]
    workspace.metadata["reference_guided_seed_diagnostics"]["trusted_donors"] = 0

    constrained = constrain_verified_reference_guided_target(workspace, broad)

    assert np.array_equal(constrained, broad)


def test_unverified_consensus_preserves_historical_inpaint_seed_priority() -> None:
    workspace = _verified_workspace()
    shape = workspace.primary.shape[:2]
    broad = workspace.metadata["reference_consensus_occlusion"]
    workspace.metadata["inpaint_target_mask"] = broad
    workspace.metadata["reference_guided_seed_diagnostics"]["trusted_donors"] = 0

    chosen = prefer_verified_reference_guided_seed(workspace, shape, broad)

    assert np.array_equal(chosen, broad)


def test_verified_authority_clamps_same_canvas_transfer_output() -> None:
    workspace = _verified_workspace()
    shape = workspace.primary.shape[:2]
    authority = workspace.metadata["reference_guided_authority_mask"] > 0
    broad = workspace.metadata["reference_consensus_occlusion"] > 0
    input_image = np.zeros((*shape, 3), dtype=np.uint8)
    repaired = input_image.copy()
    repaired[broad] = 200
    provenance = np.zeros(shape, dtype=np.uint16)
    provenance[broad] = 1

    constrained, constrained_provenance, details = constrain_verified_reference_guided_transfer(
        workspace,
        input_image,
        repaired,
        provenance,
        {"applied": True},
    )

    changed = np.any(constrained != input_image, axis=2)
    assert np.array_equal(changed, authority)
    assert np.all(constrained_provenance[~authority] == 0)
    assert details["reference_guided_clamped_transfer_pixels"] == int(np.count_nonzero(broad & ~authority))


def test_unverified_authority_does_not_change_same_canvas_transfer_output() -> None:
    workspace = _verified_workspace()
    workspace.metadata["reference_guided_seed_diagnostics"]["trusted_donors"] = 0
    shape = workspace.primary.shape[:2]
    broad = workspace.metadata["reference_consensus_occlusion"] > 0
    input_image = np.zeros((*shape, 3), dtype=np.uint8)
    repaired = input_image.copy()
    repaired[broad] = 200
    provenance = np.zeros(shape, dtype=np.uint16)
    provenance[broad] = 1

    constrained, constrained_provenance, details = constrain_verified_reference_guided_transfer(
        workspace,
        input_image,
        repaired,
        provenance,
        {"applied": True},
    )

    assert np.array_equal(constrained, repaired)
    assert np.array_equal(constrained_provenance, provenance)
    assert details == {"applied": True}
