from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.reference_guided_seed_authority_policy import (
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
    consensus = _mask(shape, 20, 40, 12, 24)
    workspace.metadata["reference_consensus_occlusion"] = consensus
    workspace.metadata["reference_guided_seed_diagnostics"] = {
        "reason": "reference_guided_frozen_seed",
        "trusted_donors": 2,
        "refined_pixels": int(np.count_nonzero(consensus)),
        "seed_expansion_from_partial_reference": False,
    }
    return workspace


def test_verified_reference_guided_consensus_blocks_later_broad_inpaint_expansion() -> None:
    workspace = _verified_workspace()
    shape = workspace.primary.shape[:2]
    consensus = workspace.metadata["reference_consensus_occlusion"]
    broad = _mask(shape, 8, 56, 6, 58)
    workspace.metadata["inpaint_target_mask"] = broad

    chosen = prefer_verified_reference_guided_seed(workspace, shape, broad)

    assert np.array_equal(chosen, consensus)
    assert int(np.count_nonzero(chosen)) < int(np.count_nonzero(broad))


def test_verified_reference_guided_consensus_keeps_narrower_inpaint_validation() -> None:
    workspace = _verified_workspace()
    shape = workspace.primary.shape[:2]
    narrower = _mask(shape, 25, 35, 14, 20)
    workspace.metadata["inpaint_target_mask"] = narrower

    chosen = prefer_verified_reference_guided_seed(workspace, shape, narrower)

    assert np.array_equal(chosen, narrower)


def test_unverified_consensus_preserves_historical_inpaint_seed_priority() -> None:
    workspace = _verified_workspace()
    shape = workspace.primary.shape[:2]
    broad = _mask(shape, 8, 56, 6, 58)
    workspace.metadata["inpaint_target_mask"] = broad
    workspace.metadata["reference_guided_seed_diagnostics"]["trusted_donors"] = 0

    chosen = prefer_verified_reference_guided_seed(workspace, shape, broad)

    assert np.array_equal(chosen, broad)


def test_verified_authority_clamps_same_canvas_transfer_output_to_consensus() -> None:
    workspace = _verified_workspace()
    shape = workspace.primary.shape[:2]
    consensus = workspace.metadata["reference_consensus_occlusion"] > 0
    broad = _mask(shape, 8, 56, 6, 58) > 0
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
    assert np.array_equal(changed, consensus)
    assert np.all(constrained_provenance[~consensus] == 0)
    assert details["reference_guided_clamped_transfer_pixels"] == int(np.count_nonzero(broad & ~consensus))


def test_unverified_authority_does_not_change_same_canvas_transfer_output() -> None:
    workspace = _verified_workspace()
    workspace.metadata["reference_guided_seed_diagnostics"]["trusted_donors"] = 0
    shape = workspace.primary.shape[:2]
    broad = _mask(shape, 8, 56, 6, 58) > 0
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
