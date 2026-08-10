from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.full_residual_reconstruction_policy import _demonstrated_residual


def test_empty_inner_unresolved_mask_does_not_hide_unrepaired_outer_target() -> None:
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    workspace = Workspace(primary=image.copy())
    target = np.zeros((32, 32), dtype=np.uint8)
    target[8:24, 8:24] = 255

    # Historical failure: an inner sub-handler could clear this mask even though the
    # larger adaptive-stage ROI had not changed at all.
    workspace.metadata["inpaint_unresolved_mask"] = np.zeros((32, 32), dtype=np.uint8)

    residual, diagnostics = _demonstrated_residual(workspace, image, image.copy(), target)

    assert np.array_equal(residual, target)
    assert diagnostics["target_pixels"] == 16 * 16
    assert diagnostics["demonstrated_residual_pixels"] == 16 * 16


def test_reference_provenance_changed_pixels_are_demonstrably_resolved() -> None:
    before = np.full((32, 32, 3), 100, dtype=np.uint8)
    after = before.copy()
    target = np.zeros((32, 32), dtype=np.uint8)
    target[8:24, 8:24] = 255

    repaired = np.zeros((32, 32), dtype=bool)
    repaired[10:18, 11:19] = True
    after[repaired] = (70, 80, 90)

    workspace = Workspace(primary=after.copy())
    provenance = np.zeros((32, 32), dtype=np.uint16)
    provenance[repaired] = np.uint16(2)
    workspace.provenance_map = provenance
    workspace.metadata["inpaint_unresolved_mask"] = np.zeros((32, 32), dtype=np.uint8)

    residual, diagnostics = _demonstrated_residual(workspace, before, after, target)

    assert not np.any(residual[repaired])
    expected = int(np.count_nonzero(target)) - int(np.count_nonzero(repaired))
    assert int(np.count_nonzero(residual)) == expected
    assert diagnostics["reference_provenance_changed_pixels"] == int(np.count_nonzero(repaired))


def test_unresolved_hint_does_not_erase_outer_residual() -> None:
    before = np.full((24, 24, 3), 80, dtype=np.uint8)
    workspace = Workspace(primary=before.copy())
    target = np.zeros((24, 24), dtype=np.uint8)
    target[5:19, 5:19] = 255
    hint = np.zeros((24, 24), dtype=np.uint8)
    hint[7:10, 7:10] = 255
    workspace.metadata["inpaint_unresolved_mask"] = hint

    residual, _ = _demonstrated_residual(workspace, before, before.copy(), target)

    # The outer target already defines every pixel that still lacks demonstrated repair;
    # a narrower inner hint is therefore not allowed to shrink it.
    assert np.all(residual[target > 0] == 255)


def test_stale_unresolved_hint_cannot_reopen_observed_reference_evidence() -> None:
    before = np.full((24, 24, 3), 90, dtype=np.uint8)
    after = before.copy()
    target = np.zeros((24, 24), dtype=np.uint8)
    target[4:20, 4:20] = 255

    repaired = np.zeros((24, 24), dtype=bool)
    repaired[8:13, 9:14] = True
    after[repaired] = (40, 60, 80)

    workspace = Workspace(primary=after.copy())
    provenance = np.zeros((24, 24), dtype=np.uint16)
    provenance[repaired] = np.uint16(3)
    workspace.provenance_map = provenance
    stale = np.zeros((24, 24), dtype=np.uint8)
    stale[repaired] = 255
    workspace.metadata["inpaint_unresolved_mask"] = stale

    residual, diagnostics = _demonstrated_residual(workspace, before, after, target)

    assert not np.any(residual[repaired])
    assert diagnostics["ignored_inner_hint_on_demonstrated_pixels"] == int(np.count_nonzero(repaired))
