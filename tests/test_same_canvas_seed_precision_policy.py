from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.same_canvas_seed_precision_policy import precise_same_canvas_damage_seed


def test_verified_inpaint_target_overrides_broad_heuristic_primary_mask() -> None:
    shape = (40, 40)
    broad = np.full(shape, 255, dtype=np.uint8)
    target = np.zeros(shape, dtype=np.uint8)
    target[17:23, 18:22] = 255
    workspace = SimpleNamespace(
        metadata={
            "inpaint_target_mask": target.copy(),
            "preflight_original_occlusion_masks": [broad.copy()],
        },
        occlusion_masks=[broad.copy()],
    )

    selected = precise_same_canvas_damage_seed(workspace, shape)

    assert np.array_equal(selected, target)
    assert np.count_nonzero(selected) == 24


def test_reference_consensus_precedes_broad_detector_when_no_inpaint_target() -> None:
    shape = (32, 32)
    broad = np.full(shape, 255, dtype=np.uint8)
    consensus = np.zeros(shape, dtype=np.uint8)
    consensus[8:12, 9:15] = 255
    workspace = SimpleNamespace(
        metadata={
            "reference_consensus_occlusion": consensus.copy(),
            "preflight_original_occlusion_masks": [broad.copy()],
        },
        occlusion_masks=[broad.copy()],
    )

    selected = precise_same_canvas_damage_seed(workspace, shape)

    assert np.array_equal(selected, consensus)
    assert np.count_nonzero(selected) == 24


def test_broad_primary_mask_remains_last_resort_fallback() -> None:
    shape = (20, 20)
    fallback = np.zeros(shape, dtype=np.uint8)
    fallback[4:8, 5:10] = 255
    workspace = SimpleNamespace(metadata={}, occlusion_masks=[fallback.copy()])

    selected = precise_same_canvas_damage_seed(workspace, shape)

    assert np.array_equal(selected, fallback)
