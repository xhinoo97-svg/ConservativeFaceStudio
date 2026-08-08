from __future__ import annotations

import cv2
import numpy as np

from app.execution import Workspace
from app.primary_anchor_policy import restore_imported_primary_for_same_canvas


def _face() -> np.ndarray:
    image = np.full((128, 128, 3), 30, dtype=np.uint8)
    cv2.ellipse(image, (64, 66), (42, 52), 0, 0, 360, (145, 170, 198), -1)
    cv2.circle(image, (50, 54), 4, (22, 22, 22), -1)
    cv2.circle(image, (78, 54), 4, (22, 22, 22), -1)
    cv2.line(image, (64, 60), (64, 80), (72, 82, 96), 2)
    return image


def test_restores_imported_primary_when_preflight_selected_clean_same_canvas_reference() -> None:
    clean = _face()
    damaged = clean.copy()
    cv2.ellipse(damaged, (64, 64), (18, 12), 0, 0, 360, (12, 12, 12), -1)

    workspace = Workspace(primary=clean.copy(), references=[damaged.copy()])
    workspace.metadata["runtime_source_order"] = [1, 0]
    workspace.metadata["selected_primary_original_source_index"] = 1
    workspace.metadata["preflight_original_occlusion_masks"] = [
        np.zeros((128, 128), dtype=np.uint8),
        np.zeros((128, 128), dtype=np.uint8),
    ]
    workspace.metadata["preflight_detail_reliability_maps"] = [
        np.full((128, 128), 255, dtype=np.uint8),
        np.full((128, 128), 255, dtype=np.uint8),
    ]

    decision = restore_imported_primary_for_same_canvas(workspace, [damaged, clean])

    assert decision.applied is True
    assert workspace.metadata["runtime_source_order"] == [0, 1]
    assert workspace.metadata["selected_primary_original_source_index"] == 0
    assert np.array_equal(workspace.primary, damaged)
    assert np.array_equal(workspace.references[0], clean)


def test_does_not_reanchor_unrelated_same_size_reference() -> None:
    clean = _face()
    damaged = clean.copy()
    cv2.rectangle(damaged, (45, 48), (83, 78), (10, 10, 10), -1)
    unrelated = np.full_like(clean, 220)
    cv2.circle(unrelated, (30, 30), 20, (20, 180, 40), -1)

    workspace = Workspace(primary=unrelated.copy(), references=[damaged.copy()])
    workspace.metadata["runtime_source_order"] = [1, 0]
    workspace.metadata["selected_primary_original_source_index"] = 1

    decision = restore_imported_primary_for_same_canvas(workspace, [damaged, unrelated])

    assert decision.applied is False
    assert workspace.metadata["runtime_source_order"] == [1, 0]
    assert np.array_equal(workspace.primary, unrelated)


def test_keeps_imported_primary_when_it_is_already_runtime_primary() -> None:
    clean = _face()
    damaged = clean.copy()
    cv2.rectangle(damaged, (48, 50), (80, 76), (10, 10, 10), -1)
    workspace = Workspace(primary=damaged.copy(), references=[clean.copy()])
    workspace.metadata["runtime_source_order"] = [0, 1]
    workspace.metadata["selected_primary_original_source_index"] = 0

    decision = restore_imported_primary_for_same_canvas(workspace, [damaged, clean])

    assert decision.applied is False
    assert np.array_equal(workspace.primary, damaged)
