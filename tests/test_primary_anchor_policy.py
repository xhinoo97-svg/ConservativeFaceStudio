from __future__ import annotations

import cv2
import numpy as np

import app.primary_anchor_policy as policy
from app.execution import Workspace
from app.primary_anchor_policy import restore_imported_primary_for_same_canvas


def _face() -> np.ndarray:
    image = np.full((128, 128, 3), 30, dtype=np.uint8)
    cv2.ellipse(image, (64, 66), (42, 52), 0, 0, 360, (145, 170, 198), -1)
    cv2.circle(image, (50, 54), 4, (22, 22, 22), -1)
    cv2.circle(image, (78, 54), 4, (22, 22, 22), -1)
    cv2.line(image, (64, 60), (64, 80), (72, 82, 96), 2)
    return image


def _face_bboxes() -> list[tuple[int, int, int, int]]:
    return [(22, 14, 84, 104), (22, 14, 84, 104)]


def test_restores_imported_primary_when_preflight_selected_clean_same_canvas_reference() -> None:
    clean = _face()
    damaged = clean.copy()
    cv2.ellipse(damaged, (64, 64), (18, 12), 0, 0, 360, (12, 12, 12), -1)

    workspace = Workspace(primary=clean.copy(), references=[damaged.copy()])
    workspace.metadata["runtime_source_order"] = [1, 0]
    workspace.metadata["selected_primary_original_source_index"] = 1
    workspace.metadata["preflight_face_bboxes"] = _face_bboxes()
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
    evidence = workspace.metadata["same_canvas_primary_anchor"]
    assert evidence["matched_original_reference_indices"] == [1]
    assert evidence["identity_bridge_original_reference_indices"] == [1]
    assert evidence["identity_bridge_requires_face_local_observed_agreement"] is True


def test_does_not_reanchor_unrelated_same_size_reference() -> None:
    clean = _face()
    damaged = clean.copy()
    cv2.rectangle(damaged, (45, 48), (83, 78), (10, 10, 10), -1)
    unrelated = np.full_like(clean, 220)
    cv2.circle(unrelated, (30, 30), 20, (20, 180, 40), -1)

    workspace = Workspace(primary=unrelated.copy(), references=[damaged.copy()])
    workspace.metadata["runtime_source_order"] = [1, 0]
    workspace.metadata["selected_primary_original_source_index"] = 1
    workspace.metadata["preflight_face_bboxes"] = _face_bboxes()

    decision = restore_imported_primary_for_same_canvas(workspace, [damaged, unrelated])

    assert decision.applied is False
    assert workspace.metadata["runtime_source_order"] == [1, 0]
    assert np.array_equal(workspace.primary, unrelated)


def test_keeps_imported_primary_and_records_same_canvas_when_already_runtime_primary() -> None:
    clean = _face()
    damaged = clean.copy()
    cv2.rectangle(damaged, (48, 50), (80, 76), (10, 10, 10), -1)
    workspace = Workspace(primary=damaged.copy(), references=[clean.copy()])
    workspace.metadata["runtime_source_order"] = [0, 1]
    workspace.metadata["selected_primary_original_source_index"] = 0
    workspace.metadata["preflight_face_bboxes"] = _face_bboxes()

    decision = restore_imported_primary_for_same_canvas(workspace, [damaged, clean])

    assert decision.applied is False
    assert decision.matched_reference_count == 1
    assert decision.reason == "already_primary_same_canvas_verified"
    assert np.array_equal(workspace.primary, damaged)
    evidence = workspace.metadata["same_canvas_primary_anchor"]
    assert evidence["applied"] is False
    assert evidence["matched_original_reference_indices"] == [1]
    assert evidence["identity_bridge_original_reference_indices"] == [1]


def test_mosaic_same_canvas_identity_bridge_does_not_require_occlusion_seed(monkeypatch) -> None:
    clean = _face()
    mosaic = clean.copy()
    crop = clean[48:80, 44:84]
    small = cv2.resize(crop, (5, 4), interpolation=cv2.INTER_AREA)
    mosaic[48:80, 44:84] = cv2.resize(small, (40, 32), interpolation=cv2.INTER_NEAREST)

    monkeypatch.setattr(
        policy,
        "detect_occlusion_candidates",
        lambda image: np.zeros(image.shape[:2], dtype=np.uint8),
    )
    workspace = Workspace(primary=mosaic.copy(), references=[clean.copy()])
    workspace.metadata["runtime_source_order"] = [0, 1]
    workspace.metadata["selected_primary_original_source_index"] = 0
    workspace.metadata["preflight_face_bboxes"] = _face_bboxes()

    decision = restore_imported_primary_for_same_canvas(workspace, [mosaic, clean])

    assert decision.applied is False
    assert decision.matched_reference_count == 1
    assert decision.reason == "already_primary_same_canvas_verified_without_occlusion_seed"
    evidence = workspace.metadata["same_canvas_primary_anchor"]
    assert evidence["primary_occlusion_seed_present"] is False
    assert evidence["identity_bridge_region"] == "inner_face_peripheral_band_v1"
    assert evidence["matched_original_reference_indices"] == [1]
    assert evidence["identity_bridge_original_reference_indices"] == [1]
