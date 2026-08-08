from __future__ import annotations

import numpy as np

from app.execution import Workspace
from app.observed_target_repair_runtime import repair_observed_target


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clean = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.indices((64, 64))
    clean[..., 0] = (40 + xx * 2).astype(np.uint8)
    clean[..., 1] = (55 + yy * 2).astype(np.uint8)
    clean[..., 2] = (70 + (xx + yy)).astype(np.uint8)
    target = np.zeros((64, 64), dtype=np.uint8)
    target[22:42, 20:44] = 255
    damaged = clean.copy()
    damaged[target > 0] = (8, 8, 8)
    return clean, damaged, target


def _workspace(primary: np.ndarray, refs: list[np.ndarray], target: np.ndarray) -> Workspace:
    shape = primary.shape[:2]
    return Workspace(
        primary=primary.copy(),
        references=[item.copy() for item in refs],
        aligned_references=[item.copy() for item in refs],
        occlusion_masks=[target.copy()],
        metadata={
            "primary_bbox": (0, 0, shape[1], shape[0]),
            "aligned_reference_support_masks": [np.where(np.max(item, axis=2) > 2, 255, 0).astype(np.uint8) for item in refs],
            "aligned_reference_detail_reliability_maps": [np.full(shape, 255, dtype=np.uint8) for _ in refs],
            "aligned_reference_original_source_indices": [index + 1 for index in range(len(refs))],
            "aligned_reference_identity_verified": [True for _ in refs],
            "aligned_reference_partial_geometry_verified": [False for _ in refs],
            "inpaint_target_mask": target.copy(),
        },
    )


def test_full_clean_donor_repairs_occluded_target_without_touching_visible_pixels() -> None:
    clean, damaged, target = _scene()
    workspace = _workspace(damaged, [clean], target)
    result, provenance, details = repair_observed_target(workspace, damaged, maximum_face_fraction=1.0)
    assert details["applied"] is True
    assert np.array_equal(result[target > 0], clean[target > 0])
    assert np.array_equal(result[target == 0], damaged[target == 0])
    assert np.all(provenance[target > 0] == 1)
    assert np.all(provenance[target == 0] == 0)


def test_complementary_partial_donors_cover_target_together() -> None:
    clean, damaged, target = _scene()
    left = np.zeros_like(clean); left[:, :34] = clean[:, :34]
    right = np.zeros_like(clean); right[:, 30:] = clean[:, 30:]
    workspace = _workspace(damaged, [left, right], target)
    workspace.metadata["aligned_reference_identity_verified"] = [False, False]
    workspace.metadata["aligned_reference_partial_geometry_verified"] = [True, True]
    result, provenance, details = repair_observed_target(workspace, damaged, maximum_face_fraction=1.0)
    assert details["applied"] is True
    assert np.array_equal(result[target > 0], clean[target > 0])
    assert set(np.unique(provenance[target > 0]).tolist()).issubset({1, 2})
    assert np.all(provenance[target > 0] > 0)


def test_component_only_donor_transfers_only_observed_support() -> None:
    clean, damaged, target = _scene()
    component = np.zeros_like(clean)
    component[28:36, 24:40] = clean[28:36, 24:40]
    workspace = _workspace(damaged, [component], target)
    workspace.metadata["aligned_reference_identity_verified"] = [False]
    workspace.metadata["aligned_reference_partial_geometry_verified"] = [True]
    result, provenance, details = repair_observed_target(workspace, damaged, maximum_face_fraction=1.0)
    observed = np.max(component, axis=2) > 2
    expected = (target > 0) & observed
    assert details["applied"] is True
    assert np.array_equal(result[expected], clean[expected])
    assert np.array_equal(result[(target > 0) & ~observed], damaged[(target > 0) & ~observed])
    assert np.all(provenance[expected] == 1)
    assert np.all(provenance[~expected] == 0)


def test_untrusted_reference_is_rejected() -> None:
    clean, damaged, target = _scene()
    workspace = _workspace(damaged, [clean], target)
    workspace.metadata["aligned_reference_identity_verified"] = [False]
    workspace.metadata["aligned_reference_partial_geometry_verified"] = [False]
    result, provenance, details = repair_observed_target(workspace, damaged, maximum_face_fraction=1.0)
    assert details["applied"] is False
    assert np.array_equal(result, damaged)
    assert not np.any(provenance)


def test_preflight_accepted_identity_remains_trusted_after_runtime_reorder() -> None:
    clean, damaged, target = _scene()
    workspace = _workspace(damaged, [clean], target)
    workspace.metadata.pop("aligned_reference_original_source_indices", None)
    workspace.metadata["aligned_reference_identity_verified"] = []
    workspace.metadata["aligned_reference_partial_geometry_verified"] = []
    workspace.metadata["aligned_reference_source_indices"] = [0]
    workspace.metadata["runtime_source_order"] = [0, 2]
    workspace.metadata["preflight_candidates"] = [
        {"source_index": 0, "accepted_identity": True},
        {"source_index": 1, "accepted_identity": False},
        {"source_index": 2, "accepted_identity": True},
    ]
    result, provenance, details = repair_observed_target(workspace, damaged, maximum_face_fraction=1.0)
    assert details["applied"] is True
    assert details["original_source_indices"] == [2]
    assert np.array_equal(result[target > 0], clean[target > 0])
    assert np.all(provenance[target > 0] == 2)


def test_same_canvas_primary_anchor_trusts_original_reference_without_legacy_flag() -> None:
    clean, damaged, target = _scene()
    workspace = _workspace(damaged, [clean], target)
    workspace.metadata.pop("aligned_reference_original_source_indices", None)
    workspace.metadata["aligned_reference_identity_verified"] = []
    workspace.metadata["aligned_reference_partial_geometry_verified"] = []
    workspace.metadata["aligned_reference_source_indices"] = [0]
    workspace.metadata["runtime_source_order"] = [0, 3]
    workspace.metadata["same_canvas_primary_anchor"] = {"applied": True, "matched_original_reference_indices": [3]}
    result, provenance, details = repair_observed_target(workspace, damaged, maximum_face_fraction=1.0)
    assert details["applied"] is True
    assert details["original_source_indices"] == [3]
    assert np.array_equal(result[target > 0], clean[target > 0])
    assert np.all(provenance[target > 0] == 3)


def test_low_detail_observed_pixels_remain_valid_repair_evidence_by_default() -> None:
    clean, damaged, target = _scene()
    smooth = np.full_like(clean, (120, 130, 140), dtype=np.uint8)
    workspace = _workspace(damaged, [smooth], target)
    workspace.metadata["aligned_reference_detail_reliability_maps"] = [np.zeros(target.shape, dtype=np.uint8)]
    result, provenance, details = repair_observed_target(workspace, damaged, maximum_face_fraction=1.0)
    assert details["minimum_reliability"] == 0
    assert details["applied"] is True
    assert np.array_equal(result[target > 0], smooth[target > 0])
    assert np.all(provenance[target > 0] == 1)
