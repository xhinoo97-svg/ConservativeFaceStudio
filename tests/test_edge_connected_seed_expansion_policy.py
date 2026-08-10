from __future__ import annotations

import cv2
import numpy as np

from app.edge_connected_seed_expansion_policy import (
    expand_edge_connected_seed,
    verify_edge_connected_seed_overlap,
)
from app.execution import Workspace


def _workspace_with_seed() -> tuple[Workspace, np.ndarray]:
    primary = np.full((64, 64, 3), 130, dtype=np.uint8)
    workspace = Workspace(primary.copy())
    damage = np.zeros((64, 64), dtype=np.uint8)
    damage[20:44, 20:44] = 255
    workspace.metadata["preflight_original_occlusion_masks"] = [damage, np.zeros_like(damage)]
    workspace.metadata["preflight_detail_reliability_maps"] = [
        np.full((64, 64), 255, dtype=np.uint8),
        np.full((64, 64), 255, dtype=np.uint8),
    ]
    return workspace, damage


def _textured_sparse_reference(x0: int, x1: int) -> np.ndarray:
    reference = np.zeros((64, 64, 3), dtype=np.uint8)
    yy, xx = np.indices((20, x1 - x0))
    values = np.where((xx + yy) % 2 == 0, 80, 175).astype(np.uint8)
    patch = np.stack((values, np.clip(values + 10, 0, 255), np.clip(values + 20, 0, 255)), axis=2)
    reference[22:42, x0:x1] = patch
    return reference


def test_two_pixel_detector_border_can_be_geometry_verified() -> None:
    workspace, _ = _workspace_with_seed()
    reference = _textured_sparse_reference(18, 46)  # two pixels outside each horizontal seed edge

    verified = verify_edge_connected_seed_overlap(workspace, reference, 0)

    assert verified is not None
    support, reliability, details = verified
    assert details["edge_connected_seed_tolerance"] is True
    assert details["may_expand_damage_seed"] is True
    assert details["edge_tolerance_pixels"] == 2
    assert details["damage_overlap_fraction"] < 0.95
    assert details["dilated_damage_overlap_fraction"] >= 0.995
    assert np.count_nonzero(support) > 0
    assert np.count_nonzero(reliability) > 0


def test_reference_extending_beyond_two_pixel_border_is_rejected() -> None:
    workspace, _ = _workspace_with_seed()
    reference = _textured_sparse_reference(15, 49)

    assert verify_edge_connected_seed_overlap(workspace, reference, 0) is None


def test_seed_expansion_is_only_connected_and_never_adds_distant_island() -> None:
    workspace, damage = _workspace_with_seed()
    support = np.zeros((64, 64), dtype=np.uint8)
    support[22:42, 18:46] = 255
    support[5:8, 5:8] = 255  # deliberately impossible distant island

    workspace.metadata["aligned_reference_support_masks"] = [support]
    workspace.metadata["aligned_reference_source_indices"] = [0]
    workspace.metadata["same_canvas_partial_alignment_diagnostics"] = [
        {
            "runtime_reference_index": 0,
            "edge_connected_seed_tolerance": True,
            "may_expand_damage_seed": True,
        }
    ]
    workspace.metadata["reference_consensus_occlusion"] = damage.copy()

    result = expand_edge_connected_seed(workspace)
    merged = np.asarray(workspace.metadata["reference_consensus_occlusion"]) > 0

    assert result["added_pixels"] > 0
    assert result["distant_expansion_allowed"] is False
    assert not np.any(merged[5:8, 5:8])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    allowed = cv2.dilate(damage, kernel, iterations=1) > 0
    assert not np.any(merged & ~allowed)


def test_existing_coordinate_preserving_sparse_verifier_can_also_recover_edge() -> None:
    workspace, damage = _workspace_with_seed()
    support = np.zeros((64, 64), dtype=np.uint8)
    support[22:42, 18:46] = 255

    workspace.metadata["aligned_reference_support_masks"] = [support]
    workspace.metadata["aligned_reference_source_indices"] = [0]
    workspace.metadata["same_canvas_partial_alignment_diagnostics"] = [
        {
            "runtime_reference_index": 0,
            "method": "verified-same-canvas-partial",
            "verification_basis": "coordinate-preserving-sparse-damage-overlap",
            "global_transform_required": False,
            "local_identity_transform": True,
        }
    ]
    workspace.metadata["reference_consensus_occlusion"] = damage.copy()

    result = expand_edge_connected_seed(workspace)

    assert result["eligible_donors"] == 1
    assert result["added_pixels"] > 0
