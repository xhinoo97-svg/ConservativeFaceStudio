from __future__ import annotations

import cv2
import numpy as np

from app.execution import Workspace
from app.same_canvas_repair_runtime import exact_same_canvas_observed_repair


def _verified_workspace(primary: np.ndarray, reference: np.ndarray, support: np.ndarray, damage: np.ndarray) -> Workspace:
    workspace = Workspace(primary=primary.copy(), references=[reference.copy()])
    workspace.aligned_references = [reference.copy()]
    workspace.occlusion_masks = [damage.copy(), np.zeros_like(damage)]
    workspace.metadata["primary_bbox"] = (24, 20, 48, 56)
    workspace.metadata["aligned_reference_source_indices"] = [0]
    workspace.metadata["aligned_reference_original_source_indices"] = [1]
    workspace.metadata["aligned_reference_support_masks"] = [support.copy()]
    workspace.metadata["preflight_original_occlusion_masks"] = [damage.copy(), np.zeros_like(damage)]
    workspace.metadata["verified_same_canvas_alignment"] = [
        {"runtime_reference_index": 0, "method": "verified-same-canvas-observed"}
    ]
    return workspace


def test_verified_black_donor_is_not_rejected_by_intensity() -> None:
    primary = np.full((96, 96, 3), 120, dtype=np.uint8)
    reference = primary.copy()
    damage = np.zeros(primary.shape[:2], dtype=np.uint8)
    cv2.rectangle(damage, (40, 40), (48, 48), 255, -1)
    support = damage.copy()
    reference[damage > 0] = (0, 0, 0)
    primary[damage > 0] = (220, 220, 220)
    workspace = _verified_workspace(primary, reference, support, damage)

    repaired, provenance, details = exact_same_canvas_observed_repair(workspace, primary)

    assert details["support_mask_is_authoritative_for_donor_validity"] is True
    assert np.array_equal(repaired[damage > 0], reference[damage > 0])
    assert np.all(provenance[damage > 0] == 1)
    assert np.array_equal(repaired[damage == 0], primary[damage == 0])


def test_verified_seed_support_outside_face_template_is_preserved() -> None:
    primary = np.full((96, 96, 3), 90, dtype=np.uint8)
    reference = primary.copy()
    damage = np.zeros(primary.shape[:2], dtype=np.uint8)
    # Simulates observed damage at a hair/face edge beyond the approximate face mask.
    cv2.rectangle(damage, (4, 34), (12, 44), 255, -1)
    support = damage.copy()
    reference[damage > 0] = (18, 24, 31)
    primary[damage > 0] = (235, 235, 235)
    workspace = _verified_workspace(primary, reference, support, damage)

    repaired, provenance, details = exact_same_canvas_observed_repair(workspace, primary)

    assert details["verified_seed_support_overrides_face_template"] is True
    assert np.array_equal(repaired[damage > 0], reference[damage > 0])
    assert np.all(provenance[damage > 0] == 1)
    assert np.array_equal(repaired[damage == 0], primary[damage == 0])
