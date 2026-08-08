from __future__ import annotations

import cv2
import numpy as np

from app.execution import Workspace
from app.same_canvas_repair_runtime import exact_same_canvas_observed_repair


def _clean() -> np.ndarray:
    image = np.full((128, 128, 3), 30, dtype=np.uint8)
    cv2.ellipse(image, (64, 66), (42, 52), 0, 0, 360, (145, 170, 198), -1)
    cv2.circle(image, (50, 54), 4, (25, 25, 25), -1)
    cv2.circle(image, (78, 54), 4, (25, 25, 25), -1)
    cv2.line(image, (64, 60), (64, 78), (70, 80, 90), 2)
    return image


def _workspace(primary: np.ndarray, references: list[np.ndarray], supports: list[np.ndarray], damage: np.ndarray) -> Workspace:
    workspace = Workspace(primary=primary.copy(), references=[item.copy() for item in references])
    workspace.aligned_references = [item.copy() for item in references]
    workspace.metadata["primary_bbox"] = (22, 14, 84, 104)
    workspace.metadata["aligned_reference_source_indices"] = list(range(len(references)))
    workspace.metadata["aligned_reference_original_source_indices"] = [index + 1 for index in range(len(references))]
    workspace.metadata["aligned_reference_support_masks"] = [item.copy() for item in supports]
    workspace.metadata["preflight_original_occlusion_masks"] = [damage.copy(), *[np.zeros_like(damage) for _ in references]]
    workspace.metadata["verified_same_canvas_alignment"] = [
        {
            "runtime_reference_index": index,
            "method": "verified-same-canvas-observed",
            "action": "restored-exact-identity-transform",
        }
        for index in range(len(references))
    ]
    workspace.occlusion_masks = [damage.copy(), *[np.zeros_like(damage) for _ in references]]
    return workspace


def test_exact_full_reference_restores_seeded_sticker_without_interpolation() -> None:
    clean = _clean()
    damage = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.ellipse(damage, (64, 64), (18, 12), 0, 0, 360, 255, -1)
    primary = clean.copy()
    primary[damage > 0] = (18, 18, 18)
    support = np.full(damage.shape, 255, dtype=np.uint8)
    workspace = _workspace(primary, [clean], [support], damage)

    repaired, provenance, details = exact_same_canvas_observed_repair(workspace, primary)

    assert details["applied"] is True
    assert details["generated_pixels"] == 0
    assert np.array_equal(repaired[damage > 0], clean[damage > 0])
    assert np.all(provenance[damage > 0] == 1)
    assert np.array_equal(repaired[damage == 0], primary[damage == 0])


def test_complementary_partial_references_restore_only_observed_damage() -> None:
    clean = _clean()
    damage = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.rectangle(damage, (42, 50), (86, 78), 255, -1)
    primary = clean.copy()
    primary[damage > 0] = (10, 10, 10)

    left_support = np.zeros_like(damage)
    left_support[:, :68] = 255
    right_support = np.zeros_like(damage)
    right_support[:, 60:] = 255
    left = np.zeros_like(clean)
    right = np.zeros_like(clean)
    left[left_support > 0] = clean[left_support > 0]
    right[right_support > 0] = clean[right_support > 0]
    workspace = _workspace(primary, [left, right], [left_support, right_support], damage)

    repaired, provenance, details = exact_same_canvas_observed_repair(workspace, primary)

    assert details["applied"] is True
    assert np.array_equal(repaired[damage > 0], clean[damage > 0])
    assert np.all(provenance[damage > 0] > 0)
    assert set(np.unique(provenance[damage > 0])).issubset({1, 2})


def test_no_verified_same_canvas_reference_abstains() -> None:
    clean = _clean()
    damage = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.rectangle(damage, (50, 52), (78, 74), 255, -1)
    primary = clean.copy()
    primary[damage > 0] = 0
    workspace = Workspace(primary=primary.copy(), references=[clean.copy()])

    repaired, provenance, details = exact_same_canvas_observed_repair(workspace, primary)

    assert details["applied"] is False
    assert details["reason"] == "no_verified_same_canvas_reference"
    assert np.array_equal(repaired, primary)
    assert np.count_nonzero(provenance) == 0
