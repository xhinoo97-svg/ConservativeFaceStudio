from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from app.case_aware_runtime import _same_canvas_partial_verification, _supplement_same_canvas_partials


def _workspace(*, support_inside_damage: bool) -> tuple[SimpleNamespace, np.ndarray]:
    clean = np.full((96, 96, 3), 36, dtype=np.uint8)
    cv2.ellipse(clean, (48, 50), (30, 38), 0, 0, 360, (145, 170, 198), -1)
    cv2.circle(clean, (37, 42), 4, (4, 4, 4), -1)
    cv2.circle(clean, (59, 42), 4, (4, 4, 4), -1)

    damage = np.zeros(clean.shape[:2], dtype=np.uint8)
    cv2.rectangle(damage, (24, 28), (72, 64), 255, -1)
    support = np.zeros_like(damage)
    if support_inside_damage:
        cv2.rectangle(support, (30, 34), (66, 58), 255, -1)
    else:
        cv2.rectangle(support, (6, 72), (42, 90), 255, -1)

    primary = clean.copy()
    primary[damage > 0] = (18, 18, 18)
    reference = np.zeros_like(clean)
    reference[support > 0] = clean[support > 0]
    reliability = np.full(clean.shape[:2], 255, dtype=np.uint8)
    zero = np.zeros_like(damage)

    workspace = SimpleNamespace(
        primary=primary,
        references=[reference],
        aligned_references=[],
        occlusion_masks=[damage.copy(), zero.copy()],
        metadata={
            "preflight_original_occlusion_masks": [damage.copy(), zero.copy()],
            "preflight_detail_reliability_maps": [reliability.copy(), reliability.copy()],
            "detail_reliability_threshold": 40,
            "runtime_source_order": [0, 1],
        },
    )
    return workspace, support


def test_damage_only_same_canvas_sheet_recovers_exact_geometry() -> None:
    workspace, support = _workspace(support_inside_damage=True)

    verified = _same_canvas_partial_verification(workspace, workspace.references[0], 0)

    assert verified is not None
    observed, reliability, details = verified
    assert details["method"] == "verified-same-canvas-partial"
    assert details["verification_basis"] == "seed-only-coordinate-preserving-damage-overlap"
    assert details["identity_status"] == "not_enough_evidence"
    assert details["may_expand_damage_seed"] is False
    assert details["damage_overlap_fraction"] >= 0.95
    assert np.array_equal(observed > 0, support > 0)
    assert np.all(reliability[observed == 0] == 0)


def test_sparse_sheet_outside_damage_is_not_promoted() -> None:
    workspace, _ = _workspace(support_inside_damage=False)
    assert _same_canvas_partial_verification(workspace, workspace.references[0], 0) is None


def test_global_abstention_can_still_recover_local_reference() -> None:
    workspace, support = _workspace(support_inside_damage=True)

    diagnostics = _supplement_same_canvas_partials(workspace)

    assert len(diagnostics) == 1
    assert len(workspace.aligned_references) == 1
    assert np.array_equal(workspace.aligned_references[0], workspace.references[0])
    assert np.array_equal(workspace.metadata["aligned_reference_support_masks"][0] > 0, support > 0)
    assert workspace.metadata["aligned_reference_partial_geometry_verified"] == [True]
