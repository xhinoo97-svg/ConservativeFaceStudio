from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.face_domain_guard_v2_policy import (
    IDENTITY_ACCEPTED,
    IDENTITY_REJECTED,
    PARTIAL_IDENTITY_UNKNOWN,
    _filter_aligned_references,
    _identity_eligibility_by_source,
)
from app.observed_target_repair_runtime import _target_mask


def _workspace(statuses: list[str]):
    shape = (32, 32)
    refs = [np.full((32, 32, 3), index + 1, dtype=np.uint8) for index in range(len(statuses))]
    candidates = [
        {
            "source_index": index + 1,
            "accepted_identity": status == IDENTITY_ACCEPTED,
            "identity_eligibility": status,
            "identity_embedding_available": status != PARTIAL_IDENTITY_UNKNOWN,
        }
        for index, status in enumerate(statuses)
    ]
    metadata = {
        "preflight_candidates": candidates,
        "aligned_reference_source_indices": list(range(len(refs))),
        "aligned_reference_original_source_indices": list(range(1, len(refs) + 1)),
        "aligned_reference_support_masks": [np.full(shape, 255, dtype=np.uint8) for _ in refs],
        "aligned_reference_detail_reliability_maps": [np.full(shape, 255, dtype=np.uint8) for _ in refs],
        "aligned_reference_identity_scores": [None for _ in refs],
        "aligned_reference_identity_verified": [False for _ in refs],
        "aligned_reference_partial_geometry_verified": [True for _ in refs],
        "component_reference_bank": {
            "left_eye": [{"source_index": index + 1, "coverage": 1.0} for index in range(len(refs))],
            "mouth": [{"source_index": index + 1, "coverage": 1.0} for index in range(len(refs))],
        },
    }
    return SimpleNamespace(
        primary=np.full((32, 32, 3), 90, dtype=np.uint8),
        aligned_references=refs,
        occlusion_masks=[np.zeros(shape, dtype=np.uint8)],
        metadata=metadata,
    )


def test_geometrically_alignable_full_wrong_person_cannot_regain_donor_eligibility() -> None:
    workspace = _workspace([IDENTITY_ACCEPTED, IDENTITY_REJECTED])
    primary_before = workspace.primary.copy()

    rejected = _filter_aligned_references(workspace)

    assert rejected == [2]
    assert len(workspace.aligned_references) == 1
    assert workspace.metadata["aligned_reference_original_source_indices"] == [1]
    assert workspace.metadata["component_reference_bank"]["left_eye"] == [{"source_index": 1, "coverage": 1.0}]
    assert np.array_equal(workspace.primary, primary_before)


def test_legitimate_eye_only_reference_remains_usable_via_existing_component_path() -> None:
    workspace = _workspace([PARTIAL_IDENTITY_UNKNOWN])
    workspace.metadata["component_reference_bank"] = {
        "left_eye": [{"source_index": 1, "coverage": 0.92}],
        "mouth": [],
    }

    rejected = _filter_aligned_references(workspace)

    assert rejected == []
    assert len(workspace.aligned_references) == 1
    assert workspace.metadata["aligned_reference_partial_geometry_verified"] == [True]
    assert workspace.metadata["component_reference_bank"]["left_eye"] == [{"source_index": 1, "coverage": 0.92}]


def test_legitimate_mouth_only_reference_remains_usable_via_existing_component_path() -> None:
    workspace = _workspace([PARTIAL_IDENTITY_UNKNOWN])
    workspace.metadata["component_reference_bank"] = {
        "left_eye": [],
        "mouth": [{"source_index": 1, "coverage": 0.88}],
    }

    rejected = _filter_aligned_references(workspace)

    assert rejected == []
    assert len(workspace.aligned_references) == 1
    assert workspace.metadata["aligned_reference_partial_geometry_verified"] == [True]
    assert workspace.metadata["component_reference_bank"]["mouth"] == [{"source_index": 1, "coverage": 0.88}]


def test_same_person_full_reference_remains_eligible() -> None:
    workspace = _workspace([IDENTITY_ACCEPTED])
    assert _identity_eligibility_by_source(workspace) == {1: IDENTITY_ACCEPTED}
    assert _filter_aligned_references(workspace) == []
    assert len(workspace.aligned_references) == 1


def test_zero_references_remains_supported() -> None:
    workspace = _workspace([])
    assert _filter_aligned_references(workspace) == []
    assert workspace.aligned_references == []


def test_nine_references_remain_supported_when_not_explicitly_rejected() -> None:
    statuses = [IDENTITY_ACCEPTED] + [PARTIAL_IDENTITY_UNKNOWN] * 8
    workspace = _workspace(statuses)
    assert _filter_aligned_references(workspace) == []
    assert len(workspace.aligned_references) == 9


def test_refined_inpaint_target_is_authoritative_over_broad_raw_preflight_seed() -> None:
    workspace = _workspace([])
    broad = np.zeros((32, 32), dtype=np.uint8)
    broad[4:28, 4:28] = 255
    refined = np.zeros((32, 32), dtype=np.uint8)
    refined[10:18, 12:20] = 255
    workspace.metadata["preflight_original_occlusion_masks"] = [broad]
    workspace.occlusion_masks = [broad.copy()]
    workspace.metadata["inpaint_target_mask"] = refined.copy()

    target = _target_mask(workspace, (32, 32))

    assert np.array_equal(target, refined)
    assert np.count_nonzero(target) < np.count_nonzero(broad)
