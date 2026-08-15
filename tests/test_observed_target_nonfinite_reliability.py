from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np

from app.observed_target_repair_runtime import repair_observed_target


def test_equal_infinite_reliability_conflict_abstains_without_runtime_warning():
    shape = (32, 32)
    primary = np.full((*shape, 3), 90, dtype=np.uint8)
    damage = np.zeros(shape, dtype=np.uint8)
    damage[8:24, 8:24] = 255

    donor_a = primary.copy()
    donor_b = primary.copy()
    donor_a[damage > 0] = (10, 10, 10)
    donor_b[damage > 0] = (240, 240, 240)

    support = damage.copy()
    infinite = np.full(shape, np.inf, dtype=np.float32)
    workspace = SimpleNamespace(
        primary=primary,
        aligned_references=[donor_a, donor_b],
        occlusion_masks=[damage],
        metadata={
            "aligned_reference_identity_verified": [True, True],
            "aligned_reference_support_masks": [support, support],
            "aligned_reference_detail_reliability_maps": [infinite, infinite],
            "aligned_reference_original_source_indices": [1, 2],
            "primary_bbox": (0, 0, 32, 32),
        },
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, provenance, diagnostics = repair_observed_target(workspace, primary.copy())

    runtime_warnings = [item for item in caught if issubclass(item.category, RuntimeWarning)]
    assert runtime_warnings == []
    assert diagnostics["applied"] is False
    assert diagnostics["reason"] == "agreement_or_fraction_gate_rejected_target"
    assert diagnostics["damage_reference_coverage"] == 0.0
    assert np.count_nonzero(provenance) == 0
    assert np.array_equal(result, primary)


def test_positive_infinity_is_bounded_for_donor_ranking():
    shape = (32, 32)
    primary = np.full((*shape, 3), 80, dtype=np.uint8)
    damage = np.zeros(shape, dtype=np.uint8)
    damage[10:22, 10:22] = 255

    donor = primary.copy()
    donor[damage > 0] = (20, 30, 40)
    reliability = np.full(shape, np.nan, dtype=np.float32)
    reliability[damage > 0] = np.inf
    workspace = SimpleNamespace(
        primary=primary,
        aligned_references=[donor],
        occlusion_masks=[damage],
        metadata={
            "aligned_reference_identity_verified": [True],
            "aligned_reference_support_masks": [damage.copy()],
            "aligned_reference_detail_reliability_maps": [reliability],
            "aligned_reference_original_source_indices": [1],
            "primary_bbox": (0, 0, 32, 32),
        },
    )

    result, provenance, diagnostics = repair_observed_target(workspace, primary.copy())

    repaired = provenance > 0
    assert diagnostics["applied"] is True
    assert diagnostics["damage_reference_coverage"] == 1.0
    assert np.array_equal(result[repaired], donor[repaired])
    assert np.array_equal(result[~repaired], primary[~repaired])
