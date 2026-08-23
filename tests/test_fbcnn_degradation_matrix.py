from __future__ import annotations

import copy

import numpy as np

from research.fbcnn_degradation_matrix import (
    FBCNN_DEVELOPMENT_PROFILES,
    REQUIRED_FAMILIES,
    disposition_for_metrics,
    materialize_degradation,
    summarize_reports,
)


def _metrics(**overrides) -> dict:
    payload = {
        "identity_threshold": 0.363,
        "sface_clean_vs_degraded": 0.80,
        "sface_clean_vs_fbcnn": 0.82,
        "psnr_degraded": 30.0,
        "psnr_fbcnn": 31.0,
        "ssim_degraded": 0.90,
        "ssim_fbcnn": 0.92,
        "wrong_person_final_pixels": 0,
        "provenance_valid": True,
    }
    payload.update(overrides)
    return payload


def _report(profile_id: str, family: str) -> dict:
    metrics = _metrics()
    disposition = disposition_for_metrics(metrics)
    return {
        "source": {
            "degradation_profile": profile_id,
            "degradation_family": family,
            "input_sha256": "a" * 64,
        },
        "model": {
            "checkpoint_sha256_observed": "b" * 64,
            "official_source_commit": "c" * 40,
        },
        "metrics": metrics,
        "disposition": disposition,
        "outputs": {
            "restored_sha256": "d" * 64,
            "final_sha256": "e" * 64,
        },
    }


def test_development_profiles_cover_compression_block_mosquito_and_social_routes() -> None:
    ids = [item.profile_id for item in FBCNN_DEVELOPMENT_PROFILES]

    assert len(ids) == len(set(ids)) == 6
    assert REQUIRED_FAMILIES == {
        "block_artifacts",
        "jpeg_compression",
        "double_jpeg",
        "social_recompression",
        "mosquito_noise",
    }


def test_every_degradation_is_deterministic_and_preserves_shape_and_type() -> None:
    grid = np.indices((96, 112)).sum(axis=0).astype(np.uint8)
    clean = np.dstack((grid, np.flipud(grid), np.fliplr(grid)))

    for profile in FBCNN_DEVELOPMENT_PROFILES:
        first = materialize_degradation(clean, profile)
        second = materialize_degradation(clean, profile)
        assert first.shape == clean.shape
        assert first.dtype == np.uint8
        assert np.array_equal(first, second)
        assert not np.array_equal(first, clean)


def test_disposition_passes_only_when_identity_quality_and_provenance_all_pass() -> None:
    passed = disposition_for_metrics(_metrics())
    identity_rollback = disposition_for_metrics(_metrics(sface_clean_vs_fbcnn=0.30))
    quality_rollback = disposition_for_metrics(_metrics(psnr_fbcnn=29.0))
    provenance_rollback = disposition_for_metrics(_metrics(provenance_valid=False))

    assert passed["decision"] == "PASS"
    assert passed["restoration_pass"] is True
    assert identity_rollback["decision"] == "ROLLBACK"
    assert quality_rollback["decision"] == "ROLLBACK"
    assert provenance_rollback["decision"] == "ROLLBACK"


def test_matrix_summary_requires_all_frozen_profiles_and_preserves_rollbacks() -> None:
    reports = [_report(item.profile_id, item.family) for item in FBCNN_DEVELOPMENT_PROFILES]
    passed = summarize_reports(reports)

    assert passed["completed_cases"] == 6
    assert passed["restoration_pass_count"] == 6
    assert passed["rollback_count"] == 0
    assert passed["development_gate_pass"] is True
    assert passed["production_qualified"] is False

    failed_reports = copy.deepcopy(reports)
    failed_reports[0]["metrics"] = _metrics(ssim_fbcnn=0.88)
    failed_reports[0]["disposition"] = disposition_for_metrics(failed_reports[0]["metrics"])
    rolled_back = summarize_reports(failed_reports)

    assert rolled_back["restoration_pass_count"] == 5
    assert rolled_back["rollback_count"] == 1
    assert rolled_back["development_gate_pass"] is False
    assert rolled_back["family_status"][failed_reports[0]["source"]["degradation_family"]] == "NOT_QUALIFIED"
