from __future__ import annotations

from scripts.run_release_failure_injection import REQUIRED_SCENARIOS, SCENARIOS


def test_release_failure_injection_matrix_is_complete_and_unique() -> None:
    expected = {
        "missing_or_corrupt_weight",
        "wrong_checksum",
        "model_smoke_failure",
        "model_update_rollback",
        "no_network",
        "no_gpu",
        "gpu_inference_failure",
        "cpu_fallback",
        "wrong_reference",
        "unreadable_reference",
        "zero_references",
        "nine_references",
        "tiny_main",
        "exif_orientation",
        "unicode_windows_path",
        "low_disk",
        "restoration_crash",
        "stale_lock",
        "restart_recovery",
    }

    assert len(REQUIRED_SCENARIOS) == len(set(REQUIRED_SCENARIOS))
    assert set(REQUIRED_SCENARIOS) == expected
    assert all(node_ids for _, node_ids in SCENARIOS)
