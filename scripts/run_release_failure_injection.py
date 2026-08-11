from __future__ import annotations

"""Run the frozen release failure-injection matrix and emit auditable JSON evidence."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

SCENARIOS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("missing_or_corrupt_weight", ("tests/test_update_worker.py::test_update_check_repairs_missing_model_even_when_version_is_current",)),
    ("wrong_checksum", ("tests/test_release_failure_injection.py::test_wrong_model_checksum_keeps_previous_working_weight",)),
    ("model_smoke_failure", ("tests/test_update_manager.py::test_model_smoke_failure_never_replaces_working_model",)),
    ("model_update_rollback", ("tests/test_update_manager.py::test_pack_activation_failure_rolls_back_every_activated_model",)),
    ("no_network", ("tests/test_core_models.py::test_core_bootstrap_records_download_failures_without_raising",)),
    ("no_gpu", ("tests/test_hardware.py::test_hardware_profile_reports_safe_cpu_fallback",)),
    ("gpu_inference_failure", ("tests/test_hardware.py::test_cuda_driver_can_be_available_while_dnn_route_remains_cpu",)),
    ("cpu_fallback", ("tests/test_hardware.py::test_unstable_opencl_forces_cpu_fallback",)),
    ("wrong_reference", ("tests/test_reference_inpainting.py::test_identity_filter_rejects_wrong_reference",)),
    ("unreadable_reference", ("tests/test_release_failure_injection.py::test_unreadable_reference_is_rejected_without_pixels",)),
    ("zero_references", ("tests/test_case_aware_automatic.py::test_single_image_runner_executes_core_abstentions_instead_of_skips",)),
    ("nine_references", ("tests/test_nine_reference_support.py::test_ninth_reference_can_repair_pixels_and_keep_exact_provenance",)),
    ("tiny_main", ("tests/test_release_failure_injection.py::test_tiny_main_is_processed_or_rejected_explicitly",)),
    ("exif_orientation", ("tests/test_release_failure_injection.py::test_unicode_windows_style_path_and_exif_orientation_are_respected",)),
    ("unicode_windows_path", ("tests/test_release_failure_injection.py::test_unicode_windows_style_path_and_exif_orientation_are_respected",)),
    ("low_disk", ("tests/test_update_manager.py::test_state_commit_failure_rolls_back_activated_model",)),
    ("restoration_crash", ("tests/test_automatic_integrity_policy.py::test_mandatory_block_execution_error_cannot_be_hidden_as_success",)),
    ("stale_lock", ("tests/test_activity_lock.py::test_stale_restoration_lock_is_removed",)),
    ("restart_recovery", ("tests/test_recovery_checkpoints.py::test_block_checkpoint_is_persisted_atomically_for_crash_recovery",)),
)

REQUIRED_SCENARIOS = tuple(name for name, _ in SCENARIOS)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def run(output: Path, *, repository_root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for scenario, node_ids in SCENARIOS:
        started = time.perf_counter()
        process = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *node_ids],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        duration = time.perf_counter() - started
        passed = process.returncode == 0
        records.append(
            {
                "scenario": scenario,
                "node_ids": list(node_ids),
                "passed": passed,
                "exit_code": int(process.returncode),
                "duration_seconds": round(duration, 3),
                "failure_output": None if passed else (process.stdout + process.stderr)[-8000:],
            }
        )
        print(f"{scenario}: {'PASS' if passed else 'FAIL'} ({duration:.2f}s)", flush=True)

    failed = [record["scenario"] for record in records if record["passed"] is not True]
    payload: dict[str, object] = {
        "format": "ConservativeFaceStudio release failure injection",
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_head": os.environ.get("GITHUB_SHA"),
        "status": "PASS" if not failed else "FAIL",
        "required_scenarios": list(REQUIRED_SCENARIOS),
        "passed_count": len(records) - len(failed),
        "failed_count": len(failed),
        "failed_scenarios": failed,
        "scenarios": records,
    }
    _write_json_atomic(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="failure-injection-summary.json")
    args = parser.parse_args()
    payload = run(Path(args.output))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
