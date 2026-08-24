from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_packaged_process_defaults_to_dynamic_shape_auto_engine() -> None:
    env = dict(os.environ)
    env.pop("OPENCV_FORCE_DNN_ENGINE", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, app; print(os.environ['OPENCV_FORCE_DNN_ENGINE'])",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert completed.stdout.strip() == "3"


@pytest.mark.parametrize(
    "script",
    (
        "scripts/audit_product_modules.py",
        "scripts/check_for_updates.py",
        "scripts/generate_update_manifest.py",
        "scripts/generate_validation_summary.py",
        "scripts/prefetch_public_portraits.py",
        "scripts/run_face_anchored_practical_benchmark.py",
        "scripts/run_female_domain_benchmark_80.py",
        "scripts/run_female_domain_benchmark_observed.py",
        "scripts/smoke_production_models.py",
        "scripts/smoke_reference_counts.py",
    ),
)
def test_direct_smoke_scripts_bootstrap_repository_imports(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "usage:" in completed.stdout.lower()
