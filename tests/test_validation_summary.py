from __future__ import annotations

import json
from pathlib import Path

import scripts.generate_validation_summary as summary


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_validation_summary_requires_complete_installed_product(monkeypatch, tmp_path: Path) -> None:
    installer = tmp_path / "Setup.exe"
    portable = tmp_path / "Portable.zip"
    installer.write_bytes(b"installer")
    portable.write_bytes(b"portable")
    _write(tmp_path / "validation-report.json", {"passed": True, "failed_cases": []})
    _write(tmp_path / "benchmark-windows-ci.json", {"total_ms": 1.0})
    _write(tmp_path / "reference-count-smoke.json", {"status": "PASS", "counts": list(range(10))})
    checks = [
        {"argument": argument, "exit_code": 0, "passed": True}
        for argument in ("--smoke-test", "--verify-installation", "--offline-test")
    ]
    _write(tmp_path / "portable-validation.json", {"passed": True, "checks": checks})
    _write(tmp_path / "installed-app-validation.json", {"passed": True, "checks": checks})
    monkeypatch.setattr(summary, "PRODUCTION_MODEL_KEYS", ("one", "two"))
    _write(tmp_path / "models/model-registry.json", {"models": [
        {"key": "one", "status": "ACTIVE", "installed": True, "checksum_ok": True},
        {"key": "two", "status": "FALLBACK", "installed": True, "checksum_ok": True},
    ]})
    _write(tmp_path / "practical-benchmark/practical-benchmark.json", {"cases": [
        {"target95_applicable": True, "target95_passed": False, "error": None}
    ]})
    _write(tmp_path / "practical-matrix/practical-matrix.json", {"cases": [
        {"target95_applicable": False, "target95_passed": False, "error": None}
    ]})

    result = summary.generate_summary(tmp_path, "a" * 40, installer, portable)

    assert result["product_complete_pre_tuning"] is True
    assert result["target95_policy"] == "REPORT_ONLY_UNTIL_TUNING"
    assert result["gates"]["practical_runtime"]["runtime_errors"] == 0
    assert result["gates"]["practical_runtime"]["target95_pass"] == 0
    assert result["gates"]["installed_app"] == "PASS"
