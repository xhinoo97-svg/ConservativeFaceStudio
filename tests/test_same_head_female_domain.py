from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.verify_same_head_female_domain as verifier


HEAD = "a" * 40


def _write_report(root: Path, *, source_head: str = HEAD, portraits: int = 60, errors: int = 0) -> Path:
    path = root / "female-domain-benchmark" / "female-domain-benchmark.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for index in range(portraits):
        base = {"portrait": f"p{index:03d}"}
        cases.extend(
            [
                {**base, "scenario": "gaussian_heavy_single" if index % 2 == 0 else "mosaic_single"},
                {**base, "scenario": "opaque_sticker_single"},
                {**base, "scenario": "opaque_sticker_full_reference"},
                {**base, "scenario": "scribble_two_partial"},
                {**base, "scenario": "component_only_references"},
            ]
        )
    payload = {
        "source_head": source_head,
        "portrait_count": portraits,
        "minimum_required_portraits": 60,
        "quick_profile_cases_per_portrait": 5,
        "summary": {"completed_cases": portraits * 5, "error_cases": errors},
        "cases": cases,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_same_head_female_domain_accepts_300_case_floor(tmp_path: Path, monkeypatch) -> None:
    report = _write_report(tmp_path)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    result = verifier.verify(report)
    assert result["portrait_count"] == 60
    assert result["completed_cases"] == 300
    assert set(result["scenario_coverage"]) == {"gaussian_heavy_single", "mosaic_single"}


def test_same_head_female_domain_rejects_wrong_sha(tmp_path: Path, monkeypatch) -> None:
    report = _write_report(tmp_path, source_head="b" * 40)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    with pytest.raises(RuntimeError, match="different Git SHA"):
        verifier.verify(report)


def test_same_head_female_domain_rejects_fewer_than_60_portraits(tmp_path: Path, monkeypatch) -> None:
    report = _write_report(tmp_path, portraits=59)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    with pytest.raises(RuntimeError, match="fewer than 60"):
        verifier.verify(report)


def test_same_head_female_domain_rejects_runtime_errors(tmp_path: Path, monkeypatch) -> None:
    report = _write_report(tmp_path, errors=1)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    with pytest.raises(RuntimeError, match="runtime error"):
        verifier.verify(report)


def test_same_head_female_domain_requires_both_blur_and_mosaic(tmp_path: Path, monkeypatch) -> None:
    report = _write_report(tmp_path)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    payload = json.loads(report.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        if case["scenario"] == "mosaic_single":
            case["scenario"] = "gaussian_heavy_single"
    report.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing severe blur/mosaic coverage"):
        verifier.verify(report)
