from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.verify_same_head_windows_product as verifier


HEAD = "a" * 40


def _write_summary(root: Path, *, source_head: str = HEAD) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_head": source_head,
        "product_complete_pre_tuning": True,
        "gates": {
            "portable": "PASS",
            "offline": "PASS",
            "installer": "PASS",
            "installed_app": "PASS",
        },
        "artifacts": {
            "ConservativeFaceStudio-Setup-x64.exe": {
                "size_bytes": 2 * 1024 * 1024,
                "sha256": "1" * 64,
            },
            "ConservativeFaceStudio-Windows-x64.zip": {
                "size_bytes": 3 * 1024 * 1024,
                "sha256": "2" * 64,
            },
        },
    }
    (root / "validation-summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_same_head_windows_product_accepts_complete_exact_sha_metadata(tmp_path: Path, monkeypatch) -> None:
    _write_summary(tmp_path)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    result = verifier.verify(tmp_path)
    assert result["source_head"] == HEAD
    assert set(result["artifacts"]) == verifier.EXPECTED_ARTIFACTS


def test_same_head_windows_product_rejects_different_sha(tmp_path: Path, monkeypatch) -> None:
    _write_summary(tmp_path, source_head="b" * 40)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    with pytest.raises(RuntimeError, match="different Git SHA"):
        verifier.verify(tmp_path)


def test_same_head_windows_product_rejects_missing_product_gate(tmp_path: Path, monkeypatch) -> None:
    _write_summary(tmp_path)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    payload = json.loads((tmp_path / "validation-summary.json").read_text(encoding="utf-8"))
    payload["gates"]["offline"] = "FAIL"
    (tmp_path / "validation-summary.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="product gates not PASS"):
        verifier.verify(tmp_path)


def test_same_head_windows_product_rejects_missing_installer_record(tmp_path: Path, monkeypatch) -> None:
    _write_summary(tmp_path)
    monkeypatch.setattr(verifier, "_head", lambda: HEAD)
    payload = json.loads((tmp_path / "validation-summary.json").read_text(encoding="utf-8"))
    payload["artifacts"].pop("ConservativeFaceStudio-Setup-x64.exe")
    (tmp_path / "validation-summary.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="installer/portable artifacts are missing"):
        verifier.verify(tmp_path)
