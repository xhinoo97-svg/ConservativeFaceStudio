from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from scripts.run_one_shot_protocol_dev import DEV_CASE_IDS, main, run_dev


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dev_one_shot_writes_marker_immediately_before_first_case_access(tmp_path: Path) -> None:
    output = tmp_path / "success"

    result = run_dev(output)

    marker = _read_json(output / "CONSUMED.json")
    events = _read_json(output / "events.json")
    marker_index = events.index("marker_started")
    assert result["state"] == "CONSUMED_PASS"
    assert marker["state"] == "CONSUMED_PASS"
    assert events[marker_index + 1] == f"case_access:{DEV_CASE_IDS[0]}"
    assert [item["case_id"] for item in result["execution"]["completed"]] == list(DEV_CASE_IDS)
    archive = output / result["artifact"]["path"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == result["artifact"]["sha256"]
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist()) == {"CONSUMED.json", "events.json", "evidence.json"}


def test_failure_before_marker_is_recoverable_and_never_accesses_a_case(tmp_path: Path) -> None:
    output = tmp_path / "before"

    result = run_dev(output, inject_failure="before_marker")

    assert result["state"] == "PRECONSUMPTION_FAIL"
    assert result["marker_written"] is False
    assert not (output / "CONSUMED.json").exists()
    assert not any(event.startswith("case_access:") for event in _read_json(output / "events.json"))
    assert (output / "protocol-hardening-evidence.zip").is_file()


def test_failure_after_marker_is_consumed_and_never_accesses_a_case(tmp_path: Path) -> None:
    output = tmp_path / "after"

    result = run_dev(output, inject_failure="after_marker")

    assert result["state"] == "CONSUMED_FAIL"
    assert result["marker_written"] is True
    assert _read_json(output / "CONSUMED.json")["state"] == "CONSUMED_FAIL"
    assert not any(event.startswith("case_access:") for event in _read_json(output / "events.json"))
    assert (output / "protocol-hardening-evidence.zip").is_file()


def test_dev_runner_refuses_to_overwrite_existing_evidence(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "evidence.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Refusing to reuse"):
        run_dev(output)


def test_cli_uses_same_dev_entrypoint_and_reports_terminal_exit_codes(tmp_path: Path) -> None:
    assert main(["--output", str(tmp_path / "success")]) == 0
    assert main(["--output", str(tmp_path / "before"), "--inject-failure", "before_marker"]) == 2
    assert main(["--output", str(tmp_path / "after"), "--inject-failure", "after_marker"]) == 2
