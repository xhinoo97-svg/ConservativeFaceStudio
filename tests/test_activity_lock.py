from __future__ import annotations

import json
from pathlib import Path

import pytest

import app.activity as activity


def test_restoration_lock_blocks_parallel_work_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "runtime" / "restoration.lock"
    first = activity.RestorationActivityLock(path)
    first.acquire()
    try:
        assert activity.is_restoration_active(path) is True
        with pytest.raises(RuntimeError, match="già attiva"):
            activity.RestorationActivityLock(path).acquire()
    finally:
        first.release()
    assert activity.is_restoration_active(path) is False


def test_stale_restoration_lock_is_removed(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "restoration.lock"
    path.write_text(json.dumps({"pid": 99999999}), encoding="utf-8")
    monkeypatch.setattr(activity, "_pid_alive", lambda pid: False)

    assert activity.is_restoration_active(path) is False
    assert not path.exists()
