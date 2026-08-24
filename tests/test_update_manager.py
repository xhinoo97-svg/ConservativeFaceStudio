from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import app.update_manager as updates


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(key: str, data: bytes, destination: str) -> updates.ModelUpdateEntry:
    return updates.ModelUpdateEntry(
        key=key,
        version="2.0.0",
        url=f"https://updates.example.test/{key}.onnx",
        sha256=_digest(data),
        destination=destination,
        max_bytes=1024,
    )


def test_model_update_is_verified_smoked_and_keeps_previous(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "models" / "detection" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old-model")
    candidate = b"new-model"

    def download(url: str, destination: Path, **kwargs) -> None:
        destination.write_bytes(candidate)

    seen: list[tuple[bytes, str]] = []
    monkeypatch.setattr(updates, "_download_https", download)
    updater = updates.ModelUpdater(
        tmp_path,
        smoke_tests={"detector": lambda path: seen.append((path.read_bytes(), path.suffix))},
    )

    result = updater.install(_entry("detector", candidate, "models/detection/model.onnx"))

    assert result.installed == ("detector",)
    assert target.read_bytes() == candidate
    assert target.with_name("model.onnx.previous").read_bytes() == b"old-model"
    assert seen == [(candidate, ".onnx"), (candidate, ".onnx")]
    assert (tmp_path / "models" / "model-update-state.json").is_file()


def test_model_smoke_failure_never_replaces_working_model(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "models" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old-model")
    candidate = b"bad-model"
    monkeypatch.setattr(
        updates,
        "_download_https",
        lambda url, destination, **kwargs: destination.write_bytes(candidate),
    )

    def reject(path: Path) -> None:
        raise RuntimeError("inference failed")

    updater = updates.ModelUpdater(tmp_path, smoke_tests={"model": reject})
    with pytest.raises(RuntimeError, match="inference failed"):
        updater.install(_entry("model", candidate, "models/model.onnx"))

    assert target.read_bytes() == b"old-model"
    assert not target.with_name("model.onnx.previous").exists()


def test_pack_activation_failure_rolls_back_every_activated_model(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "models" / "first.onnx"
    second = tmp_path / "models" / "second.onnx"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"old-first")
    second.write_bytes(b"old-second")
    payloads = {"first": b"new-first", "second": b"new-second"}

    def download(url: str, destination: Path, **kwargs) -> None:
        key = "first" if "first" in url else "second"
        destination.write_bytes(payloads[key])

    second_calls = 0

    def smoke_second(path: Path) -> None:
        nonlocal second_calls
        second_calls += 1
        if second_calls == 2:
            raise RuntimeError("post-activation failure")

    monkeypatch.setattr(updates, "_download_https", download)
    updater = updates.ModelUpdater(
        tmp_path,
        smoke_tests={"first": lambda path: None, "second": smoke_second},
    )
    with pytest.raises(updates.UpdateError, match="previous model pack restored"):
        updater.install_pack(
            (
                _entry("first", payloads["first"], "models/first.onnx"),
                _entry("second", payloads["second"], "models/second.onnx"),
            )
        )

    assert first.read_bytes() == b"old-first"
    assert second.read_bytes() == b"old-second"


def test_update_is_blocked_during_restoration(tmp_path: Path) -> None:
    updater = updates.ModelUpdater(tmp_path, smoke_tests={}, restoration_active=lambda: True)
    with pytest.raises(updates.UpdateBusyError):
        updater.install_pack((_entry("x", b"x", "models/x.onnx"),))


def test_atomic_move_failure_restores_current_model(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "models" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"working")
    candidate = b"candidate"
    monkeypatch.setattr(
        updates,
        "_download_https",
        lambda url, destination, **kwargs: destination.write_bytes(candidate),
    )
    real_replace = updates.os.replace

    def fail_candidate_move(source, destination) -> None:
        if ".update-" in Path(source).name and Path(source).suffix == ".onnx":
            raise OSError("activation move failed")
        real_replace(source, destination)

    monkeypatch.setattr(updates.os, "replace", fail_candidate_move)
    updater = updates.ModelUpdater(tmp_path, smoke_tests={"model": lambda path: None})

    with pytest.raises(updates.UpdateError, match="previous model pack restored"):
        updater.install(_entry("model", candidate, "models/model.onnx"))

    assert target.read_bytes() == b"working"


def test_state_commit_failure_rolls_back_activated_model(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "models" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"working")
    candidate = b"candidate"
    monkeypatch.setattr(
        updates,
        "_download_https",
        lambda url, destination, **kwargs: destination.write_bytes(candidate),
    )
    monkeypatch.setattr(
        updates,
        "_write_json_atomic",
        lambda path, payload: (_ for _ in ()).throw(OSError("disk full")),
    )
    updater = updates.ModelUpdater(tmp_path, smoke_tests={"model": lambda path: None})

    with pytest.raises(updates.UpdateError, match="state commit failed"):
        updater.install(_entry("model", candidate, "models/model.onnx"))

    assert target.read_bytes() == b"working"


def test_app_update_is_staged_without_overwriting_running_app(monkeypatch, tmp_path: Path) -> None:
    payload = b"verified installer"
    monkeypatch.setattr(
        updates,
        "_download_https",
        lambda url, destination, **kwargs: destination.write_bytes(payload),
    )
    entry = updates.AppUpdateEntry(
        version="2.0.0",
        url="https://updates.example.test/ConservativeFaceStudio-Setup-x64.exe",
        sha256=_digest(payload),
        filename="ConservativeFaceStudio-Setup-x64.exe",
        max_bytes=1024,
    )

    result = updates.AppUpdater(tmp_path / "updates").stage(entry)

    staged = Path(result.staged_path or "")
    assert staged.read_bytes() == payload
    assert (staged.parent / "verified-update.json").is_file()
    assert result.kind == "app"


def test_version_comparison_is_numeric() -> None:
    assert updates.is_newer_version("1.10.0", "1.9.9") is True
    assert updates.is_newer_version("1.9.0", "1.10.0") is False


def test_app_updater_refuses_installer_tampered_after_staging(monkeypatch, tmp_path: Path) -> None:
    payload = b"verified installer"
    monkeypatch.setattr(
        updates,
        "_download_https",
        lambda url, destination, **kwargs: destination.write_bytes(payload),
    )
    entry = updates.AppUpdateEntry(
        version="2.0.0",
        url="https://updates.example.test/setup.exe",
        sha256=_digest(payload),
        filename="setup.exe",
        max_bytes=1024,
    )
    updater = updates.AppUpdater(tmp_path / "updates")
    staged = Path(updater.stage(entry).staged_path or "")
    staged.write_bytes(b"tampered")
    monkeypatch.setattr(updates.sys, "platform", "win32")

    with pytest.raises(updates.UpdateError, match="changed after verification"):
        updater.launch_installer(staged)
