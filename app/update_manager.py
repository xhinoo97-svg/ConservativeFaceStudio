from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping


class UpdateError(RuntimeError):
    pass


class UpdateBusyError(UpdateError):
    pass


SmokeTest = Callable[[Path], None]


@dataclass(frozen=True)
class ModelUpdateEntry:
    key: str
    version: str
    url: str
    sha256: str
    destination: str
    max_bytes: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ModelUpdateEntry":
        entry = cls(
            key=str(payload["key"]),
            version=str(payload["version"]),
            url=str(payload["url"]),
            sha256=str(payload["sha256"]).lower(),
            destination=str(payload["destination"]),
            max_bytes=int(payload["max_bytes"]),
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if not self.key or not self.version:
            raise UpdateError("Model key/version missing")
        _validate_https(self.url)
        _validate_sha256(self.sha256)
        destination = Path(self.destination)
        if destination.is_absolute() or ".." in destination.parts:
            raise UpdateError("Unsafe model destination")
        if self.max_bytes <= 0:
            raise UpdateError("Invalid model size limit")


@dataclass(frozen=True)
class AppUpdateEntry:
    version: str
    url: str
    sha256: str
    filename: str
    max_bytes: int

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AppUpdateEntry":
        entry = cls(
            version=str(payload["version"]),
            url=str(payload["url"]),
            sha256=str(payload["sha256"]).lower(),
            filename=str(payload["filename"]),
            max_bytes=int(payload["max_bytes"]),
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if not self.version:
            raise UpdateError("App version missing")
        _validate_https(self.url)
        _validate_sha256(self.sha256)
        filename = Path(self.filename)
        if filename.name != self.filename or filename.suffix.lower() != ".exe":
            raise UpdateError("App update must be one Windows installer filename")
        if self.max_bytes <= 0:
            raise UpdateError("Invalid app package size limit")


@dataclass(frozen=True)
class UpdateResult:
    kind: str
    version: str
    installed: tuple[str, ...]
    staged_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["installed"] = list(self.installed)
        return payload


def _validate_https(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise UpdateError("Only credential-free HTTPS update URLs are allowed")


def _validate_sha256(value: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise UpdateError("Invalid SHA-256")


def _safe_target(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise UpdateError("Update destination escapes product root") from exc
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _download_https(url: str, destination: Path, *, max_bytes: int, timeout_seconds: int) -> None:
    _validate_https(url)
    request = urllib.request.Request(url, headers={"User-Agent": "ConservativeFaceStudio-Updater/1.0"})
    written = 0
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as output:
        final_url = response.geturl()
        _validate_https(final_url)
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise UpdateError("Update exceeds registered size limit")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                raise UpdateError("Update exceeded registered size limit")
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if written == 0:
        raise UpdateError("Downloaded update is empty")


def fetch_update_manifest(url: str, *, timeout_seconds: int = 20, max_bytes: int = 1_000_000) -> dict[str, object]:
    """Fetch a small HTTPS manifest without changing the installation."""
    fd, name = tempfile.mkstemp(prefix="cfs-update-manifest-", suffix=".json")
    os.close(fd)
    path = Path(name)
    try:
        _download_https(url, path, max_bytes=max_bytes, timeout_seconds=timeout_seconds)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise UpdateError("Update manifest must be a JSON object")
        return payload
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise UpdateError(f"Invalid update manifest: {exc}") from exc
    finally:
        path.unlink(missing_ok=True)


def _version_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple((0, int(item)) if item.isdigit() else (1, item.lower()) for item in re.findall(r"\d+|[A-Za-z]+", value))


def is_newer_version(candidate: str, current: str) -> bool:
    return _version_key(candidate) > _version_key(current)


class ModelUpdater:
    """Stage, verify, smoke and atomically activate one model or a model pack."""

    def __init__(
        self,
        product_root: str | Path,
        *,
        smoke_tests: Mapping[str, SmokeTest],
        restoration_active: Callable[[], bool] | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self.root = Path(product_root).resolve()
        self.smoke_tests = dict(smoke_tests)
        self.restoration_active = restoration_active or (lambda: False)
        self.state_path = Path(state_path).resolve() if state_path else self.root / "models" / "model-update-state.json"

    def install(self, entry: ModelUpdateEntry, *, timeout_seconds: int = 120) -> UpdateResult:
        return self.install_pack((entry,), timeout_seconds=timeout_seconds)

    def install_pack(
        self,
        entries: Iterable[ModelUpdateEntry],
        *,
        timeout_seconds: int = 120,
    ) -> UpdateResult:
        if self.restoration_active():
            raise UpdateBusyError("A restoration is active")
        requested = tuple(entries)
        if not requested:
            raise UpdateError("Empty model update pack")
        if len({entry.key for entry in requested}) != len(requested):
            raise UpdateError("Duplicate model key in update pack")

        staged: dict[str, Path] = {}
        targets: dict[str, Path] = {}
        try:
            for entry in requested:
                entry.validate()
                smoke = self.smoke_tests.get(entry.key)
                if smoke is None:
                    raise UpdateError(f"No real smoke test registered for {entry.key}")
                target = _safe_target(self.root, entry.destination)
                target.parent.mkdir(parents=True, exist_ok=True)
                # OpenCV's generic face-model loader detects ONNX from the final
                # extension.  Keep the registered suffix while still using a unique
                # hidden staging name; a trailing `.update` makes real YuNet/SFace
                # smoke inference fail before activation.
                fd, name = tempfile.mkstemp(
                    prefix=f".{target.stem}.update-",
                    suffix=target.suffix,
                    dir=target.parent,
                )
                os.close(fd)
                temporary = Path(name)
                staged[entry.key] = temporary
                targets[entry.key] = target
                _download_https(entry.url, temporary, max_bytes=entry.max_bytes, timeout_seconds=timeout_seconds)
                if _sha256(temporary).lower() != entry.sha256:
                    raise UpdateError(f"Checksum failed for {entry.key}")
                smoke(temporary)

            activated: list[tuple[ModelUpdateEntry, Path, Path, bool]] = []
            try:
                for entry in requested:
                    target = targets[entry.key]
                    previous = target.with_name(target.name + ".previous")
                    previous.unlink(missing_ok=True)
                    had_previous = target.is_file()
                    if had_previous:
                        os.replace(target, previous)
                    try:
                        os.replace(staged[entry.key], target)
                    except Exception:
                        if had_previous and previous.is_file():
                            os.replace(previous, target)
                        raise
                    activated.append((entry, target, previous, had_previous))

                for entry, target, _, _ in activated:
                    self.smoke_tests[entry.key](target)
            except Exception as exc:
                for _, target, previous, had_previous in reversed(activated):
                    target.unlink(missing_ok=True)
                    if had_previous and previous.is_file():
                        os.replace(previous, target)
                raise UpdateError(f"Activation failed; previous model pack restored: {exc}") from exc

            installed_state = {
                entry.key: {
                    "version": entry.version,
                    "destination": entry.destination,
                    "sha256": entry.sha256,
                    "status": "ACTIVE_VERIFIED",
                }
                for entry in requested
            }
            if self.state_path.is_file():
                try:
                    previous_state = json.loads(self.state_path.read_text(encoding="utf-8"))
                    previous_models = previous_state.get("models", {}) if isinstance(previous_state, dict) else {}
                    if isinstance(previous_models, dict):
                        installed_state = {**previous_models, **installed_state}
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
            try:
                _write_json_atomic(
                    self.state_path,
                    {"format": "ConservativeFaceStudio verified model updates", "version": 1, "models": installed_state},
                )
            except Exception as exc:
                for _, target, previous, had_previous in reversed(activated):
                    target.unlink(missing_ok=True)
                    if had_previous and previous.is_file():
                        os.replace(previous, target)
                raise UpdateError(f"Update state commit failed; previous model pack restored: {exc}") from exc
            version = requested[0].version if len(requested) == 1 else "pack"
            return UpdateResult("model", version, tuple(entry.key for entry in requested))
        finally:
            for temporary in staged.values():
                temporary.unlink(missing_ok=True)


class AppUpdater:
    """Verify an installer beside the active app; never overwrite a running executable."""

    def __init__(
        self,
        staging_root: str | Path,
        *,
        restoration_active: Callable[[], bool] | None = None,
    ) -> None:
        self.staging_root = Path(staging_root).resolve()
        self.restoration_active = restoration_active or (lambda: False)

    def stage(self, entry: AppUpdateEntry, *, timeout_seconds: int = 180) -> UpdateResult:
        if self.restoration_active():
            raise UpdateBusyError("A restoration is active")
        entry.validate()
        directory = _safe_target(self.staging_root, entry.version)
        directory.mkdir(parents=True, exist_ok=True)
        target = _safe_target(directory, entry.filename)
        fd, name = tempfile.mkstemp(prefix=entry.filename, suffix=".update", dir=directory)
        os.close(fd)
        temporary = Path(name)
        try:
            _download_https(entry.url, temporary, max_bytes=entry.max_bytes, timeout_seconds=timeout_seconds)
            if _sha256(temporary).lower() != entry.sha256:
                raise UpdateError("App update checksum failed")
            os.replace(temporary, target)
            _write_json_atomic(
                directory / "verified-update.json",
                {"status": "STAGED_VERIFIED", "version": entry.version, "sha256": entry.sha256, "package": target.name},
            )
            return UpdateResult("app", entry.version, (), str(target))
        finally:
            temporary.unlink(missing_ok=True)

    def launch_installer(self, verified_package: str | Path, *, silent: bool = False) -> subprocess.Popen[bytes]:
        if self.restoration_active():
            raise UpdateBusyError("A restoration is active")
        package = Path(verified_package).resolve()
        try:
            package.relative_to(self.staging_root)
        except ValueError as exc:
            raise UpdateError("Installer is outside the verified update staging root") from exc
        if not package.is_file() or package.suffix.lower() != ".exe":
            raise UpdateError("Verified Windows installer is missing")
        verification = package.parent / "verified-update.json"
        try:
            state = json.loads(verification.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise UpdateError("Installer verification record is missing or invalid") from exc
        if not isinstance(state, dict):
            raise UpdateError("Installer verification record is missing or invalid")
        expected = str(state.get("sha256") or "").lower()
        if state.get("status") != "STAGED_VERIFIED" or state.get("package") != package.name:
            raise UpdateError("Installer verification record does not match the staged package")
        if len(expected) != 64 or _sha256(package).lower() != expected:
            raise UpdateError("Staged installer changed after verification")
        if not sys.platform.startswith("win"):
            raise UpdateError("App installer launch is supported only on Windows")
        command = [str(package)]
        if silent:
            command.extend(["/VERYSILENT", "/NORESTART"])
        return subprocess.Popen(command, creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
