from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from app.paths import user_data_root


def restoration_lock_path() -> Path:
    return user_data_root().resolve() / "runtime" / "restoration-active.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except (OSError, PermissionError):
        # A permission failure means another process may own the PID: fail closed.
        return True


def is_restoration_active(path: str | Path | None = None) -> bool:
    lock = Path(path).resolve() if path is not None else restoration_lock_path()
    if not lock.is_file():
        return False
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
        pid = int(payload.get("pid", 0)) if isinstance(payload, dict) else 0
    except (OSError, ValueError, TypeError):
        return True
    if _pid_alive(pid):
        return True
    lock.unlink(missing_ok=True)
    return False


class RestorationActivityLock:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path).resolve() if path is not None else restoration_lock_path()
        self._owned = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if is_restoration_active(self.path):
            raise RuntimeError("Un'altra restoration è già attiva")
        payload = json.dumps({
            "pid": os.getpid(),
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }).encode("utf-8")
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise RuntimeError("Un'altra restoration è già attiva") from exc
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._owned = True

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def __enter__(self) -> "RestorationActivityLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
