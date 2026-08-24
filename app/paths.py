from __future__ import annotations

import os
import sys
from pathlib import Path


APP_DIR_NAME = "ConservativeFaceStudio"


def runtime_root() -> Path:
    """Directory containing the installed executable or repository root in development."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_root() -> Path:
    """Return a per-user writable root without requiring administrator privileges."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / APP_DIR_NAME


def models_root() -> Path:
    root = user_data_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def model_search_roots() -> tuple[Path, ...]:
    candidates = [models_root(), runtime_root(), Path.cwd().resolve()]
    unique: list[Path] = []
    for item in candidates:
        resolved = item.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)
