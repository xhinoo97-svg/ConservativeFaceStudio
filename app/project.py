from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_VERSION = 1


@dataclass
class OperationRecord:
    block: str
    parameters: dict[str, Any] = field(default_factory=dict)
    input_sha256: str | None = None
    output_sha256: str | None = None
    conservative: bool = True
    model: str | None = None
    model_license: str | None = None
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ProjectDocument:
    name: str
    sources: list[str] = field(default_factory=list)
    accepted_blocks: list[str] = field(default_factory=list)
    skipped_blocks: list[str] = field(default_factory=list)
    operations: list[OperationRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: int = PROJECT_VERSION


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_project(project: ProjectDocument, path: str | Path) -> None:
    """Scrittura atomica: un crash non sostituisce il progetto valido con un file parziale."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(project)
    payload["version"] = PROJECT_VERSION
    fd, temp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def load_project(path: str | Path) -> ProjectDocument:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    version = int(payload.get("version", 0))
    if version != PROJECT_VERSION:
        raise ValueError(f"Versione progetto non supportata: {version}")
    operations = [OperationRecord(**item) for item in payload.get("operations", [])]
    return ProjectDocument(
        name=str(payload["name"]),
        sources=[str(item) for item in payload.get("sources", [])],
        accepted_blocks=[str(item) for item in payload.get("accepted_blocks", [])],
        skipped_blocks=[str(item) for item in payload.get("skipped_blocks", [])],
        operations=operations,
        metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
        version=version,
    )


def export_provenance(project: ProjectDocument, path: str | Path) -> None:
    """Esporta un report JSON separato, leggibile senza aprire il progetto."""
    save_project(project, path)
