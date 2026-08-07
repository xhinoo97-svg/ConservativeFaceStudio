from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from app.project import ProjectDocument


@dataclass(frozen=True)
class BlockSnapshot:
    order: int
    block: str
    title: str
    filename: str
    sha256: str
    width: int
    height: int
    details: dict[str, Any]
    timestamp_utc: str


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class BlockArtifactArchive:
    """Conserva un PNG lossless dopo ogni blocco e crea un archivio ZIP verificabile."""

    def __init__(self) -> None:
        self._entries: list[tuple[BlockSnapshot, bytes]] = []

    @property
    def snapshots(self) -> tuple[BlockSnapshot, ...]:
        return tuple(item[0] for item in self._entries)

    def _encode(self, image: np.ndarray) -> bytes:
        if image is None or image.size == 0:
            raise ValueError("Immagine snapshot non valida")
        if image.ndim not in (2, 3):
            raise ValueError("Formato snapshot non supportato")
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("Impossibile codificare lo snapshot PNG")
        return encoded.tobytes()

    def record(self, block: str, title: str, image: np.ndarray, details: dict[str, Any] | None = None) -> BlockSnapshot:
        payload = self._encode(image)
        order = len(self._entries) + 1
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", block).strip("_") or "block"
        filename = f"{order:02d}_{safe}.png"
        height, width = image.shape[:2]
        snapshot = BlockSnapshot(order, block, title, filename, hashlib.sha256(payload).hexdigest(), int(width), int(height), dict(details or {}), datetime.now(timezone.utc).isoformat())
        self._entries.append((snapshot, payload))
        return snapshot

    def replace_last(self, image: np.ndarray, details: dict[str, Any]) -> BlockSnapshot:
        if not self._entries:
            raise RuntimeError("Nessuno snapshot da sostituire")
        previous, _ = self._entries[-1]
        payload = self._encode(image)
        h, w = image.shape[:2]
        replacement = BlockSnapshot(previous.order, previous.block, previous.title, previous.filename, hashlib.sha256(payload).hexdigest(), int(w), int(h), dict(details), datetime.now(timezone.utc).isoformat())
        self._entries[-1] = (replacement, payload)
        return replacement

    def export_zip(self, destination: str | Path, *, project: ProjectDocument | None = None, attachments: Iterable[str | Path] = ()) -> Path:
        target = Path(destination)
        if target.suffix.lower() != ".zip":
            target = target.with_suffix(target.suffix + ".zip" if target.suffix else ".zip")
        target.parent.mkdir(parents=True, exist_ok=True)
        attachment_paths: list[Path] = []
        seen: set[Path] = set()
        for item in attachments:
            path = Path(item)
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            attachment_paths.append(path)
        manifest: dict[str, Any] = {
            "format": "ConservativeFaceStudio block archive",
            "version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_count": len(self._entries),
            "snapshots": [asdict(item[0]) for item in self._entries],
            "attachments": [{"filename": p.name, "archive_path": f"results/{p.name}", "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size} for p in attachment_paths],
        }
        if project is not None:
            manifest["project"] = asdict(project)
        fd, temp_name = tempfile.mkstemp(prefix=target.name, suffix=".tmp", dir=target.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for snapshot, payload in self._entries:
                    archive.writestr(f"blocks/{snapshot.filename}", payload)
                for path in attachment_paths:
                    archive.write(path, arcname=f"results/{path.name}")
                archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default))
            with zipfile.ZipFile(temp_path, "r") as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise RuntimeError(f"Archivio ZIP corrotto: {bad}")
            os.replace(temp_path, target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return target
