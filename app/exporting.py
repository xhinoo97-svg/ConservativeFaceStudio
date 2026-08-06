from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from app.project import ProjectDocument, sha256_file


def export_image_atomic(
    image: np.ndarray,
    destination: str | Path,
    *,
    project: ProjectDocument | None = None,
    png_compression: int = 3,
) -> tuple[Path, Path | None]:
    """Esporta PNG/JPEG senza lasciare file parziali e, se richiesto, un report sidecar."""
    if image is None or image.size == 0:
        raise ValueError("Immagine non valida")
    target = Path(destination)
    suffix = target.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Formato export supportato: PNG o JPEG")
    target.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_PNG_COMPRESSION, int(np.clip(png_compression, 0, 9))]
    if suffix in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError("Codifica immagine fallita")

    fd, temp_name = tempfile.mkstemp(prefix=target.stem, suffix=suffix, dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise

    sidecar: Path | None = None
    if project is not None:
        sidecar = target.with_suffix(target.suffix + ".provenance.json")
        payload = asdict(project)
        payload["exported_file"] = target.name
        payload["export_sha256"] = sha256_file(target)
        sidecar.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return target, sidecar
