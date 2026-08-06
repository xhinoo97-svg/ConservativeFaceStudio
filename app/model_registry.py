from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ModelManifest:
    key: str
    title: str
    filename: str
    destination: str
    source_url: str | None
    code_license: str
    weights_license: str
    conservative_default: bool
    expected_sha256: str | None = None
    max_bytes: int = 2_000_000_000
    notes: str = ""


OFFICIAL_MODELS: tuple[ModelManifest, ...] = (
    ModelManifest(
        key="realesrgan_x2plus",
        title="Real-ESRGAN x2plus",
        filename="RealESRGAN_x2plus.pth",
        destination="models/realesrgan/RealESRGAN_x2plus.pth",
        source_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/RealESRGAN_x2plus.pth",
        code_license="BSD-3-Clause",
        weights_license="Verify upstream release terms before redistribution",
        conservative_default=False,
        notes="Generative super-resolution; strict mode must keep it disabled by default.",
    ),
    ModelManifest(
        key="3ddfa_mb1",
        title="3DDFA_V2 MobileNet v1 ONNX",
        filename="mb1_120x120.onnx",
        destination="models/3ddfa/mb1_120x120.onnx",
        source_url=None,
        code_license="MIT",
        weights_license="Verify upstream repository terms before redistribution",
        conservative_default=True,
        notes="Manual acquisition only until an official stable release URL and checksum are recorded.",
    ),
    ModelManifest(
        key="mediapipe_face_landmarker",
        title="MediaPipe Face Landmarker",
        filename="face_landmarker.task",
        destination="models/landmarks/face_landmarker.task",
        source_url=None,
        code_license="Apache-2.0",
        weights_license="Verify model card terms before redistribution",
        conservative_default=True,
        notes="Model bundle must be supplied by the user or downloaded from an approved official source.",
    ),
    ModelManifest(
        key="insightface_identity",
        title="InsightFace identity model",
        filename="model.onnx",
        destination="models/insightface/model.onnx",
        source_url=None,
        code_license="MIT",
        weights_license="Separate license required for many pretrained recognition models",
        conservative_default=True,
        notes="Never auto-download or redistribute recognition weights without explicit licensing.",
    ),
)


class DownloadError(RuntimeError):
    pass


def registry_by_key() -> dict[str, ModelManifest]:
    return {item.key: item for item in OFFICIAL_MODELS}


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: ModelManifest) -> None:
    destination = Path(manifest.destination)
    if destination.is_absolute() or ".." in destination.parts:
        raise ValueError(f"Destination non sicura: {manifest.destination}")
    if manifest.max_bytes <= 0:
        raise ValueError("max_bytes deve essere positivo")
    if manifest.expected_sha256 is not None:
        value = manifest.expected_sha256.lower()
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("Checksum SHA-256 non valido")
    if manifest.source_url is not None:
        parsed = urllib.parse.urlparse(manifest.source_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Sono consentiti solo URL HTTPS")


def download_model(
    manifest: ModelManifest,
    root: str | Path,
    *,
    accept_license: bool,
    timeout_seconds: int = 60,
) -> Path:
    """Download atomico con limite dimensione e verifica checksum.

    La chiamata richiede un'accettazione esplicita della licenza. I manifest senza
    URL ufficiale non vengono scaricati automaticamente.
    """
    validate_manifest(manifest)
    if not accept_license:
        raise PermissionError("È richiesta l'accettazione esplicita della licenza del modello")
    if manifest.source_url is None:
        raise DownloadError("Nessun URL ufficiale approvato per questo modello")

    root_path = Path(root).resolve()
    target = (root_path / manifest.destination).resolve()
    try:
        target.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("La destinazione esce dalla directory del progetto") from exc
    target.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        manifest.source_url,
        headers={"User-Agent": "ConservativeFaceStudio/1.0"},
    )
    fd, temp_name = tempfile.mkstemp(prefix=target.name, suffix=".download", dir=target.parent)
    written = 0
    digest = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output, urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > manifest.max_bytes:
                raise DownloadError("Il file supera il limite massimo consentito")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > manifest.max_bytes:
                    raise DownloadError("Il download ha superato il limite massimo consentito")
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())

        if written == 0:
            raise DownloadError("Il file scaricato è vuoto")
        actual = digest.hexdigest()
        if manifest.expected_sha256 and actual.lower() != manifest.expected_sha256.lower():
            raise DownloadError("Checksum SHA-256 non corrispondente")
        os.replace(temp_name, target)
        return target
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def export_registry(path: str | Path, manifests: Iterable[ModelManifest] = OFFICIAL_MODELS) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in manifests]
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
