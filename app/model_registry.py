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
        key="nafnet_gopro_width32",
        title="NAFNet GoPro width32 deblur",
        filename="NAFNet-GoPro-width32.pth",
        destination="models/nafnet/NAFNet-GoPro-width32.pth",
        source_url=None,
        code_license="NAFNet upstream license",
        weights_license="Use under upstream model terms",
        conservative_default=True,
        notes=(
            "CPU-first deblur candidate. Official repository reports 32.8705 dB / 0.9606 SSIM on GoPro "
            "and provides the pretrained checkpoint at Google Drive id 1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj. "
            "Keep the classical deblur fallback and identity rollback enabled."
        ),
    ),
    ModelManifest(
        key="nafnet_sidd_width32",
        title="NAFNet SIDD width32 denoise",
        filename="NAFNet-SIDD-width32.pth",
        destination="models/nafnet/NAFNet-SIDD-width32.pth",
        source_url=None,
        code_license="NAFNet upstream license",
        weights_license="Use under upstream model terms",
        conservative_default=True,
        notes=(
            "CPU-first real-image denoise candidate. Official repository reports 39.9672 dB / 0.9599 SSIM "
            "on SIDD and provides the checkpoint at Google Drive id 1lsByk21Xw-6aW7epCwOQxvm6HYCQZPHZ."
        ),
    ),
    ModelManifest(
        key="restormer_motion_deblur",
        title="Restormer motion deblurring",
        filename="motion_deblurring.pth",
        destination="models/restormer/motion_deblurring.pth",
        source_url="https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth",
        code_license="MIT",
        weights_license="Use under upstream release terms",
        conservative_default=False,
        notes="Official Restormer v1.0 checkpoint. Heavier quality alternative to NAFNet width32; use tiled inference and identity rollback.",
    ),
    ModelManifest(
        key="restormer_real_denoise",
        title="Restormer real-image denoising",
        filename="real_denoising.pth",
        destination="models/restormer/real_denoising.pth",
        source_url="https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth",
        code_license="MIT",
        weights_license="Use under upstream release terms",
        conservative_default=False,
        notes="Official Restormer v1.0 real-image denoising checkpoint. Prefer only when the lighter NAFNet path is unavailable or validation favours it.",
    ),
    ModelManifest(
        key="mirnet_fivek_enhance",
        title="MIRNet FiveK enhancement",
        filename="model_fivek.pth",
        destination="models/mirnet/model_fivek.pth",
        source_url=None,
        code_license="MIRNet upstream license",
        weights_license="Use under upstream model terms",
        conservative_default=False,
        notes=(
            "General photographic enhancement candidate. Official MIRNet repository publishes model_fivek.pth "
            "in its pretrained-model Google Drive folder. Apply only behind colour/identity validation."
        ),
    ),
    ModelManifest(
        key="zero_dce_plus",
        title="Zero-DCE++ low-light enhancement",
        filename="Epoch99.pth",
        destination="models/zero_dce_plus/Epoch99.pth",
        source_url="https://raw.githubusercontent.com/Li-Chongyi/Zero-DCE_extension/main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth",
        code_license="CC-BY-NC-4.0 upstream",
        weights_license="Non-commercial research terms upstream",
        conservative_default=False,
        max_bytes=5_000_000,
        notes="Very small pretrained checkpoint; use only when exposure analysis classifies the input as low-light. It is not a generic old-photo enhancer.",
    ),
    ModelManifest(
        key="mediapipe_face_landmarker",
        title="MediaPipe Face Landmarker float16",
        filename="face_landmarker.task",
        destination="models/landmarks/face_landmarker.task",
        source_url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        code_license="Apache-2.0",
        weights_license="Google MediaPipe model terms",
        conservative_default=True,
        max_bytes=100_000_000,
        notes="Official pretrained Face Landmarker bundle; preferred dense-landmark backend on CPU when installed.",
    ),
    ModelManifest(
        key="bisenet_face_parsing",
        title="BiSeNet CelebAMask-HQ face parsing",
        filename="79999_iter.pth",
        destination="models/face_parsing/79999_iter.pth",
        source_url=None,
        code_license="MIT",
        weights_license="Use under upstream pretrained-model terms",
        conservative_default=True,
        notes=(
            "Official zllrunning face-parsing.PyTorch pretrained checkpoint. Upstream download is Google Drive id "
            "154JgKpzCPW82qINcVieuPH3fZ2e0P812. Parsing is semantic support, not proof of occlusion; strict repair still requires multi-reference consensus."
        ),
    ),
    ModelManifest(
        key="dmdnet",
        title="DMDNet specific/generic blind face restoration",
        filename="DMDNet.pth",
        destination="models/dmdnet/DMDNet.pth",
        source_url="https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth",
        code_license="CC-BY-NC-SA-4.0 upstream",
        weights_license="Non-commercial research terms upstream",
        conservative_default=False,
        max_bytes=800_000_000,
        notes=(
            "Optional research backend using generic and same-identity specific memory dictionaries. "
            "Strict mode instead uses the independent observed-pixel specific-reference memory with exact provenance."
        ),
    ),
    ModelManifest(
        key="lama_big",
        title="LaMa big-lama inpainting",
        filename="big-lama",
        destination="models/lama/big-lama",
        source_url=None,
        code_license="Apache-2.0 upstream",
        weights_license="Use under upstream checkpoint terms",
        conservative_default=False,
        notes=(
            "Official LaMa instructions provide pretrained Places2/CelebA-HQ models. This is generative inpainting; "
            "use only outside strict mode when no real reference observes the damaged area."
        ),
    ),
    ModelManifest(
        key="3ddfa_mb1",
        title="3DDFA_V2 MobileNet v1 ONNX",
        filename="mb1_120x120.onnx",
        destination="models/3ddfa/mb1_120x120.onnx",
        source_url=None,
        code_license="MIT",
        weights_license="Use under upstream model terms",
        conservative_default=True,
        notes=(
            "Official pre-converted ONNX model. Upstream Google Drive id: 1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk. "
            "3DDFA_V2 documents ONNX Runtime CPU inference; use for pose/3D geometry, not texture synthesis."
        ),
    ),
    ModelManifest(
        key="insightface_identity",
        title="InsightFace buffalo_l identity/alignment pack",
        filename="buffalo_l",
        destination="models/models/buffalo_l",
        source_url=None,
        code_license="MIT",
        weights_license="Non-commercial research use for upstream pretrained model packs",
        conservative_default=True,
        notes=(
            "Preferred identity guardrail backend and secondary alignment backend. buffalo_l includes detection, "
            "recognition and 2D106/3D68 alignment models. Install through the official InsightFace model mechanism or place the pack locally."
        ),
    ),
    ModelManifest(
        key="realesrgan_x2plus",
        title="Real-ESRGAN x2plus",
        filename="RealESRGAN_x2plus.pth",
        destination="models/realesrgan/RealESRGAN_x2plus.pth",
        source_url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        code_license="BSD-3-Clause",
        weights_license="Use under upstream release terms",
        conservative_default=False,
        notes="Optional learned x2 super-resolution. Strict mode keeps Lanczos as the default because learned SR can synthesize texture.",
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


def inspect_model(manifest: ModelManifest, root: str | Path) -> dict[str, object]:
    """Restituisce uno stato locale senza modificare o scaricare nulla."""
    validate_manifest(manifest)
    root_path = Path(root).resolve()
    target = (root_path / manifest.destination).resolve()
    try:
        target.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("La destinazione esce dalla directory del progetto") from exc
    exists = target.exists()
    status: dict[str, object] = {
        "key": manifest.key,
        "path": str(target),
        "exists": exists,
        "is_directory": target.is_dir() if exists else False,
        "checksum_required": manifest.expected_sha256 is not None,
        "checksum_ok": None,
    }
    if exists and target.is_file():
        status["size_bytes"] = target.stat().st_size
        actual = sha256_path(target)
        status["sha256"] = actual
        if manifest.expected_sha256 is not None:
            status["checksum_ok"] = actual.lower() == manifest.expected_sha256.lower()
    return status


def download_model(
    manifest: ModelManifest,
    root: str | Path,
    *,
    accept_license: bool,
    timeout_seconds: int = 60,
    allow_unverified_download: bool = False,
) -> Path:
    """Download atomico con limiti e verifica SHA-256.

    Richiede accettazione esplicita dei termini del modello. Per ridurre il rischio
    supply-chain, il flusso UI standard rifiuta download automatici senza SHA-256
    atteso. ``allow_unverified_download`` resta disponibile solo per sviluppo.
    """
    validate_manifest(manifest)
    if not accept_license:
        raise PermissionError("È richiesta l'accettazione esplicita dei termini del modello")
    if manifest.source_url is None:
        raise DownloadError("Nessun URL diretto approvato per questo modello")
    if manifest.expected_sha256 is None and not allow_unverified_download:
        raise DownloadError("Download automatico bloccato: checksum SHA-256 non registrato")

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
