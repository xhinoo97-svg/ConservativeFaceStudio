from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from app.face_restorer_adapter import (
    GENERATED_MODEL_INFERRED,
    RestorationCandidate,
    RestorationContext,
)

OFFICIAL_REPOSITORY = "jiaxi-jiang/FBCNN"
PINNED_REVISION = "54d1831927506b3247e2d4d245abb4f4dab1a1cd"
APPROVED_CHECKPOINT_FILENAME = "fbcnn_color.pth"
APPROVED_CHECKPOINT_SIZE_BYTES = 287755111
APPROVED_CHECKPOINT_SHA256 = "8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8"
CONSERVATIVE_RESTORATION_FRACTION = 0.25
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_DAMAGE_MARKERS = ("jpeg", "compression", "recompression")
_NETWORK_SPATIAL_MULTIPLE = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_checkout_metadata(upstream_root: Path) -> dict[str, Any]:
    path = upstream_root / ".cfs-upstream.json"
    if not path.is_file():
        raise RuntimeError(
            "FBCNN upstream checkout is missing .cfs-upstream.json; use the pinned CFS bootstrap"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("official_repository") != OFFICIAL_REPOSITORY:
        raise RuntimeError("FBCNN checkout metadata points to the wrong official repository")
    if payload.get("pinned_revision") != PINNED_REVISION:
        raise RuntimeError("FBCNN checkout metadata has the wrong pinned revision")
    if payload.get("actual_revision") != PINNED_REVISION:
        raise RuntimeError("FBCNN checkout is not at the approved detached revision")
    if payload.get("architecture_reimplemented_by_cfs") is not False:
        raise RuntimeError("FBCNN checkout metadata does not preserve upstream-first provenance")
    return payload


def _load_network_module(upstream_root: Path) -> ModuleType:
    source = upstream_root / "models" / "network_fbcnn.py"
    if not source.is_file():
        raise RuntimeError(f"Official FBCNN network module is missing: {source}")
    spec = importlib.util.spec_from_file_location("cfs_pinned_fbcnn_network", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot create an import spec for the official FBCNN network")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _damage_route_allowed(context: RestorationContext) -> bool:
    damage = str(context.damage_class).strip().lower()
    if any(marker in damage for marker in _ALLOWED_DAMAGE_MARKERS):
        return True
    return bool(context.metadata.get("jpeg_detected") is True)


def _conservative_blend(original_bgr: np.ndarray, restored_bgr: np.ndarray) -> np.ndarray:
    """Limit FBCNN pixel authority while preserving its measured JPEG benefit.

    Phase02 Windows validation showed that the full-strength official output can
    improve PSNR/SSIM while moving SFace identity materially away from the same
    clean identity on a small subset of adult identities. CFS therefore applies a
    fixed conservative authority fraction to every FBCNN result. This is not a
    threshold change and does not use clean ground truth at runtime.
    """
    if original_bgr.shape != restored_bgr.shape:
        raise ValueError("FBCNN conservative blend requires equal image shapes")
    if original_bgr.dtype != np.uint8 or restored_bgr.dtype != np.uint8:
        raise ValueError("FBCNN conservative blend requires uint8 images")
    alpha = float(CONSERVATIVE_RESTORATION_FRACTION)
    mixed = (
        original_bgr.astype(np.float32) * (1.0 - alpha)
        + restored_bgr.astype(np.float32) * alpha
    )
    return np.uint8(np.rint(np.clip(mixed, 0.0, 255.0)))


def _pad_for_network(
    image_bgr: np.ndarray,
    *,
    multiple: int = _NETWORK_SPATIAL_MULTIPLE,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Pad arbitrary detector crops for the official three-level FBCNN network.

    Installed face detectors return pixel-exact boxes whose dimensions are not
    necessarily divisible by eight. Padding is adapter preprocessing only: the
    official network is unchanged and its output is cropped back to the exact input
    canvas before any candidate can leave this boundary.
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("FBCNN network padding requires an HxWx3 image")
    divisor = int(multiple)
    if divisor <= 0:
        raise ValueError("FBCNN network padding multiple must be positive")
    height, width = image_bgr.shape[:2]
    pad_bottom = (-height) % divisor
    pad_right = (-width) % divisor
    padding = (0, 0, int(pad_right), int(pad_bottom))
    if pad_right == 0 and pad_bottom == 0:
        return np.ascontiguousarray(image_bgr), padding
    mode = "reflect" if height > 1 and width > 1 else "edge"
    padded = np.pad(
        image_bgr,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode=mode,
    )
    return np.ascontiguousarray(padded), padding


class FBCNNUpstreamBackend:
    """Thin CFS adapter around the exact official FBCNN implementation.

    No FBCNN architecture is copied into ConservativeFaceStudio. The backend imports
    ``models/network_fbcnn.py`` from a detached, pinned official checkout created by
    ``scripts/bootstrap_pinned_upstream.py``. CFS adds only lifecycle, input/output,
    provenance and qualification guards around the upstream network.
    """

    key = "fbcnn"
    version = PINNED_REVISION
    backend_name = "official-fbcnn-pytorch-upstream"
    estimated_load_bytes = 1_500_000_000

    def __init__(
        self,
        upstream_root: str | Path,
        checkpoint_path: str | Path,
        *,
        expected_checkpoint_sha256: str,
    ) -> None:
        expected = str(expected_checkpoint_sha256).strip().lower()
        if not _SHA256.fullmatch(expected):
            raise ValueError("expected_checkpoint_sha256 must be a full lowercase SHA-256")
        if expected != APPROVED_CHECKPOINT_SHA256:
            raise ValueError("expected_checkpoint_sha256 does not match the approved FBCNN checkpoint")
        self.upstream_root = Path(upstream_root).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.expected_checkpoint_sha256 = expected
        self._torch: Any | None = None
        self._model: Any | None = None
        self._checkout_metadata: dict[str, Any] | None = None
        self._checkpoint_sha256: str | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        metadata = _load_checkout_metadata(self.upstream_root)
        if not self.checkpoint_path.is_file():
            raise RuntimeError(f"FBCNN checkpoint is missing: {self.checkpoint_path}")
        actual_size = self.checkpoint_path.stat().st_size
        if actual_size != APPROVED_CHECKPOINT_SIZE_BYTES:
            raise RuntimeError(
                "FBCNN checkpoint byte size mismatch: "
                f"{actual_size} != {APPROVED_CHECKPOINT_SIZE_BYTES}"
            )
        actual_hash = _sha256(self.checkpoint_path)
        if actual_hash != self.expected_checkpoint_sha256:
            raise RuntimeError(
                "FBCNN checkpoint SHA-256 mismatch: "
                f"{actual_hash} != {self.expected_checkpoint_sha256}"
            )

        try:
            import torch
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("PyTorch is required for the official FBCNN upstream") from exc

        module = _load_network_module(self.upstream_root)
        network = getattr(module, "FBCNN", None)
        if network is None:
            raise RuntimeError("Official FBCNN module does not expose FBCNN")

        model = network(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode="R")
        try:
            state = torch.load(
                str(self.checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "Installed PyTorch cannot use weights_only=True; unsafe pickle fallback is disabled"
            ) from exc
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        self._torch = torch
        self._model = model.to(torch.device("cpu"))
        self._checkout_metadata = metadata
        self._checkpoint_sha256 = actual_hash

    def restore(
        self,
        face_bgr: np.ndarray,
        context: RestorationContext,
    ) -> RestorationCandidate:
        if self._model is None or self._torch is None:
            raise RuntimeError("FBCNN backend must be loaded before restore")
        if not _damage_route_allowed(context):
            raise RuntimeError("FBCNN is restricted to detected JPEG/recompression damage")
        if not isinstance(face_bgr, np.ndarray) or face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            raise ValueError("face_bgr must be an HxWx3 numpy array")
        if face_bgr.dtype != np.uint8:
            raise ValueError("face_bgr must use uint8 pixels")
        if self._checkpoint_sha256 is None:
            raise RuntimeError("FBCNN checkpoint identity was not verified during load")

        torch = self._torch
        network_input, padding = _pad_for_network(face_bgr)
        rgb = np.ascontiguousarray(network_input[:, :, ::-1])
        tensor = (
            torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(torch.device("cpu"))
        )

        qf_override = context.metadata.get("fbcnn_quality_factor")
        qf_tensor = None
        if qf_override is not None:
            qf_value = float(qf_override)
            if not 1.0 <= qf_value <= 100.0:
                raise ValueError("fbcnn_quality_factor must be in [1, 100]")
            qf_tensor = torch.tensor([[1.0 - qf_value / 100.0]], dtype=tensor.dtype)

        with torch.inference_mode():
            restored, raw_qf = self._model(tensor, qf_tensor) if qf_tensor is not None else self._model(tensor)

        output_rgb = (
            restored.detach()
            .squeeze(0)
            .float()
            .clamp(0.0, 1.0)
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )
        output_rgb = np.uint8(np.rint(output_rgb * 255.0))
        raw_output_bgr = np.ascontiguousarray(output_rgb[:, :, ::-1])
        raw_output_bgr = np.ascontiguousarray(
            raw_output_bgr[: face_bgr.shape[0], : face_bgr.shape[1]]
        )
        if raw_output_bgr.shape != face_bgr.shape:
            raise RuntimeError(
                f"FBCNN crop-back shape mismatch: {raw_output_bgr.shape} != {face_bgr.shape}"
            )
        output_bgr = _conservative_blend(face_bgr, raw_output_bgr)
        changed = np.any(output_bgr != face_bgr, axis=2)
        generated_mask = np.zeros(face_bgr.shape[:2], dtype=np.uint8)
        generated_mask[changed] = 255

        predicted_qf = float((1.0 - raw_qf.detach().float().cpu().reshape(-1)[0].item()) * 100.0)
        return RestorationCandidate(
            image=output_bgr,
            model_key=self.key,
            model_version=self.version,
            backend=self.backend_name,
            generated_mask=generated_mask,
            upstream_repository=OFFICIAL_REPOSITORY,
            upstream_revision=PINNED_REVISION,
            checkpoint_sha256=self._checkpoint_sha256,
            provenance_class=GENERATED_MODEL_INFERRED,
            quality_metrics={
                "predicted_jpeg_quality_factor": predicted_qf,
                "controlled_quality_factor": float(qf_override) if qf_override is not None else None,
                "conservative_restoration_fraction": float(CONSERVATIVE_RESTORATION_FRACTION),
                "official_repository": OFFICIAL_REPOSITORY,
                "official_revision": PINNED_REVISION,
                "checkpoint_sha256": self._checkpoint_sha256,
                "architecture_reimplemented_by_cfs": False,
                "cpu_only": True,
                "network_spatial_multiple": _NETWORK_SPATIAL_MULTIPLE,
                "input_padding_left": padding[0],
                "input_padding_top": padding[1],
                "input_padding_right": padding[2],
                "input_padding_bottom": padding[3],
            },
        )

    def unload(self) -> None:
        self._model = None
        self._torch = None
        self._checkout_metadata = None
        try:
            import torch

            if hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
