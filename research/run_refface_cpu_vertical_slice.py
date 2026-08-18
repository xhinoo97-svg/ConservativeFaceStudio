from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

from app.face_analysis import cosine_similarity
from app.face_parsing_runtime import FaceParsingRuntime, one_hot_parsing
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pretrained_values import FACE_MODEL_DEFAULTS
from app.resource_budget import (
    apply_resource_budget,
    assert_memory_within_budget,
    detect_resource_budget,
    resource_snapshot,
)


REF_FACE_COMMIT = "0f1ad75677cc8fae4ae14d878e4c6cfce9365f28"
REF_FACE_LICENSE = "MIT"
GENERATED_PROVENANCE = "GENERATED_MODEL_INFERRED"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PeakRSSSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.process = psutil.Process(os.getpid())
        self.interval_seconds = interval_seconds
        self.peak = int(self.process.memory_info().rss)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, int(self.process.memory_info().rss))
            except psutil.Error:
                pass
            self._stop.wait(self.interval_seconds)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            self.peak = max(self.peak, int(self.process.memory_info().rss))
        except psutil.Error:
            pass


def _rss_bytes() -> int:
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _align_with_sface(engine: OpenCVZooFaceEngine, image: np.ndarray) -> tuple[np.ndarray, object]:
    observation = engine.analyze(image)
    recognizer = engine.recognizer
    detector = engine.detector
    if recognizer is None or detector is None:
        raise RuntimeError("YuNet/SFace engine is incomplete")
    detector.setInputSize((int(image.shape[1]), int(image.shape[0])))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise RuntimeError("YuNet could not redetect face for SFace alignment")
    faces = np.asarray(faces, np.float32)
    areas = np.maximum(faces[:, 2], 0) * np.maximum(faces[:, 3], 0)
    scores = faces[:, 14] if faces.shape[1] > 14 else np.ones(len(faces), np.float32)
    face = faces[int(np.argmax(areas * np.maximum(scores, 1e-6)))]
    aligned112 = recognizer.alignCrop(image, face[:-1])
    if aligned112 is None or aligned112.shape[:2] != (112, 112):
        raise RuntimeError(f"Unexpected SFace aligned crop: {getattr(aligned112, 'shape', None)}")
    aligned256 = cv2.resize(aligned112, (256, 256), interpolation=cv2.INTER_CUBIC)
    return aligned256, observation


def _make_mouth_occlusion(clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = clean.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    # Aligned-face deterministic lower-central opaque block. Exact mask is retained.
    x1, x2 = int(round(0.27 * w)), int(round(0.73 * w))
    y1, y2 = int(round(0.59 * h)), int(round(0.78 * h))
    cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, -1)
    degraded = clean.copy()
    degraded[mask > 0] = np.asarray([35, 35, 35], dtype=np.uint8)
    return degraded, mask


def _to_ref_face_tensor(image_bgr: np.ndarray, torch):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    value = rgb * 2.0 - 1.0
    return torch.from_numpy(np.transpose(value, (2, 0, 1))).unsqueeze(0).float()


def _to_bgr_uint8(tensor, torch) -> np.ndarray:
    value = tensor.detach().cpu().float().clamp(-1.0, 1.0)[0]
    rgb = ((value.permute(1, 2, 0).numpy() + 1.0) * 127.5).round().clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _load_refface(ref_face_root: Path, generator_checkpoint: Path, arcface_checkpoint: Path, torch):
    if not (ref_face_root / "networks" / "UnetG.py").is_file():
        raise RuntimeError("Pinned RefFaceInpainting source tree missing")
    sys.path.insert(0, str(ref_face_root))
    from networks.UnetG import UnetG  # type: ignore
    from networks.arcface_models import resnet101  # type: ignore

    cfg = {
        "input_nc": 3,
        "output_nc": 3,
        "ngf": 32,
        "G_norm_type": "in",
        "style_dim": 64,
    }
    generator = UnetG(cfg).cpu().eval()
    payload = torch.load(generator_checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "netG" not in payload:
        raise RuntimeError("RefFace checkpoint does not contain netG")
    generator.load_state_dict(payload["netG"], strict=True)

    arcface = resnet101().cpu().eval()
    arc_payload = torch.load(arcface_checkpoint, map_location="cpu")
    if isinstance(arc_payload, dict) and "state_dict" in arc_payload and isinstance(arc_payload["state_dict"], dict):
        arc_payload = arc_payload["state_dict"]
    if not isinstance(arc_payload, dict):
        raise RuntimeError("RefFace ArcFace checkpoint is not a state dict")
    arcface.load_state_dict(arc_payload, strict=True)
    for model in (generator, arcface):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return generator, arcface


def _psnr_masked(clean: np.ndarray, candidate: np.ndarray, mask: np.ndarray) -> float:
    active = mask > 0
    if not np.any(active):
        return float("nan")
    diff = clean[active].astype(np.float32) - candidate[active].astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _write_comparison(path: Path, clean: np.ndarray, degraded: np.ndarray, reference: np.ndarray, restored: np.ndarray) -> None:
    panels = []
    for label, image in (
        ("CLEAN MAIN", clean),
        ("OCCLUDED MAIN", degraded),
        ("SAME-ID REF", reference),
        ("REFFACE GENERATED", restored),
    ):
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (256, 27), (0, 0, 0), -1)
        cv2.putText(panel, label, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise RuntimeError("Could not save RefFace comparison")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-image", required=True, type=Path)
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--ref-face-root", required=True, type=Path)
    parser.add_argument("--generator-checkpoint", required=True, type=Path)
    parser.add_argument("--arcface-checkpoint", required=True, type=Path)
    parser.add_argument("--parser-model", required=True, type=Path)
    parser.add_argument("--yunet-model", required=True, type=Path)
    parser.add_argument("--sface-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    budget = detect_resource_budget(0.80)
    apply_resource_budget(budget)
    args.output.mkdir(parents=True, exist_ok=True)

    for path in (
        args.main_image,
        args.reference_image,
        args.generator_checkpoint,
        args.arcface_checkpoint,
        args.parser_model,
        args.yunet_model,
        args.sface_model,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    main_raw = cv2.imread(str(args.main_image), cv2.IMREAD_COLOR)
    ref_raw = cv2.imread(str(args.reference_image), cv2.IMREAD_COLOR)
    if main_raw is None or ref_raw is None:
        raise RuntimeError("Development MAIN/reference could not be decoded")

    identity_engine = OpenCVZooFaceEngine(args.yunet_model, args.sface_model, dnn_target="cpu")
    clean, _ = _align_with_sface(identity_engine, main_raw)
    reference, _ = _align_with_sface(identity_engine, ref_raw)
    degraded, mask_u8 = _make_mouth_occlusion(clean)
    if np.array_equal(clean, degraded):
        raise RuntimeError("Synthetic occlusion did not alter MAIN")

    parser_runtime = FaceParsingRuntime(args.parser_model)
    ref_labels = parser_runtime.predict(reference)
    ref_onehot_np = one_hot_parsing(ref_labels)
    if ref_onehot_np.shape != (19, 256, 256):
        raise RuntimeError(f"Unexpected parsing one-hot: {ref_onehot_np.shape}")

    clean_tensor = _to_ref_face_tensor(clean, torch)
    degraded_tensor = _to_ref_face_tensor(degraded, torch)
    reference_tensor = _to_ref_face_tensor(reference, torch)
    mask = torch.from_numpy((mask_u8.astype(np.float32) / 255.0)[None, None, ...])
    ref_onehot = torch.from_numpy(ref_onehot_np[None, ...]).float()

    before = resource_snapshot(budget)
    estimated_load = int(args.generator_checkpoint.stat().st_size + args.arcface_checkpoint.stat().st_size) * 3
    assert_memory_within_budget(budget, stage="refface_preload", reserve_bytes=estimated_load)

    load_started = time.perf_counter()
    with PeakRSSSampler() as load_sampler:
        generator, arcface = _load_refface(
            args.ref_face_root,
            args.generator_checkpoint,
            args.arcface_checkpoint,
            torch,
        )
    load_seconds = time.perf_counter() - load_started
    assert_memory_within_budget(budget, stage="refface_postload")
    after_load = resource_snapshot(budget)

    with torch.inference_mode():
        ref_id = arcface(F.interpolate(reference_tensor, size=112, mode="bilinear", align_corners=False))
        # Warm-up once under the exact measured inputs.
        _ = generator(
            torch.cat((degraded_tensor, mask), dim=1),
            ref_id,
            reference_tensor,
            ref_onehot,
            mask,
            train_mode="inpainting",
        )[0]

    infer_started = time.perf_counter()
    with PeakRSSSampler() as infer_sampler:
        with torch.inference_mode():
            fake = generator(
                torch.cat((degraded_tensor, mask), dim=1),
                ref_id,
                reference_tensor,
                ref_onehot,
                mask,
                train_mode="inpainting",
            )[0]
            fused_tensor = fake * mask + clean_tensor * (1.0 - mask)
    inference_seconds = time.perf_counter() - infer_started
    assert_memory_within_budget(budget, stage="refface_postinference")

    restored = _to_bgr_uint8(fused_tensor, torch)
    if restored.shape != clean.shape or restored.dtype != np.uint8:
        raise RuntimeError("RefFace output has invalid shape/dtype")
    outside = mask_u8 == 0
    # Round-trip normalization may differ by at most one code value. Enforce exact MAIN
    # in the exported candidate, as CFS fusion will do in production.
    restored[outside] = clean[outside]
    if not np.array_equal(restored[outside], clean[outside]):
        raise RuntimeError("RefFace exported candidate changed healthy MAIN outside mask")

    clean_obs = identity_engine.analyze(clean)
    ref_obs = identity_engine.analyze(reference)
    restored_obs = identity_engine.analyze(restored)
    if clean_obs.embedding is None or ref_obs.embedding is None or restored_obs.embedding is None:
        raise RuntimeError("SFace embedding unavailable for RefFace evidence")
    sface_clean_ref = float(cosine_similarity(clean_obs.embedding, ref_obs.embedding))
    sface_clean_restored = float(cosine_similarity(clean_obs.embedding, restored_obs.embedding))
    sface_ref_restored = float(cosine_similarity(ref_obs.embedding, restored_obs.embedding))
    threshold = float(FACE_MODEL_DEFAULTS.sface_same_identity_cosine)
    identity_pass = sface_clean_restored >= threshold and sface_ref_restored >= threshold

    cv2.imwrite(str(args.output / "clean_main.png"), clean)
    cv2.imwrite(str(args.output / "occluded_main.png"), degraded)
    cv2.imwrite(str(args.output / "same_identity_reference.png"), reference)
    cv2.imwrite(str(args.output / "exact_occlusion_mask.png"), mask_u8)
    cv2.imwrite(str(args.output / "reference_parsing_labels.png"), ref_labels)
    restored_path = args.output / "restored_refface.png"
    cv2.imwrite(str(restored_path), restored)
    _write_comparison(args.output / "comparison.png", clean, degraded, reference, restored)

    checkpoint_evidence = {
        "generator": {
            "bytes": int(args.generator_checkpoint.stat().st_size),
            "sha256_observed": sha256_path(args.generator_checkpoint),
            "sha256_expected_upstream": None,
        },
        "arcface": {
            "bytes": int(args.arcface_checkpoint.stat().st_size),
            "sha256_observed": sha256_path(args.arcface_checkpoint),
            "sha256_expected_upstream": None,
        },
        "parser": {
            "bytes": int(args.parser_model.stat().st_size),
            "sha256": sha256_path(args.parser_model),
        },
    }

    del generator, arcface, parser_runtime
    gc.collect()
    after_unload = resource_snapshot(budget)

    report = {
        "experiment": "reffaceinpainting_cpu_vertical_slice_v1",
        "qualification_scope": "development_host_cpu_only",
        "production_qualified": False,
        "provenance_class": GENERATED_PROVENANCE,
        "final_holdout_used": False,
        "upstream": {
            "repository": "https://github.com/WuyangLuo/RefFaceInpainting",
            "commit": REF_FACE_COMMIT,
            "license": REF_FACE_LICENSE,
            "cpu_patch": "SegBranch torch.cuda.FloatTensor -> input tensor new_zeros only; no architecture/weight change",
            "generator_checkpoint_google_drive_id": "1qn1fKj-4iwykSZl_GT9kjz2UTnbMlU36",
            "arcface_checkpoint_google_drive_id": "1VpD27jHOPaOJRKFqOO_txLHAE05CbtLU",
        },
        "inputs": {
            "main_sha256": sha256_path(args.main_image),
            "reference_sha256": sha256_path(args.reference_image),
            "same_identity_reference_gate": sface_clean_ref >= threshold,
            "alignment": "OpenCV SFace official aligned crop 112 resized to 256 for RefFace input",
            "damage": "deterministic opaque lower-central mouth-region block",
            "damage_pixels": int(np.count_nonzero(mask_u8)),
        },
        "models": checkpoint_evidence,
        "identity": {
            "backend": "OpenCV Zoo SFace",
            "threshold": threshold,
            "clean_vs_reference": sface_clean_ref,
            "clean_vs_restored": sface_clean_restored,
            "reference_vs_restored": sface_ref_restored,
            "gate_pass": bool(identity_pass),
        },
        "quality": {
            "masked_psnr_occluded": _psnr_masked(clean, degraded, mask_u8),
            "masked_psnr_refface": _psnr_masked(clean, restored, mask_u8),
            "healthy_outside_exact_main": True,
            "generated_fraction": float(np.mean(mask_u8 > 0)),
        },
        "runtime": {
            "device": "cpu",
            "torch_version": torch.__version__,
            "model_load_seconds": float(load_seconds),
            "inference_256_seconds_after_warmup": float(inference_seconds),
            "peak_load_rss_bytes": int(load_sampler.peak),
            "peak_inference_rss_bytes": int(infer_sampler.peak),
            "resource_budget": budget.to_dict(),
            "before_load": before,
            "after_load": after_load,
            "after_unload_gc": after_unload,
        },
        "output": {
            "sha256": sha256_path(restored_path),
            "comparison": "comparison.png",
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_ram_gib": float(psutil.virtual_memory().total / (1024.0 ** 3)),
        },
        "remaining_blockers": [
            "Windows execution not yet measured",
            "HP EliteBook 1030 G3 target-hardware execution not yet measured",
            "single DEVELOPMENT identity is insufficient for qualification",
            "upstream Google Drive checkpoint SHA-256 values are not published; observed hashes only",
        ],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if sface_clean_ref < threshold:
        raise RuntimeError(f"Development reference is not accepted same identity: {sface_clean_ref:.6f} < {threshold:.6f}")
    if not identity_pass:
        raise RuntimeError(
            f"RefFace identity gate failed: clean={sface_clean_restored:.6f} ref={sface_ref_restored:.6f} threshold={threshold:.6f}"
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
