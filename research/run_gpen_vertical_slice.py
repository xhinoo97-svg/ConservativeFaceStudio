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

from app.core_models import ensure_core_pretrained_models
from app.face_analysis import cosine_similarity
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pretrained_values import FACE_MODEL_DEFAULTS


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_degrade(clean: np.ndarray) -> np.ndarray:
    """Development-only mixed blur/JPEG degradation; geometry is unchanged."""
    h, w = clean.shape[:2]
    scale = 0.58
    small = cv2.resize(clean, (max(32, int(round(w * scale))), max(32, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    restored_size = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(restored_size, (0, 0), 1.65)
    ok, encoded = cv2.imencode(".jpg", blurred, [int(cv2.IMWRITE_JPEG_QUALITY), 38])
    if not ok:
        raise RuntimeError("JPEG degradation encoding failed")
    degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if degraded is None or degraded.shape != clean.shape:
        raise RuntimeError("JPEG degradation decoding failed")
    return degraded


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Small deterministic SSIM implementation for this development slice."""
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a_gray, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b_gray, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a_gray * a_gray, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b_gray * b_gray, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a_gray * b_gray, (11, 11), 1.5) - mu_ab
    value = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2) + 1e-12)
    return float(np.mean(value))


class PeakRSSSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = interval_seconds
        self.process = psutil.Process(os.getpid())
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


def _rss_mb() -> float:
    return float(psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0))


def _load_gpen(gpen_root: Path, checkpoint: Path):
    face_model = gpen_root / "face_model"
    if not (face_model / "face_gan.py").is_file():
        raise RuntimeError(f"Official GPEN source not found: {face_model}")
    sys.path.insert(0, str(face_model))
    from face_gan import FaceGAN  # type: ignore

    expected = gpen_root / "weights" / "GPEN-BFR-512.pth"
    expected.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.resolve() != expected.resolve():
        if expected.exists():
            expected.unlink()
        try:
            expected.symlink_to(checkpoint.resolve())
        except OSError:
            import shutil
            shutil.copy2(checkpoint, expected)

    return FaceGAN(
        base_dir=str(gpen_root),
        in_size=512,
        out_size=512,
        model="GPEN-BFR-512",
        channel_multiplier=2,
        narrow=1,
        key=None,
        is_norm=True,
        device="cpu",
    )


def _align_for_gpen(image: np.ndarray, landmarks5: np.ndarray, gpen_root: Path) -> np.ndarray:
    if str(gpen_root) not in sys.path:
        sys.path.insert(0, str(gpen_root))
    from align_faces import get_reference_facial_points, warp_and_crop_face  # type: ignore

    reference = get_reference_facial_points(
        (512, 512),
        inner_padding_factor=0.25,
        outer_padding=(0, 0),
        default_square=True,
    )
    face, _ = warp_and_crop_face(
        image,
        np.asarray(landmarks5, dtype=np.float32),
        reference_pts=reference,
        crop_size=(512, 512),
    )
    if face.shape != (512, 512, 3):
        raise RuntimeError(f"Unexpected aligned shape: {face.shape}")
    return face


def _write_comparison(path: Path, clean: np.ndarray, degraded: np.ndarray, restored: np.ndarray) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    panels = []
    for label, image in (("CLEAN GT", clean), ("DEGRADED MAIN", degraded), ("GPEN BFR-512", restored)):
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (512, 42), (0, 0, 0), -1)
        cv2.putText(panel, label, (12, 29), font, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    sheet = np.hstack(panels)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError("Failed to save comparison sheet")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--gpen-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gpen-source-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if not args.input.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError("Input or GPEN checkpoint missing")
    input_sha = sha256_path(args.input)
    if input_sha.lower() != args.expected_input_sha256.lower():
        raise RuntimeError(f"Development source SHA-256 mismatch: {input_sha}")

    args.output.mkdir(parents=True, exist_ok=True)
    clean = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if clean is None:
        raise RuntimeError("Development source image cannot be decoded")
    degraded = deterministic_degrade(clean)

    model_root = args.output / "cfs-core-models"
    core = ensure_core_pretrained_models(model_root, timeout_seconds=60)
    if not core.ready:
        raise RuntimeError(f"YuNet/SFace bootstrap failed: {core.errors}")
    face_engine = OpenCVZooFaceEngine(
        core.paths["opencv_yunet"],
        core.paths["opencv_sface"],
        dnn_target="cpu",
    )

    main_observation = face_engine.analyze(degraded)
    clean_aligned = _align_for_gpen(clean, main_observation.landmarks5, args.gpen_root)
    degraded_aligned = _align_for_gpen(degraded, main_observation.landmarks5, args.gpen_root)

    cv2.imwrite(str(args.output / "clean_aligned.png"), clean_aligned)
    cv2.imwrite(str(args.output / "degraded_aligned.png"), degraded_aligned)

    baseline_rss = _rss_mb()
    load_started = time.perf_counter()
    with PeakRSSSampler() as load_sampler:
        model = _load_gpen(args.gpen_root, args.checkpoint)
    load_seconds = time.perf_counter() - load_started
    after_load_rss = _rss_mb()

    # Warm-up and measured inference are separate. Both remain CPU batch=1.
    with torch.inference_mode():
        _ = model.process(degraded_aligned)
    infer_started = time.perf_counter()
    with PeakRSSSampler() as infer_sampler:
        with torch.inference_mode():
            restored = model.process(degraded_aligned)
    inference_seconds = time.perf_counter() - infer_started
    if restored is None or restored.shape != (512, 512, 3) or restored.dtype != np.uint8:
        raise RuntimeError(f"Invalid GPEN output: shape={getattr(restored, 'shape', None)} dtype={getattr(restored, 'dtype', None)}")
    if not np.isfinite(restored.astype(np.float32)).all():
        raise RuntimeError("GPEN output contains non-finite values")

    restored_path = args.output / "restored_gpen_bfr_512.png"
    cv2.imwrite(str(restored_path), restored)
    _write_comparison(args.output / "comparison.png", clean_aligned, degraded_aligned, restored)

    clean_obs = face_engine.analyze(clean_aligned)
    degraded_obs = face_engine.analyze(degraded_aligned)
    restored_obs = face_engine.analyze(restored)
    if clean_obs.embedding is None or degraded_obs.embedding is None or restored_obs.embedding is None:
        raise RuntimeError("SFace embedding missing from vertical slice")
    identity_clean_restored = float(cosine_similarity(clean_obs.embedding, restored_obs.embedding))
    identity_clean_degraded = float(cosine_similarity(clean_obs.embedding, degraded_obs.embedding))
    identity_gate = identity_clean_restored >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine

    degraded_psnr = psnr(clean_aligned, degraded_aligned)
    restored_psnr = psnr(clean_aligned, restored)
    degraded_ssim = ssim_gray(clean_aligned, degraded_aligned)
    restored_ssim = ssim_gray(clean_aligned, restored)

    checkpoint_sha = sha256_path(args.checkpoint)
    checkpoint_size = int(args.checkpoint.stat().st_size)
    output_sha = sha256_path(restored_path)

    del model
    gc.collect()
    post_unload_rss = _rss_mb()

    report = {
        "experiment": "gpen_bfr_512_vertical_slice_v1",
        "qualification_scope": "development_host_cpu_only",
        "production_qualified": False,
        "production_blockers": [
            "GPEN code/weights redistribution license not yet explicitly verified from upstream",
            "upstream does not publish a pinned SHA-256 for GPEN-BFR-512; this run records observed SHA-256 only",
            "Windows installer execution not yet tested",
            "HP EliteBook 1030 G3 target-hardware timing/RAM not yet measured",
        ],
        "source": {
            "dataset": "cfs-face-smartphone-v1 development/calibration source bank",
            "input_path": str(args.input),
            "input_sha256": input_sha,
            "final_holdout_used": False,
        },
        "gpen": {
            "official_repository": "https://github.com/yangxy/GPEN",
            "official_source_commit": args.gpen_source_sha,
            "checkpoint_source": "https://public-vigen-video.oss-cn-shanghai.aliyuncs.com/robin/models/GPEN-BFR-512.pth",
            "checkpoint_sha256_observed": checkpoint_sha,
            "checkpoint_sha256_expected_upstream": None,
            "checkpoint_bytes": checkpoint_size,
            "device": "cpu",
            "torch_version": torch.__version__,
            "threads": int(args.threads),
        },
        "face_pipeline": {
            "detector": "OpenCV Zoo YuNet",
            "identity": "OpenCV Zoo SFace",
            "identity_threshold": float(FACE_MODEL_DEFAULTS.sface_same_identity_cosine),
            "alignment": "GPEN official 5-point similarity template at 512x512, landmarks supplied by YuNet",
            "main_yunet_score": float(main_observation.score),
        },
        "timing_seconds": {
            "model_load": float(load_seconds),
            "inference_512_measured_after_warmup": float(inference_seconds),
        },
        "rss_mb": {
            "baseline": baseline_rss,
            "peak_model_load": float(load_sampler.peak / (1024.0 * 1024.0)),
            "after_model_load": after_load_rss,
            "peak_inference": float(infer_sampler.peak / (1024.0 * 1024.0)),
            "post_unload_gc": post_unload_rss,
        },
        "metrics": {
            "sface_clean_vs_degraded": identity_clean_degraded,
            "sface_clean_vs_gpen": identity_clean_restored,
            "sface_identity_gate_pass": bool(identity_gate),
            "psnr_degraded": degraded_psnr,
            "psnr_gpen": restored_psnr,
            "ssim_degraded": degraded_ssim,
            "ssim_gpen": restored_ssim,
        },
        "outputs": {
            "restored_sha256": output_sha,
            "comparison": "comparison.png",
            "restored": restored_path.name,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_ram_gib": float(psutil.virtual_memory().total / (1024.0 ** 3)),
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if not identity_gate:
        raise RuntimeError(
            f"GPEN vertical slice identity gate failed: {identity_clean_restored:.6f} < {FACE_MODEL_DEFAULTS.sface_same_identity_cosine:.6f}"
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
