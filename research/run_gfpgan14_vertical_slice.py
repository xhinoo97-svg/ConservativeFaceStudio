from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
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


# Exact GPEN 512 alignment convention used by the Phase-3 comparison.
_REFERENCE_FACIAL_POINTS = np.array([
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.36550140],
    [62.72990036, 92.20410156],
], dtype=np.float64)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_degrade(clean: np.ndarray) -> np.ndarray:
    h, w = clean.shape[:2]
    scale = 0.58
    small = cv2.resize(clean, (max(32, int(round(w * scale))), max(32, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    restored_size = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(restored_size, (0, 0), 1.65)
    ok, encoded = cv2.imencode('.jpg', blurred, [int(cv2.IMWRITE_JPEG_QUALITY), 38])
    if not ok:
        raise RuntimeError('JPEG degradation encoding failed')
    degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if degraded is None or degraded.shape != clean.shape:
        raise RuntimeError('JPEG degradation decoding failed')
    return degraded


def _gpen_reference_512() -> np.ndarray:
    # Mirrors GPEN get_reference_facial_points((512,512), 0.25, (0,0), True).
    points = _REFERENCE_FACIAL_POINTS.copy()
    crop = np.array([96.0, 112.0], dtype=np.float64)
    size_diff = max(crop) - crop
    points += size_diff / 2.0
    crop += size_diff
    padding = crop * 0.25 * 2.0
    points += padding / 2.0
    crop += np.round(padding)
    points *= 512.0 / crop[0]
    return points.astype(np.float32)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    num = src.shape[0]
    dim = src.shape[1]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean
    a = dst_demean.T @ src_demean / num
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(a) < 0:
        d[-1] = -1
    u, s, v = np.linalg.svd(a)
    transform = np.eye(dim + 1, dtype=np.float64)
    rank = np.linalg.matrix_rank(a)
    if rank == 0:
        raise RuntimeError('Five-point similarity transform is rank deficient')
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(v) > 0:
            transform[:dim, :dim] = u @ v
        else:
            saved = d[-1]
            d[-1] = -1
            transform[:dim, :dim] = u @ np.diag(d) @ v
            d[-1] = saved
    else:
        transform[:dim, :dim] = u @ np.diag(d) @ v
    scale = 1.0 / src_demean.var(axis=0).sum() * (s @ d)
    transform[:dim, dim] = dst_mean - scale * (transform[:dim, :dim] @ src_mean.T)
    transform[:dim, :dim] *= scale
    return transform[:2].astype(np.float32)


def align_512(image: np.ndarray, landmarks5: np.ndarray) -> np.ndarray:
    matrix = _umeyama(np.asarray(landmarks5, dtype=np.float32), _gpen_reference_512())
    face = cv2.warpAffine(image, matrix, (512, 512), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT_101)
    if face.shape != (512, 512, 3):
        raise RuntimeError(f'Unexpected aligned shape: {face.shape}')
    return face


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float32) - b.astype(np.float32)
    mse = float(np.mean(diff * diff))
    if mse <= 1e-12:
        return float('inf')
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b_gray = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = cv2.GaussianBlur(a_gray, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b_gray, (11, 11), 1.5)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
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


def load_model(checkpoint: Path):
    import torch
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

    model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True,
    )
    payload = torch.load(checkpoint, map_location='cpu')
    key = 'params_ema' if isinstance(payload, dict) and 'params_ema' in payload else 'params'
    if not isinstance(payload, dict) or key not in payload:
        raise RuntimeError('GFPGAN checkpoint does not contain params_ema/params')
    model.load_state_dict(payload[key], strict=True)
    model.eval().to('cpu')
    return model, key


def infer(model, aligned_bgr: np.ndarray) -> np.ndarray:
    import torch

    rgb = aligned_bgr[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    with torch.inference_mode():
        output = model(tensor, return_rgb=False, randomize_noise=False)[0]
    output = output.squeeze(0).detach().cpu().float().clamp(-1.0, 1.0)
    output = ((output + 1.0) * 0.5 * 255.0).round().to(torch.uint8).numpy().transpose(1, 2, 0)
    return np.ascontiguousarray(output[:, :, ::-1])


def write_comparison(path: Path, clean: np.ndarray, degraded: np.ndarray, restored: np.ndarray) -> None:
    panels = []
    for label, image in (('CLEAN GT', clean), ('DEGRADED MAIN', degraded), ('GFPGAN v1.4', restored)):
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (512, 42), (0, 0, 0), -1)
        cv2.putText(panel, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise RuntimeError('Failed to save comparison sheet')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--expected-input-sha256', required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    if not args.input.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError('Input or GFPGAN checkpoint missing')
    input_sha = sha256_path(args.input)
    if input_sha.lower() != args.expected_input_sha256.lower():
        raise RuntimeError(f'Development source SHA-256 mismatch: {input_sha}')

    args.output.mkdir(parents=True, exist_ok=True)
    clean = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if clean is None:
        raise RuntimeError('Development source image cannot be decoded')
    degraded = deterministic_degrade(clean)

    core = ensure_core_pretrained_models(args.output / 'cfs-core-models', timeout_seconds=60)
    if not core.ready:
        raise RuntimeError(f'YuNet/SFace bootstrap failed: {core.errors}')
    engine = OpenCVZooFaceEngine(core.paths['opencv_yunet'], core.paths['opencv_sface'], dnn_target='cpu')
    main_obs = engine.analyze(degraded)
    clean_aligned = align_512(clean, main_obs.landmarks5)
    degraded_aligned = align_512(degraded, main_obs.landmarks5)
    cv2.imwrite(str(args.output / 'clean_aligned.png'), clean_aligned)
    cv2.imwrite(str(args.output / 'degraded_aligned.png'), degraded_aligned)

    baseline_rss = _rss_mb()
    load_start = time.perf_counter()
    with PeakRSSSampler() as load_sampler:
        model, checkpoint_key = load_model(args.checkpoint)
    load_seconds = time.perf_counter() - load_start
    after_load_rss = _rss_mb()

    _ = infer(model, degraded_aligned)  # warm-up
    infer_start = time.perf_counter()
    with PeakRSSSampler() as infer_sampler:
        restored = infer(model, degraded_aligned)
    inference_seconds = time.perf_counter() - infer_start

    if restored.shape != (512, 512, 3) or restored.dtype != np.uint8:
        raise RuntimeError(f'Invalid GFPGAN output: shape={restored.shape} dtype={restored.dtype}')
    restored_path = args.output / 'restored_gfpgan_v1_4.png'
    cv2.imwrite(str(restored_path), restored)
    write_comparison(args.output / 'comparison.png', clean_aligned, degraded_aligned, restored)

    clean_obs = engine.analyze(clean_aligned)
    degraded_obs = engine.analyze(degraded_aligned)
    restored_obs = engine.analyze(restored)
    if clean_obs.embedding is None or degraded_obs.embedding is None or restored_obs.embedding is None:
        raise RuntimeError('SFace embedding missing')
    identity_degraded = float(cosine_similarity(clean_obs.embedding, degraded_obs.embedding))
    identity_restored = float(cosine_similarity(clean_obs.embedding, restored_obs.embedding))
    identity_pass = identity_restored >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine

    report = {
        'experiment': 'gfpgan_v1_4_vertical_slice_v1',
        'qualification_scope': 'development_host_cpu_only',
        'production_qualified': False,
        'production_blockers': [
            'Windows installer execution not yet tested',
            'HP EliteBook 1030 G3 target-hardware timing/RAM not yet measured',
            'single development image is insufficient for production quality qualification',
            'GitHub release asset does not publish an authoritative SHA-256 digest; observed digest is recorded',
        ],
        'source': {
            'dataset': 'cfs-face-smartphone-v1 development/calibration source bank',
            'input_path': str(args.input),
            'input_sha256': input_sha,
            'final_holdout_used': False,
        },
        'model': {
            'key': 'gfpgan_v1_4',
            'official_repository': 'https://github.com/TencentARC/GFPGAN',
            'official_source_commit': args.source_sha,
            'checkpoint_source': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth',
            'checkpoint_bytes': int(args.checkpoint.stat().st_size),
            'checkpoint_sha256_observed': sha256_path(args.checkpoint),
            'checkpoint_sha256_expected_upstream': None,
            'checkpoint_state_key': checkpoint_key,
            'architecture': 'GFPGANv1Clean',
            'channel_multiplier': 2,
            'randomize_noise': False,
            'device': 'cpu',
            'torch_version': torch.__version__,
            'threads': int(args.threads),
        },
        'face_pipeline': {
            'detector': 'OpenCV Zoo YuNet',
            'identity': 'OpenCV Zoo SFace',
            'identity_threshold': float(FACE_MODEL_DEFAULTS.sface_same_identity_cosine),
            'alignment': 'same GPEN official 5-point similarity template at 512x512 used by Phase 3',
            'main_yunet_score': float(main_obs.score),
        },
        'timing_seconds': {
            'model_load': float(load_seconds),
            'inference_512_measured_after_warmup': float(inference_seconds),
        },
        'rss_mb': {
            'baseline': baseline_rss,
            'peak_model_load': float(load_sampler.peak / (1024.0 * 1024.0)),
            'after_model_load': after_load_rss,
            'peak_inference': float(infer_sampler.peak / (1024.0 * 1024.0)),
        },
        'metrics': {
            'sface_clean_vs_degraded': identity_degraded,
            'sface_clean_vs_gfpgan': identity_restored,
            'sface_identity_gate_pass': bool(identity_pass),
            'psnr_degraded': psnr(clean_aligned, degraded_aligned),
            'psnr_gfpgan': psnr(clean_aligned, restored),
            'ssim_degraded': ssim_gray(clean_aligned, degraded_aligned),
            'ssim_gfpgan': ssim_gray(clean_aligned, restored),
        },
        'outputs': {
            'restored': restored_path.name,
            'restored_sha256': sha256_path(restored_path),
            'comparison': 'comparison.png',
        },
        'host': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'cpu': platform.processor(),
            'logical_cpu_count': psutil.cpu_count(logical=True),
            'physical_cpu_count': psutil.cpu_count(logical=False),
            'total_ram_gib': float(psutil.virtual_memory().total / (1024.0 ** 3)),
        },
    }

    del model
    gc.collect()
    report['rss_mb']['post_unload_gc'] = _rss_mb()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')

    if not identity_pass:
        raise RuntimeError(f'GFPGAN v1.4 identity gate failed: {identity_restored:.6f} < {FACE_MODEL_DEFAULTS.sface_same_identity_cosine:.6f}')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
