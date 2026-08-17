from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil

import run_gfpgan14_vertical_slice as common
from run_gfpgan14_vertical_slice_exact import exact_phase3_alignment

from app.core_models import ensure_core_pretrained_models
from app.face_analysis import cosine_similarity
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pretrained_values import FACE_MODEL_DEFAULTS


def load_model(root: Path, checkpoint: Path):
    import torch
    sys.path.insert(0, str(root))
    from basicsr.archs.codeformer_arch import CodeFormer

    net = CodeFormer(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256'],
    ).to('cpu')
    payload = torch.load(checkpoint, map_location='cpu')
    if not isinstance(payload, dict) or 'params_ema' not in payload:
        raise RuntimeError('CodeFormer checkpoint missing params_ema')
    net.load_state_dict(payload['params_ema'], strict=True)
    net.eval()
    return net


def infer(net, aligned_bgr: np.ndarray, w: float) -> np.ndarray:
    import torch
    rgb = aligned_bgr[:, :, ::-1].astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    with torch.inference_mode():
        output = net(tensor, w=float(w), adain=True)[0]
    output = output.squeeze(0).detach().cpu().float().clamp(-1.0, 1.0)
    output = ((output + 1.0) * 0.5 * 255.0).round().to(torch.uint8).numpy().transpose(1, 2, 0)
    return np.ascontiguousarray(output[:, :, ::-1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--expected-input-sha256', required=True)
    parser.add_argument('--codeformer-root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fidelity-weight', type=float, default=0.5)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    import torch
    torch.set_num_threads(max(1, int(args.threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    input_sha = common.sha256_path(args.input)
    if input_sha.lower() != args.expected_input_sha256.lower():
        raise RuntimeError(f'Development source SHA mismatch: {input_sha}')
    clean = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if clean is None:
        raise RuntimeError('Input decode failed')
    degraded = common.deterministic_degrade(clean)
    args.output.mkdir(parents=True, exist_ok=True)

    core = ensure_core_pretrained_models(args.output / 'cfs-core-models', timeout_seconds=60)
    if not core.ready:
        raise RuntimeError(f'YuNet/SFace bootstrap failed: {core.errors}')
    engine = OpenCVZooFaceEngine(core.paths['opencv_yunet'], core.paths['opencv_sface'], dnn_target='cpu')
    main_obs = engine.analyze(degraded)
    clean_aligned = exact_phase3_alignment(clean, main_obs.landmarks5)
    degraded_aligned = exact_phase3_alignment(degraded, main_obs.landmarks5)

    cv2.imwrite(str(args.output / 'clean_aligned.png'), clean_aligned)
    cv2.imwrite(str(args.output / 'degraded_aligned.png'), degraded_aligned)

    baseline = common._rss_mb()
    load_start = time.perf_counter()
    with common.PeakRSSSampler() as load_sampler:
        net = load_model(args.codeformer_root, args.checkpoint)
    load_seconds = time.perf_counter() - load_start
    after_load = common._rss_mb()

    _ = infer(net, degraded_aligned, args.fidelity_weight)
    infer_start = time.perf_counter()
    with common.PeakRSSSampler() as infer_sampler:
        restored = infer(net, degraded_aligned, args.fidelity_weight)
    infer_seconds = time.perf_counter() - infer_start

    restored_path = args.output / 'restored_codeformer_w05.png'
    cv2.imwrite(str(restored_path), restored)
    common.write_comparison(args.output / 'comparison.png', clean_aligned, degraded_aligned, restored)

    clean_obs = engine.analyze(clean_aligned)
    degraded_obs = engine.analyze(degraded_aligned)
    restored_obs = engine.analyze(restored)
    if clean_obs.embedding is None or degraded_obs.embedding is None or restored_obs.embedding is None:
        raise RuntimeError('SFace embedding missing')
    identity_degraded = float(cosine_similarity(clean_obs.embedding, degraded_obs.embedding))
    identity_restored = float(cosine_similarity(clean_obs.embedding, restored_obs.embedding))
    identity_pass = identity_restored >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine

    report = {
        'experiment': 'codeformer_vertical_slice_v1',
        'qualification_scope': 'development_host_cpu_only',
        'production_qualified': False,
        'production_blockers': [
            'NTU S-Lab License 1.0 constraints must be respected; production redistribution/use not qualified',
            'Windows installer execution not tested',
            'HP EliteBook 1030 G3 not measured',
            'single development image insufficient for production quality qualification',
            'official GitHub asset provides no expected SHA-256 digest; observed digest only',
        ],
        'source': {
            'dataset': 'cfs-face-smartphone-v1 development/calibration source bank',
            'input_sha256': input_sha,
            'final_holdout_used': False,
        },
        'model': {
            'key': 'codeformer',
            'official_repository': 'https://github.com/sczhou/CodeFormer',
            'official_source_commit': args.source_sha,
            'checkpoint_source': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
            'checkpoint_bytes': int(args.checkpoint.stat().st_size),
            'checkpoint_sha256_observed': common.sha256_path(args.checkpoint),
            'checkpoint_sha256_expected_upstream': None,
            'fidelity_weight': float(args.fidelity_weight),
            'device': 'cpu',
            'torch_version': torch.__version__,
            'threads': int(args.threads),
        },
        'face_pipeline': {
            'detector': 'OpenCV Zoo YuNet',
            'identity': 'OpenCV Zoo SFace',
            'identity_threshold': float(FACE_MODEL_DEFAULTS.sface_same_identity_cosine),
            'alignment': 'exact Phase-3 GPEN official 5-point 512 convention',
            'main_yunet_score': float(main_obs.score),
        },
        'timing_seconds': {
            'model_load': float(load_seconds),
            'inference_512_measured_after_warmup': float(infer_seconds),
        },
        'rss_mb': {
            'baseline': baseline,
            'peak_model_load': float(load_sampler.peak / (1024.0 * 1024.0)),
            'after_model_load': after_load,
            'peak_inference': float(infer_sampler.peak / (1024.0 * 1024.0)),
        },
        'metrics': {
            'sface_clean_vs_degraded': identity_degraded,
            'sface_clean_vs_codeformer': identity_restored,
            'sface_identity_gate_pass': bool(identity_pass),
            'psnr_degraded': common.psnr(clean_aligned, degraded_aligned),
            'psnr_codeformer': common.psnr(clean_aligned, restored),
            'ssim_degraded': common.ssim_gray(clean_aligned, degraded_aligned),
            'ssim_codeformer': common.ssim_gray(clean_aligned, restored),
        },
        'outputs': {
            'restored': restored_path.name,
            'restored_sha256': common.sha256_path(restored_path),
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
    del net
    gc.collect()
    report['rss_mb']['post_unload_gc'] = common._rss_mb()
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')

    if not identity_pass:
        raise RuntimeError(f'CodeFormer identity gate failed: {identity_restored:.6f} < {FACE_MODEL_DEFAULTS.sface_same_identity_cosine:.6f}')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
