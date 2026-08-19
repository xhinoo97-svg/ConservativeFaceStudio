from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

import run_gfpgan14_vertical_slice as common
from run_gfpgan14_vertical_slice_exact import exact_phase3_alignment

from app.core_models import ensure_core_pretrained_models
from app.face_analysis import cosine_similarity
from app.face_restorer_adapter import GENERATED_MODEL_INFERRED, RestorationContext
from app.fbcnn_upstream_backend import FBCNNUpstreamBackend, PINNED_REVISION
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pretrained_values import FACE_MODEL_DEFAULTS
from app.resource_budget import (
    apply_resource_budget,
    assert_memory_within_budget,
    detect_resource_budget,
    resource_snapshot,
)


def jpeg_degrade(clean_bgr: np.ndarray, quality: int) -> np.ndarray:
    q = int(quality)
    if not 1 <= q <= 100:
        raise ValueError('JPEG quality must be 1..100')
    ok, encoded = cv2.imencode('.jpg', clean_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    degraded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if degraded is None or degraded.shape != clean_bgr.shape:
        raise RuntimeError('JPEG decoding failed')
    return degraded


def write_comparison(path: Path, clean: np.ndarray, degraded: np.ndarray, restored: np.ndarray, qf: int) -> None:
    panels: list[np.ndarray] = []
    for label, image in (
        ('CLEAN GT', clean),
        (f'JPEG QF={qf}', degraded),
        ('FBCNN OFFICIAL', restored),
    ):
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (512, 42), (0, 0, 0), -1)
        cv2.putText(panel, label, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise RuntimeError('Failed to save FBCNN comparison')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--expected-input-sha256', required=True)
    parser.add_argument('--fbcnn-root', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--expected-checkpoint-sha256', required=True)
    parser.add_argument('--source-sha', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--jpeg-quality', type=int, default=20)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    if args.source_sha != PINNED_REVISION:
        raise RuntimeError(f'Unexpected FBCNN source SHA: {args.source_sha} != {PINNED_REVISION}')

    budget = detect_resource_budget(0.80)
    apply_resource_budget(budget)

    import torch
    effective_threads = max(1, min(int(args.threads), int(budget.allowed_processors)))
    torch.set_num_threads(effective_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cv2.setNumThreads(effective_threads)

    if not args.input.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError('Input or FBCNN checkpoint missing')
    input_sha = common.sha256_path(args.input)
    if input_sha.lower() != args.expected_input_sha256.lower():
        raise RuntimeError(f'Development source SHA mismatch: {input_sha}')
    checkpoint_sha = common.sha256_path(args.checkpoint)
    if checkpoint_sha.lower() != args.expected_checkpoint_sha256.lower():
        raise RuntimeError(
            f'FBCNN checkpoint SHA mismatch: {checkpoint_sha} != {args.expected_checkpoint_sha256}'
        )

    clean = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if clean is None:
        raise RuntimeError('Input decode failed')
    degraded = jpeg_degrade(clean, args.jpeg_quality)
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

    assert_memory_within_budget(
        budget,
        stage='fbcnn_preload',
        reserve_bytes=1_500_000_000,
    )
    resource_before_load = resource_snapshot(budget)

    backend = FBCNNUpstreamBackend(
        args.fbcnn_root,
        args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    baseline = common._rss_mb()
    load_start = time.perf_counter()
    with common.PeakRSSSampler() as load_sampler:
        backend.load()
    load_seconds = time.perf_counter() - load_start
    assert_memory_within_budget(budget, stage='fbcnn_postload')
    resource_after_load = resource_snapshot(budget)

    context = RestorationContext(
        damage_class='single_jpeg',
        severity='heavy',
        metadata={'jpeg_detected': True},
    )
    _warm = backend.restore(degraded_aligned, context)
    assert_memory_within_budget(budget, stage='fbcnn_post_warmup')
    infer_start = time.perf_counter()
    with common.PeakRSSSampler() as infer_sampler:
        candidate = backend.restore(degraded_aligned, context)
    infer_seconds = time.perf_counter() - infer_start
    assert_memory_within_budget(budget, stage='fbcnn_post_inference')
    resource_after_inference = resource_snapshot(budget)

    restored = candidate.image
    if restored.shape != degraded_aligned.shape or restored.dtype != np.uint8:
        raise RuntimeError(f'Invalid FBCNN output: {restored.shape} {restored.dtype}')
    if candidate.provenance_class != GENERATED_MODEL_INFERRED:
        raise RuntimeError(f'Invalid FBCNN provenance: {candidate.provenance_class}')
    if candidate.model_version != PINNED_REVISION:
        raise RuntimeError(f'FBCNN candidate revision drift: {candidate.model_version}')

    restored_path = args.output / 'restored_fbcnn_color.png'
    cv2.imwrite(str(restored_path), restored)
    write_comparison(args.output / 'comparison.png', clean_aligned, degraded_aligned, restored, args.jpeg_quality)

    clean_obs = engine.analyze(clean_aligned)
    degraded_obs = engine.analyze(degraded_aligned)
    restored_obs = engine.analyze(restored)
    if clean_obs.embedding is None or degraded_obs.embedding is None or restored_obs.embedding is None:
        raise RuntimeError('SFace embedding missing')
    identity_degraded = float(cosine_similarity(clean_obs.embedding, degraded_obs.embedding))
    identity_restored = float(cosine_similarity(clean_obs.embedding, restored_obs.embedding))
    identity_pass = identity_restored >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine

    predicted_quality = float(candidate.quality_metrics['predicted_jpeg_quality_factor'])
    report = {
        'experiment': 'fbcnn_color_jpeg_specialist_vertical_slice_v2_upstream_adapter',
        'qualification_scope': 'development_host_cpu_only',
        'production_qualified': False,
        'production_blockers': [
            'Windows installer execution not tested',
            'HP EliteBook 1030 G3 not measured',
            'single development JPEG case insufficient for production qualification',
            'checkpoint SHA-256 is run-observed discovery until promoted into the CFS registry',
        ],
        'source': {
            'dataset': 'cfs-face-smartphone-v1 development/calibration source bank',
            'input_sha256': input_sha,
            'final_holdout_used': False,
            'degradation': 'single_jpeg',
            'jpeg_quality': int(args.jpeg_quality),
        },
        'model': {
            'key': candidate.model_key,
            'backend': candidate.backend,
            'official_repository': 'https://github.com/jiaxi-jiang/FBCNN',
            'official_source_commit': args.source_sha,
            'architecture_reimplemented_by_cfs': False,
            'checkpoint_source': 'https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth',
            'checkpoint_bytes': int(args.checkpoint.stat().st_size),
            'checkpoint_sha256_observed': checkpoint_sha,
            'checkpoint_sha256_runtime_expected': args.expected_checkpoint_sha256.lower(),
            'license': 'Apache-2.0',
            'device': 'cpu',
            'torch_version': torch.__version__,
            'requested_threads': int(args.threads),
            'effective_threads': int(effective_threads),
            'predicted_input_quality_factor': predicted_quality,
            'provenance_class': candidate.provenance_class,
            'generated_pixels': int(np.count_nonzero(candidate.generated_mask)),
        },
        'resource_budget': {
            'max_total_pc_fraction': 0.80,
            'max_parallel_heavy_models': 1,
            'before_load': resource_before_load,
            'after_load': resource_after_load,
            'after_inference': resource_after_inference,
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
            'peak_inference': float(infer_sampler.peak / (1024.0 * 1024.0)),
        },
        'metrics': {
            'sface_clean_vs_degraded': identity_degraded,
            'sface_clean_vs_fbcnn': identity_restored,
            'sface_identity_gate_pass': bool(identity_pass),
            'psnr_degraded': common.psnr(clean_aligned, degraded_aligned),
            'psnr_fbcnn': common.psnr(clean_aligned, restored),
            'ssim_degraded': common.ssim_gray(clean_aligned, degraded_aligned),
            'ssim_fbcnn': common.ssim_gray(clean_aligned, restored),
        },
        'outputs': {
            'restored': restored_path.name,
            'restored_sha256': common.sha256_path(restored_path),
            'comparison': 'comparison.png',
            'comparison_restored_label': 'FBCNN OFFICIAL',
        },
        'host': {
            'platform': platform.platform(),
            'python': platform.python_version(),
            'logical_cpu_count': int(__import__('os').cpu_count() or 1),
        },
    }

    backend.unload()
    del backend
    gc.collect()
    assert_memory_within_budget(budget, stage='fbcnn_post_unload')
    report['rss_mb']['post_unload_gc'] = common._rss_mb()
    report['resource_budget']['post_unload'] = resource_snapshot(budget)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')

    if not identity_pass:
        raise RuntimeError(
            f'FBCNN identity gate failed: {identity_restored:.6f} < '
            f'{FACE_MODEL_DEFAULTS.sface_same_identity_cosine:.6f}'
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
