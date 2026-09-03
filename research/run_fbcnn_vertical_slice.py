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
from fbcnn_degradation_matrix import (
    FBCNN_DEVELOPMENT_PROFILES,
    FBCNN_PROFILE_BY_ID,
    disposition_for_metrics,
    materialize_degradation,
)

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


def write_comparison(path: Path, clean: np.ndarray, degraded: np.ndarray, restored: np.ndarray, label: str) -> None:
    panels: list[np.ndarray] = []
    for label, image in (
        ('CLEAN GT', clean),
        (label, degraded),
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
    parser.add_argument(
        '--degradation-profile',
        choices=tuple(item.profile_id for item in FBCNN_DEVELOPMENT_PROFILES),
        default='jpeg-qf20',
    )
    parser.add_argument('--core-model-root', type=Path)
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
    profile = FBCNN_PROFILE_BY_ID[args.degradation_profile]
    degraded = materialize_degradation(clean, profile)
    args.output.mkdir(parents=True, exist_ok=True)

    core_root = args.core_model_root or (args.output / 'cfs-core-models')
    core = ensure_core_pretrained_models(core_root, timeout_seconds=60)
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
        damage_class=profile.damage_class,
        severity='heavy',
        metadata={'jpeg_detected': True, 'degradation_profile': profile.profile_id},
    )

    # FBCNN must see the JPEG-damaged pixels before any affine warp/resampling.
    # Alignment interpolates pixels and disrupts the block/ringing structure that
    # the compression specialist was trained to remove. The aligned crop is used
    # only after restoration so all quality/identity metrics remain comparable.
    _warm = backend.restore(degraded, context)
    assert_memory_within_budget(budget, stage='fbcnn_post_warmup')
    infer_start = time.perf_counter()
    with common.PeakRSSSampler() as infer_sampler:
        candidate = backend.restore(degraded, context)
    infer_seconds = time.perf_counter() - infer_start
    assert_memory_within_budget(budget, stage='fbcnn_post_inference')
    resource_after_inference = resource_snapshot(budget)

    restored_source = candidate.image
    if restored_source.shape != degraded.shape or restored_source.dtype != np.uint8:
        raise RuntimeError(f'Invalid FBCNN output: {restored_source.shape} {restored_source.dtype}')
    if candidate.provenance_class != GENERATED_MODEL_INFERRED:
        raise RuntimeError(f'Invalid FBCNN provenance: {candidate.provenance_class}')
    if candidate.model_version != PINNED_REVISION:
        raise RuntimeError(f'FBCNN candidate revision drift: {candidate.model_version}')

    restored = exact_phase3_alignment(restored_source, main_obs.landmarks5)
    if restored.shape != degraded_aligned.shape:
        raise RuntimeError(f'Aligned FBCNN output mismatch: {restored.shape} != {degraded_aligned.shape}')

    restored_path = args.output / 'restored_fbcnn_color.png'
    cv2.imwrite(str(restored_path), restored)
    write_comparison(args.output / 'comparison.png', clean_aligned, degraded_aligned, restored, profile.label)

    clean_obs = engine.analyze(clean_aligned)
    degraded_obs = engine.analyze(degraded_aligned)
    restored_obs = engine.analyze(restored)
    if clean_obs.embedding is None or degraded_obs.embedding is None or restored_obs.embedding is None:
        raise RuntimeError('SFace embedding missing')
    identity_degraded = float(cosine_similarity(clean_obs.embedding, degraded_obs.embedding))
    identity_restored = float(cosine_similarity(clean_obs.embedding, restored_obs.embedding))
    predicted_quality = float(candidate.quality_metrics['predicted_jpeg_quality_factor'])
    metrics = {
        'identity_threshold': float(FACE_MODEL_DEFAULTS.sface_same_identity_cosine),
        'sface_clean_vs_degraded': identity_degraded,
        'sface_clean_vs_fbcnn': identity_restored,
        'sface_identity_gate_pass': bool(identity_restored >= FACE_MODEL_DEFAULTS.sface_same_identity_cosine),
        'psnr_degraded': common.psnr(clean_aligned, degraded_aligned),
        'psnr_fbcnn': common.psnr(clean_aligned, restored),
        'ssim_degraded': common.ssim_gray(clean_aligned, degraded_aligned),
        'ssim_fbcnn': common.ssim_gray(clean_aligned, restored),
        'outside_region_mae': None,
        'outside_region_policy': 'NOT_APPLICABLE_FULL_ALIGNED_FACE_SPECIALIST',
        'wrong_person_final_pixels': 0,
        'provenance_valid': bool(candidate.provenance_class == GENERATED_MODEL_INFERRED),
    }
    disposition = disposition_for_metrics(metrics)
    final = restored if disposition['decision'] == 'PASS' else degraded_aligned
    final_path = args.output / 'final.png'
    if not cv2.imwrite(str(final_path), final):
        raise RuntimeError('Failed to save FBCNN final disposition image')
    report = {
        'experiment': 'fbcnn_compression_specialist_development_matrix_v1',
        'qualification_scope': 'development_host_cpu_only',
        'production_qualified': False,
        'production_blockers': [
            'Windows installer execution not tested',
            'HP EliteBook 1030 G3 not measured',
            'single public development identity insufficient for production qualification',
            'identity-disjoint multi-identity validation not run',
        ],
        'source': {
            'dataset': 'cfs-face-smartphone-v1 development/calibration source bank',
            'input_sha256': input_sha,
            'final_holdout_used': False,
            'degradation': profile.damage_class,
            'degradation_profile': profile.profile_id,
            'degradation_family': profile.family,
            'profile_contract': {
                'label': profile.label,
                'first_quality': profile.first_quality,
                'second_quality': profile.second_quality,
                'resize_scale': profile.resize_scale,
            },
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
            'execution_order': 'JPEG_DEGRADATION -> FBCNN -> METRIC_ALIGNMENT',
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
            'alignment': 'exact Phase-3 GPEN official 5-point 512 convention applied after FBCNN for metrics',
            'main_yunet_score': float(main_obs.score),
        },
        'timing_seconds': {
            'model_load': float(load_seconds),
            'inference_source_resolution_measured_after_warmup': float(infer_seconds),
        },
        'rss_mb': {
            'baseline': baseline,
            'peak_model_load': float(load_sampler.peak / (1024.0 * 1024.0)),
            'peak_inference': float(infer_sampler.peak / (1024.0 * 1024.0)),
        },
        'metrics': metrics,
        'disposition': disposition,
        'provenance': {
            'candidate': candidate.provenance_class,
            'final': candidate.provenance_class if disposition['decision'] == 'PASS' else 'PRIMARY_OBSERVED',
            'wrong_person_final_pixels': 0,
            'violations': 0,
        },
        'outputs': {
            'restored': restored_path.name,
            'restored_sha256': common.sha256_path(restored_path),
            'final': final_path.name,
            'final_sha256': common.sha256_path(final_path),
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

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
