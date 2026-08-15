from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from app.reference_memory import specific_reference_memory_fusion
from app.restoration import DeblurSettings, conservative_deblur, conservative_upscale, detect_occlusion_candidates, quality_enhance


@dataclass(frozen=True)
class BenchmarkResult:
    width: int
    height: int
    deblur_ms: float
    enhance_ms: float
    occlusion_ms: float
    reference_memory_ms: float
    upscale2_ms: float
    total_ms: float


def _synthetic(width: int, height: int) -> np.ndarray:
    rng = np.random.default_rng(12345)
    base = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(30, 220, width, dtype=np.uint8)
    base[:] = gradient[None, :, None]
    noise = rng.normal(0, 8, base.shape).astype(np.int16)
    result = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.ellipse(result, (width // 2, height // 2), (max(20, width // 6), max(20, height // 4)), 0, 0, 360, (130, 160, 190), -1)
    return result


def _time_ms(callable_):
    start = time.perf_counter()
    value = callable_()
    return value, (time.perf_counter() - start) * 1000.0


def _memory_geometry(width: int, height: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    face_w = max(32, int(width * 0.34))
    face_h = max(40, int(height * 0.58))
    x = max(0, width // 2 - face_w // 2)
    y = max(0, height // 2 - face_h // 2)
    landmarks = np.array(
        [
            [x + face_w * 0.34, y + face_h * 0.36],
            [x + face_w * 0.66, y + face_h * 0.36],
            [x + face_w * 0.50, y + face_h * 0.53],
            [x + face_w * 0.40, y + face_h * 0.72],
            [x + face_w * 0.60, y + face_h * 0.72],
        ],
        dtype=np.float32,
    )
    return landmarks, (x, y, min(face_w, width - x), min(face_h, height - y))


def run_cpu_benchmark(width: int = 768, height: int = 512) -> BenchmarkResult:
    if width < 64 or height < 64:
        raise ValueError("Benchmark dimensions must be at least 64x64")
    image = _synthetic(width, height)
    start = time.perf_counter()
    deblurred, deblur_ms = _time_ms(lambda: conservative_deblur(image, DeblurSettings(denoise=4, sharpen=0.2, contrast=1.0)))
    enhanced, enhance_ms = _time_ms(lambda: quality_enhance(deblurred))
    occlusion, occlusion_ms = _time_ms(lambda: detect_occlusion_candidates(enhanced))

    landmarks, bbox = _memory_geometry(width, height)
    ref_a = image.copy()
    ref_b = cv2.convertScaleAbs(image, alpha=1.0, beta=2)
    zero = np.zeros((height, width), dtype=np.uint8)
    _, reference_memory_ms = _time_ms(
        lambda: specific_reference_memory_fusion(
            [enhanced, ref_a, ref_b],
            [occlusion, zero, zero],
            landmarks,
            bbox,
        )
    )
    _, upscale_ms = _time_ms(lambda: conservative_upscale(enhanced, 2))
    total_ms = (time.perf_counter() - start) * 1000.0
    return BenchmarkResult(
        width=width,
        height=height,
        deblur_ms=round(deblur_ms, 3),
        enhance_ms=round(enhance_ms, 3),
        occlusion_ms=round(occlusion_ms, 3),
        reference_memory_ms=round(reference_memory_ms, 3),
        upscale2_ms=round(upscale_ms, 3),
        total_ms=round(total_ms, 3),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU smoke benchmark for ConservativeFaceStudio")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_cpu_benchmark(args.width, args.height)
    payload = json.dumps(asdict(result), indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
