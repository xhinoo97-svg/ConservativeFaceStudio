from __future__ import annotations

from app.benchmark import run_cpu_benchmark


def test_cpu_benchmark_returns_positive_timings() -> None:
    result = run_cpu_benchmark(96, 96)
    assert result.width == 96
    assert result.height == 96
    assert result.deblur_ms >= 0
    assert result.enhance_ms >= 0
    assert result.occlusion_ms >= 0
    assert result.reference_memory_ms >= 0
    assert result.upscale2_ms >= 0
    assert result.total_ms >= result.deblur_ms
