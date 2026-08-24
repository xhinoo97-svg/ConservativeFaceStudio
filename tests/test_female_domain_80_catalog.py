from __future__ import annotations

from scripts.run_female_domain_benchmark_80 import CURATED_80_PLUS


def test_curated_domain_contains_at_least_eighty_unique_portraits() -> None:
    keys = [item.key for item in CURATED_80_PLUS]
    assert len(keys) >= 80
    assert len(keys) == len(set(keys))
