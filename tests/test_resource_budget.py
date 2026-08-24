from __future__ import annotations

from dataclasses import replace

import pytest

from app.resource_budget import (
    DEFAULT_MAX_RESOURCE_FRACTION,
    ResourceBudgetExceeded,
    assert_memory_within_budget,
    detect_resource_budget,
)


def test_default_budget_never_exceeds_eighty_percent() -> None:
    budget = detect_resource_budget()
    assert budget.max_fraction == DEFAULT_MAX_RESOURCE_FRACTION == 0.80
    assert budget.max_parallel_heavy_models == 1
    assert 1 <= budget.allowed_processors <= budget.logical_processors
    assert budget.allowed_processors <= max(1, int(budget.logical_processors * 0.80))
    if budget.total_ram_bytes is not None:
        assert budget.process_ram_limit_bytes == int(budget.total_ram_bytes * 0.80)


def test_budget_rejects_fraction_above_eighty_percent() -> None:
    with pytest.raises(ValueError):
        detect_resource_budget(0.81)


def test_budget_rejects_too_small_fraction() -> None:
    with pytest.raises(ValueError):
        detect_resource_budget(0.09)


def test_memory_reservation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.resource_budget as module

    budget = replace(
        detect_resource_budget(),
        total_ram_bytes=1000,
        process_ram_limit_bytes=800,
    )
    monkeypatch.setattr(module, "_process_rss_bytes", lambda: 700)
    with pytest.raises(ResourceBudgetExceeded):
        assert_memory_within_budget(budget, stage="model_load", reserve_bytes=101)


def test_memory_reservation_accepts_exact_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.resource_budget as module

    budget = replace(
        detect_resource_budget(),
        total_ram_bytes=1000,
        process_ram_limit_bytes=800,
    )
    monkeypatch.setattr(module, "_process_rss_bytes", lambda: 700)
    assert_memory_within_budget(budget, stage="model_load", reserve_bytes=100)
