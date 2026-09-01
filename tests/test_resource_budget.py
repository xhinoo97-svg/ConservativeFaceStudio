from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / 'app' / 'resource_budget.py'
SPEC = importlib.util.spec_from_file_location('cfs_resource_budget_under_test', MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

DEFAULT_MAX_RESOURCE_FRACTION = module.DEFAULT_MAX_RESOURCE_FRACTION
ResourceBudgetExceeded = module.ResourceBudgetExceeded
assert_memory_within_budget = module.assert_memory_within_budget
detect_resource_budget = module.detect_resource_budget
select_windows_affinity_mask = module._select_windows_affinity_mask


def test_default_budget_never_exceeds_eighty_percent() -> None:
    budget = detect_resource_budget()
    assert budget.max_fraction == DEFAULT_MAX_RESOURCE_FRACTION == 0.80
    assert budget.max_parallel_heavy_models == 1
    assert 1 <= budget.allowed_processors <= budget.logical_processors
    assert budget.allowed_processors <= max(1, int(budget.logical_processors * 0.80))
    if budget.total_ram_bytes is not None:
        assert budget.process_ram_limit_bytes == int(budget.total_ram_bytes * 0.80)
        assert budget.system_ram_limit_bytes == int(budget.total_ram_bytes * 0.80)


def test_budget_rejects_fraction_above_eighty_percent() -> None:
    with pytest.raises(ValueError):
        detect_resource_budget(0.81)


def test_budget_rejects_too_small_fraction() -> None:
    with pytest.raises(ValueError):
        detect_resource_budget(0.09)


def test_windows_affinity_narrows_existing_non_contiguous_mask_without_inventing_cpus() -> None:
    # Hosted runners/job objects may grant CPUs 2, 4 and 7 rather than 0, 1 and 2.
    current = (1 << 2) | (1 << 4) | (1 << 7)
    selected = select_windows_affinity_mask(current, 2)
    assert selected == ((1 << 2) | (1 << 4))
    assert selected & ~current == 0
    assert selected.bit_count() == 2


def test_windows_affinity_keeps_stricter_host_mask() -> None:
    current = (1 << 5) | (1 << 9)
    selected = select_windows_affinity_mask(current, 8)
    assert selected == current


def test_windows_affinity_rejects_empty_mask() -> None:
    with pytest.raises(ValueError, match='non-zero'):
        select_windows_affinity_mask(0, 2)


def test_process_memory_reservation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = replace(
        detect_resource_budget(),
        total_ram_bytes=1000,
        process_ram_limit_bytes=800,
        system_ram_limit_bytes=800,
    )
    monkeypatch.setattr(module, '_process_rss_bytes', lambda: 700)
    monkeypatch.setattr(module, '_physical_memory_status_bytes', lambda: (1000, 900))
    with pytest.raises(ResourceBudgetExceeded, match='Process RAM budget exceeded'):
        assert_memory_within_budget(budget, stage='model_load', reserve_bytes=101)


def test_process_memory_reservation_accepts_exact_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = replace(
        detect_resource_budget(),
        total_ram_bytes=1000,
        process_ram_limit_bytes=800,
        system_ram_limit_bytes=800,
    )
    monkeypatch.setattr(module, '_process_rss_bytes', lambda: 700)
    monkeypatch.setattr(module, '_physical_memory_status_bytes', lambda: (1000, 900))
    assert_memory_within_budget(budget, stage='model_load', reserve_bytes=100)


def test_system_memory_reservation_fails_when_total_pc_use_would_exceed_eighty_percent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = replace(
        detect_resource_budget(),
        total_ram_bytes=1000,
        process_ram_limit_bytes=800,
        system_ram_limit_bytes=800,
    )
    monkeypatch.setattr(module, '_process_rss_bytes', lambda: 100)
    # 750 bytes are already used system-wide; another 51 would exceed the 800-byte cap.
    monkeypatch.setattr(module, '_physical_memory_status_bytes', lambda: (1000, 250))
    with pytest.raises(ResourceBudgetExceeded, match='System RAM budget exceeded'):
        assert_memory_within_budget(budget, stage='model_load', reserve_bytes=51)


def test_system_memory_exact_eighty_percent_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = replace(
        detect_resource_budget(),
        total_ram_bytes=1000,
        process_ram_limit_bytes=800,
        system_ram_limit_bytes=800,
    )
    monkeypatch.setattr(module, '_process_rss_bytes', lambda: 100)
    monkeypatch.setattr(module, '_physical_memory_status_bytes', lambda: (1000, 250))
    assert_memory_within_budget(budget, stage='model_load', reserve_bytes=50)
