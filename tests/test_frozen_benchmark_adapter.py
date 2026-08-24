from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_face_smartphone_baseline as baseline
from scripts.frozen_benchmark_adapter import (
    build_freeze_payload,
    prepare_frozen_benchmark,
)


def test_build_freeze_supports_legacy_cases_only_signature() -> None:
    received: list[dict] = []

    class Provider:
        @staticmethod
        def build_freeze(cases_payload: dict) -> dict:
            received.append(cases_payload)
            return {"kind": "legacy"}

    cases = {"benchmark_id": "dev", "cases": []}
    result = build_freeze_payload(Provider, cases, {"contract": "loaded"})

    assert result == {"kind": "legacy"}
    assert received == [cases]


def test_build_freeze_passes_cases_and_contract_to_generic_signature() -> None:
    received: list[tuple[dict, dict]] = []

    class Provider:
        @staticmethod
        def build_freeze(cases_payload: dict, contract_payload: dict) -> dict:
            received.append((cases_payload, contract_payload))
            return {"kind": "generic"}

    cases = {"benchmark_id": "dev", "cases": []}
    contract = {"benchmark_id": "dev", "case_count": 0}
    result = build_freeze_payload(Provider, cases, contract)

    assert result == {"kind": "generic"}
    assert received == [(cases, contract)]


def test_provider_type_error_is_not_interpreted_as_signature_mismatch() -> None:
    calls = 0

    class Provider:
        @staticmethod
        def build_freeze(cases_payload: dict, contract_payload: dict) -> dict:
            nonlocal calls
            calls += 1
            raise TypeError("provider implementation failed")

    with pytest.raises(TypeError, match="provider implementation failed"):
        build_freeze_payload(Provider, {"cases": []}, {"case_count": 0})

    assert calls == 1


def test_incompatible_build_freeze_signature_is_rejected_without_execution() -> None:
    calls = 0

    class Provider:
        @staticmethod
        def build_freeze(cases_payload: dict, contract_payload: dict, extra: dict) -> dict:
            nonlocal calls
            calls += 1
            return {}

    with pytest.raises(TypeError, match="must accept"):
        build_freeze_payload(Provider, {"cases": []}, {"case_count": 0})

    assert calls == 0


def test_prepare_frozen_benchmark_loads_contract_before_two_argument_freeze(tmp_path: Path) -> None:
    contract = {"benchmark_id": "dev", "case_count": 1}
    (tmp_path / "contract.json").write_text(json.dumps(contract), encoding="utf-8")

    class Provider:
        BENCHMARK_ROOT = tmp_path

        @staticmethod
        def build_cases() -> dict:
            return {"benchmark_id": "dev", "cases": [{"case_id": "dev-001"}]}

        @staticmethod
        def build_freeze(cases_payload: dict, contract_payload: dict) -> dict:
            assert contract_payload == contract
            return {"case_count": len(cases_payload["cases"])}

    prepared = prepare_frozen_benchmark(Provider)

    assert prepared.contract_payload == contract
    assert prepared.freeze_payload == {"case_count": 1}


def test_real_certification_entrypoint_accepts_two_argument_freeze_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    contract = {"benchmark_id": "dev-entrypoint", "case_count": 1}
    (benchmark_root / "contract.json").write_text(json.dumps(contract), encoding="utf-8")
    (benchmark_root / "sources.json").write_text(
        json.dumps({"sources": [{"source_id": "synthetic-source"}]}),
        encoding="utf-8",
    )
    freeze_calls: list[tuple[dict, dict]] = []
    case = {
        "case_id": "dev-entrypoint-001",
        "calibration_or_holdout": "calibration",
        "damage_type": "synthetic",
        "damage_style": "synthetic",
        "face_overlap_ratio": 1.0,
        "reference_ids": [],
        "target95_applicable_pre_score": True,
    }

    class Provider:
        BENCHMARK_ROOT = benchmark_root

        @staticmethod
        def build_cases() -> dict:
            return {"benchmark_id": "dev-entrypoint", "cases": [case]}

        @staticmethod
        def build_freeze(cases_payload: dict, contract_payload: dict) -> dict:
            freeze_calls.append((cases_payload, contract_payload))
            return {"benchmark_id": "dev-entrypoint", "case_count": 1}

    source_file = tmp_path / "source.bin"
    source_file.write_bytes(b"synthetic source")
    model_file = tmp_path / "model.bin"
    model_file.write_bytes(b"synthetic model")
    clean = np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(baseline, "freeze", Provider)
    monkeypatch.setattr(baseline, "acquire_sources", lambda cache, offline=False: {"synthetic-source": source_file})
    monkeypatch.setattr(baseline, "load_clean_images", lambda paths: {"synthetic-source": clean})
    monkeypatch.setattr(baseline, "production_model_paths", lambda root: {"synthetic-model": model_file})
    monkeypatch.setattr(baseline, "materialize_scenario", lambda item, sources, images: (clean, object()))
    monkeypatch.setattr(
        baseline,
        "evaluate_scenario",
        lambda ground_truth, scenario, output, core_paths: {
            "conservative_recovery_score": 100.0,
            "identity_similarity": 1.0,
            "outside_region_mae": 0.0,
            "target95_passed": True,
            "primary_fraction": 1.0,
            "reference_fraction": 0.0,
            "symmetry_fraction": 0.0,
            "generated_fraction": 0.0,
        },
    )

    report = baseline.run_baseline(
        tmp_path / "output",
        cache=tmp_path / "cache",
        model_root=tmp_path,
    )

    assert freeze_calls == [({"benchmark_id": "dev-entrypoint", "cases": [case]}, contract)]
    assert report["summary"]["completed_cases"] == 1
    assert report["summary"]["error_cases"] == 0
    assert report["benchmark_freeze"]["case_count"] == 1
