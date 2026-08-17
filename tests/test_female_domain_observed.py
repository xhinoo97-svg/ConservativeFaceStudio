from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import run_female_domain_benchmark_observed as observed


class _FakeBackend:
    def analyze(self, image):
        return SimpleNamespace(
            landmarks5=np.array(
                [[90.0, 95.0], [165.0, 92.0], [128.0, 130.0], [105.0, 165.0], [151.0, 164.0]],
                dtype=np.float32,
            )
        )


def test_curated_domain_is_broad_not_nasa_only():
    assert len(observed.CURATED_FEMALE_DOMAIN) >= 30
    notes = " ".join(item.domain_note.lower() for item in observed.CURATED_FEMALE_DOMAIN[:30])
    assert "vintage" in notes
    assert "19th-century" in notes
    assert "eyewear" in notes
    assert sum("nasa" in item.search_query.lower() for item in observed.CURATED_FEMALE_DOMAIN[:30]) < 15


def test_component_damage_is_exact_union_of_observed_partial_references(monkeypatch):
    monkeypatch.setattr(observed, "_LANDMARK_BACKEND", _FakeBackend())
    clean = np.full((256, 256, 3), 127, dtype=np.uint8)
    fallback = observed.Scenario(
        "component_only_references",
        clean.copy(),
        (clean.copy(),),
        np.zeros((256, 256), dtype=np.uint8),
        True,
    )
    scenario = observed._observed_component_scenario(clean, fallback)
    assert scenario is not fallback
    assert len(scenario.references) == 3
    support = np.any(np.stack([np.any(ref != 0, axis=2) for ref in scenario.references]), axis=0)
    damage = scenario.damage_mask > 0
    assert np.array_equal(support, damage)
    assert np.all(scenario.primary[damage] == 18)
    assert np.array_equal(scenario.primary[~damage], clean[~damage])


def test_quick_profile_stays_five_cases_and_alternates_blur_with_mosaic(monkeypatch):
    monkeypatch.setattr(observed, "_LANDMARK_BACKEND", None)
    clean = np.full((256, 256, 3), 127, dtype=np.uint8)

    odd = observed._observed_make_scenarios(clean, seed=20260809, profile="quick")
    even = observed._observed_make_scenarios(clean, seed=20260810, profile="quick")
    odd_names = {item.name for item in odd}
    even_names = {item.name for item in even}

    assert len(odd) == 5
    assert len(even) == 5
    assert "mosaic_single" in odd_names
    assert "gaussian_heavy_single" not in odd_names
    assert "gaussian_heavy_single" in even_names
    assert "mosaic_single" not in even_names
    common = {
        "opaque_sticker_single",
        "opaque_sticker_full_reference",
        "scribble_two_partial",
        "component_only_references",
    }
    assert common.issubset(odd_names)
    assert common.issubset(even_names)


def test_no_evidence_identity_guardrail_is_recorded_as_abstention_not_runtime_error(tmp_path: Path):
    report = {
        "cases": [
            {
                "portrait": "example",
                "scenario": "opaque_sticker_single",
                "error": "Controllo identità SFace sotto soglia: 0.290 < 0.363",
            },
            {
                "portrait": "broken",
                "scenario": "opaque_sticker_full_reference",
                "error": "unexpected runtime failure",
            },
        ],
        "summary": {"error_cases": 2},
    }
    observed._postprocess_guardrail_abstentions(report, tmp_path)
    guardrail = report["cases"][0]
    assert guardrail["abstained"] is True
    assert guardrail["abstention_reason"] == "identity_guardrail_low_evidence"
    assert guardrail["target95_applicable"] is False
    assert "error" not in guardrail
    assert report["summary"]["identity_guardrail_abstention_count"] == 1
    assert report["summary"]["error_cases"] == 1
    assert report["cases"][1]["error"] == "unexpected runtime failure"


def test_v4_fail_closed_identity_message_abstains_only_on_no_evidence_scenario(tmp_path: Path):
    report = {
        "cases": [{
            "portrait": "example",
            "scenario": "opaque_sticker_single",
            "error": "Controllo identità senza confronto SFace reale: il fallback proxy non è autorità V4",
        }],
        "summary": {"error_cases": 1},
    }
    observed._postprocess_guardrail_abstentions(report, tmp_path)
    row = report["cases"][0]
    assert row["abstained"] is True
    assert row["abstention_reason"] == "identity_guardrail_low_evidence"
    assert "error" not in row
    assert report["summary"]["error_cases"] == 0


def test_identity_failure_on_recoverable_mosaic_remains_runtime_error(tmp_path: Path):
    report = {
        "cases": [{
            "portrait": "example",
            "scenario": "mosaic_single",
            "error": "Controllo identità SFace sotto soglia: 0.290 < 0.363",
        }],
        "summary": {"error_cases": 1},
    }
    observed._postprocess_guardrail_abstentions(report, tmp_path)
    row = report["cases"][0]
    assert row["error"].startswith("Controllo identità SFace sotto soglia")
    assert row.get("abstained") is not True
    assert report["summary"]["identity_guardrail_abstention_count"] == 0
    assert report["summary"]["error_cases"] == 1


def test_unrelated_female_domain_runtime_error_never_becomes_abstention(tmp_path: Path):
    report = {
        "cases": [{
            "portrait": "example",
            "scenario": "opaque_sticker_single",
            "error": "unexpected tensor shape",
        }],
        "summary": {"error_cases": 1},
    }
    observed._postprocess_guardrail_abstentions(report, tmp_path)
    assert report["cases"][0]["error"] == "unexpected tensor shape"
    assert report["summary"]["identity_guardrail_abstention_count"] == 0
    assert report["summary"]["error_cases"] == 1
