from __future__ import annotations

import json

import numpy as np

from scripts import freeze_face_smartphone_benchmark as freeze
from scripts import run_face_smartphone_baseline as baseline


def _fixture_images() -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    payload = json.loads((freeze.BENCHMARK_ROOT / "sources.json").read_text(encoding="utf-8"))
    sources = {item["source_id"]: item for item in payload["sources"]}
    clean = {
        key: np.full((freeze.CANVAS_SIZE, freeze.CANVAS_SIZE, 3), 80 + index, dtype=np.uint8)
        for index, key in enumerate(sources)
    }
    return sources, clean


def test_every_frozen_case_materializes_without_changing_source0_canvas() -> None:
    sources, clean = _fixture_images()
    for case in freeze.build_cases()["cases"]:
        ground_truth, scenario = baseline.materialize_scenario(case, sources, clean)
        assert ground_truth.shape == scenario.primary.shape == (freeze.CANVAS_SIZE, freeze.CANVAS_SIZE, 3)
        assert len(scenario.references) == len(case["reference_ids"])
        assert np.count_nonzero(scenario.damage_mask) > 0
        assert np.array_equal(ground_truth, clean[case["main_source_id"]])


def test_wrong_person_and_useless_references_do_not_claim_main_identity() -> None:
    sources, clean = _fixture_images()
    cases = freeze.build_cases()["cases"]
    case = next(item for item in cases if any("wrong_person" in ref for ref in item["reference_ids"]))
    ground_truth, scenario = baseline.materialize_scenario(case, sources, clean)
    assert any(not np.array_equal(reference, ground_truth) for reference in scenario.references)
    assert case["main_contract"] == "SOURCE0_IMMUTABLE_TARGET_CANVAS"


def test_rendered_mask_checksum_and_face_overlap_match_frozen_manifest() -> None:
    payload = json.loads((freeze.BENCHMARK_ROOT / "sources.json").read_text(encoding="utf-8"))
    sources = {item["source_id"]: item for item in payload["sources"]}
    clean = np.full((freeze.CANVAS_SIZE, freeze.CANVAS_SIZE, 3), 127, dtype=np.uint8)
    for case in freeze.build_cases()["cases"]:
        _, mask = baseline.render_primary(clean, sources[case["main_source_id"]], case)
        face = freeze._face_domain_mask(sources[case["main_source_id"]])
        overlap = np.count_nonzero((mask > 0) & (face > 0)) / np.count_nonzero(mask)
        assert overlap == case["face_overlap_ratio"]


def test_targeted_case_filter_is_exact_and_split_scoped() -> None:
    cases = freeze.build_cases()["cases"]
    target = cases[0]

    selected = baseline.select_cases(cases, target["calibration_or_holdout"], {target["case_id"]})

    assert [item["case_id"] for item in selected] == [target["case_id"]]
    with np.testing.assert_raises_regex(ValueError, "not found in selected split"):
        baseline.select_cases(cases, target["calibration_or_holdout"], {"not-a-frozen-case"})


def test_peak_rss_uses_windows_fallback_without_posix_resource(monkeypatch) -> None:
    monkeypatch.setattr(baseline, "_resource", None)
    monkeypatch.setattr(baseline.sys, "platform", "win32")
    monkeypatch.setattr(baseline, "_windows_peak_rss_mib", lambda: 123.5)

    assert baseline._peak_rss_mib() == 123.5
