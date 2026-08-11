from __future__ import annotations

from collections import Counter
import hashlib
import json

from scripts import freeze_face_smartphone_benchmark as freeze


def test_frozen_face_smartphone_manifest_is_exactly_reproducible() -> None:
    cases = freeze.build_cases()
    expected_cases = json.loads((freeze.BENCHMARK_ROOT / "cases.json").read_text(encoding="utf-8"))
    expected_freeze = json.loads((freeze.BENCHMARK_ROOT / "freeze.json").read_text(encoding="utf-8"))
    assert cases == expected_cases
    assert freeze.build_freeze(cases) == expected_freeze


def test_contract_digest_is_independent_of_windows_line_endings() -> None:
    contract = (freeze.BENCHMARK_ROOT / "contract.json").read_bytes()
    canonical_contract = freeze._normalized_text_bytes(contract)
    windows_checkout = canonical_contract.replace(b"\n", b"\r\n")
    expected = json.loads((freeze.BENCHMARK_ROOT / "freeze.json").read_text(encoding="utf-8"))

    assert hashlib.sha256(freeze._normalized_text_bytes(windows_checkout)).hexdigest() == expected["contract_sha256"]


def test_primary_case_distribution_and_split_are_frozen() -> None:
    cases = freeze.build_cases()["cases"]
    assert len(cases) == 100
    assert Counter(item["damage_type"] for item in cases) == Counter(freeze.CATEGORY_COUNTS)
    assert Counter(item["calibration_or_holdout"] for item in cases) == {"calibration": 60, "holdout": 40}
    assert sum(bool(item["primary_face_case"]) for item in cases) / len(cases) >= 0.95
    assert all(item["target95_policy"] == "REPORT_ONLY" for item in cases)


def test_ordinary_masks_are_face_centered_and_not_duplicates() -> None:
    cases = freeze.build_cases()["cases"]
    ordinary = [item for item in cases if item["damage_type"] != "extreme_low_evidence"]
    assert all(item["face_overlap_ratio"] >= 0.95 for item in ordinary)
    assert len({item["damage_mask_checksum"] for item in cases}) >= 90
    covered = {region for item in cases for region in item["evaluated_face_regions"]}
    assert set(json.loads((freeze.BENCHMARK_ROOT / "contract.json").read_text(encoding="utf-8"))["face_domain"]) <= covered


def test_holdout_identities_are_disjoint_and_reference_counts_cover_zero_to_nine() -> None:
    cases = freeze.build_cases()["cases"]
    calibration_sources = {item["main_source_id"] for item in cases if item["calibration_or_holdout"] == "calibration"}
    holdout_sources = {item["main_source_id"] for item in cases if item["calibration_or_holdout"] == "holdout"}
    assert calibration_sources.isdisjoint(holdout_sources)
    assert {len(item["reference_ids"]) for item in cases} == set(range(10))
    assert all(item["main_contract"] == "SOURCE0_IMMUTABLE_TARGET_CANVAS" for item in cases)


def test_sources_have_frozen_rights_checksums_and_dimensions() -> None:
    payload = json.loads((freeze.BENCHMARK_ROOT / "sources.json").read_text(encoding="utf-8"))
    assert len(payload["sources"]) >= 8
    for source in payload["sources"]:
        assert source["page_url"].startswith("https://commons.wikimedia.org/wiki/File:")
        assert source["download_url"].startswith("https://upload.wikimedia.org/")
        assert source["license_url"].startswith("https://creativecommons.org/")
        assert source["redistribution_status"].startswith(("allowed_", "public_domain_"))
        assert len(source["clean_source_sha256"]) == 64
        assert min(source["original_dimensions"]) >= 256
