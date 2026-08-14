from __future__ import annotations

"""Build or verify the frozen independent 40-case final holdout."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import freeze_face_smartphone_benchmark as v1

BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v3-final-holdout"
V1_ROOT = REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v1"
V2_ROOT = REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v2-final-holdout"
CANVAS_SIZE = 384
BASE_SEED = 202608142300
BENCHMARK_ID = "cfs-face-smartphone-v3-final-holdout"

# Export the exact frozen v1 primitives so the existing renderer can use this module.
_fitted_face_bbox = v1._fitted_face_bbox
_face_domain_mask = v1._face_domain_mask
_raw_mask = v1._raw_mask
_reference_ids = v1._reference_ids
_case_severity = v1._case_severity
STYLES = v1.STYLES
REGION_COMPONENTS = v1.REGION_COMPONENTS

CATEGORY_COUNTS = {
    "opaque_graphic_face_occlusion": 10,
    "face_blur_censor": 8,
    "face_mosaic_pixelation": 8,
    "black_paint_scribble": 4,
    "natural_facial_occluder": 4,
    "mixed_smartphone": 4,
    "extreme_low_evidence": 2,
}


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _assert_independent(sources: list[dict[str, Any]]) -> dict[str, str]:
    old_payloads = {"v1": _load(V1_ROOT / "sources.json"), "v2": _load(V2_ROOT / "sources.json")}
    old_ids = {str(item["source_id"]) for payload in old_payloads.values() for item in payload["sources"]}
    old_hashes = {str(item["clean_source_sha256"]) for payload in old_payloads.values() for item in payload["sources"]}
    new_ids = {str(item["source_id"]) for item in sources}
    new_hashes = {str(item["clean_source_sha256"]) for item in sources}
    if new_ids & old_ids:
        raise RuntimeError(f"Final holdout source-id overlap with consumed data: {sorted(new_ids & old_ids)!r}")
    if new_hashes & old_hashes:
        raise RuntimeError("Final holdout clean-image checksum overlap with consumed data")
    frozen = {
        "v1": str(_load(V1_ROOT / "freeze.json").get("sources_manifest_sha256", "")),
        "v2": _sha(_canonical_json(old_payloads["v2"])),
    }
    if any(len(value) != 64 for value in frozen.values()):
        raise RuntimeError("Consumed frozen source-manifest checksum missing")
    return frozen


def build_cases() -> dict[str, Any]:
    sources = list(_load(BENCHMARK_ROOT / "sources.json")["sources"])
    if len(sources) != 4:
        raise RuntimeError(f"Expected exactly four new final-holdout identities, got {len(sources)}")
    _assert_independent(sources)
    by_id = {str(item["source_id"]): item for item in sources}
    source_ids = list(by_id)
    cases: list[dict[str, Any]] = []
    global_index = 0

    for category, count in CATEGORY_COUNTS.items():
        styles = STYLES[category]
        for category_index in range(count):
            source_id = source_ids[global_index % len(source_ids)]
            source = by_id[source_id]
            style, shape, region = styles[category_index % len(styles)]
            seed = BASE_SEED + global_index
            mask = _raw_mask(source, shape, region, seed)
            face = _face_domain_mask(source)
            pixels = int(np.count_nonzero(mask))
            overlap = float(np.count_nonzero((mask > 0) & (face > 0)) / max(1, pixels))
            extreme = category == "extreme_low_evidence"
            reference_count = category_index % 3 if extreme else global_index % 10
            source_pos = source_ids.index(source_id)
            wrong_source_id = source_ids[(source_pos + 1) % len(source_ids)]
            references = _reference_ids(source_id, wrong_source_id, reference_count, extreme=extreme)
            target95 = bool(
                not extreme
                and any(ref.startswith(f"{source_id}:") and "useless" not in ref for ref in references)
            )
            cases.append({
                "case_id": f"cfsfs3-fin-{global_index + 1:03d}-{style}",
                "calibration_or_holdout": "final_holdout",
                "clean_source_checksum": source["clean_source_sha256"],
                "damage_mask_checksum": _sha(mask.tobytes(order="C")),
                "damage_mask_encoding": f"uint8_{CANVAS_SIZE}x{CANVAS_SIZE}_raw",
                "damage_seed": seed,
                "damage_severity": _case_severity(category, category_index),
                "damage_style": style,
                "damage_type": category,
                "evaluated_face_regions": REGION_COMPONENTS[region],
                "face_overlap_ratio": round(overlap, 8),
                "main_contract": "SOURCE0_IMMUTABLE_TARGET_CANVAS",
                "main_source_id": source_id,
                "mask_region": region,
                "mask_shape": shape,
                "primary_face_case": True,
                "provenance_contract": "GENERATED_OR_RESTORED_NEVER_BECOMES_OBSERVED_EVIDENCE",
                "recoverability_pre_score": "LOW_EVIDENCE_ABSTAIN" if extreme else (
                    "REFERENCE_RECOVERABLE" if target95 else "SEVERE_RECOVERABLE"
                ),
                "reference_ids": references,
                "target95_applicable_pre_score": target95,
                "target95_policy": "REPORT_ONLY",
                "wrong_person_source_ids": sorted({
                    ref.split(":", 1)[0] for ref in references if ref.split(":", 1)[0] != source_id
                }),
            })
            global_index += 1

    counts = {source_id: sum(c["main_source_id"] == source_id for c in cases) for source_id in source_ids}
    if global_index != 40 or any(value != 10 for value in counts.values()):
        raise RuntimeError(f"Final holdout allocation drift: cases={global_index}, sources={counts!r}")
    return {
        "benchmark_id": BENCHMARK_ID,
        "cases": cases,
        "generation_algorithm": "freeze_face_smartphone_v3_final_holdout.py:v1",
        "mask_checksum_encoding": f"uint8_{CANVAS_SIZE}x{CANVAS_SIZE}_raw",
        "split": "final_holdout",
        "version": 1,
    }


def build_freeze(cases_payload: dict[str, Any]) -> dict[str, Any]:
    sources_payload = _load(BENCHMARK_ROOT / "sources.json")
    old_sources_sha = _assert_independent(list(sources_payload["sources"]))
    sources_bytes = _canonical_json(sources_payload)
    cases_bytes = _canonical_json(cases_payload)
    contract_bytes = _normalized((BENCHMARK_ROOT / "contract.json").read_bytes())
    masks = [{"case_id": c["case_id"], "damage_mask_checksum": c["damage_mask_checksum"]} for c in cases_payload["cases"]]
    refs = [{"case_id": c["case_id"], "reference_ids": c["reference_ids"]} for c in cases_payload["cases"]]
    distribution = {
        category: sum(c["damage_type"] == category for c in cases_payload["cases"])
        for category in CATEGORY_COUNTS
    }
    return {
        "architecture_freeze_base_sha": "368f00c122520f471e7ef310f9daf8781b51f111",
        "benchmark_id": BENCHMARK_ID,
        "cases_manifest_sha256": _sha(cases_bytes),
        "category_distribution": distribution,
        "contract_sha256": _sha(contract_bytes),
        "damage_seed_base": BASE_SEED,
        "frozen_before_candidate_evaluation": True,
        "independent_from_consumed_v1_and_v2_holdouts": True,
        "mask_hashes_sha256": _sha(_canonical_json(masks)),
        "consumed_sources_manifest_sha256": old_sources_sha,
        "production_pipeline_changed_by_freeze": False,
        "reference_assignment_sha256": _sha(_canonical_json(refs)),
        "sources_manifest_sha256": _sha(sources_bytes),
        "split": {"final_holdout": 40},
        "target95_policy": "REPORT_ONLY",
        "version": 1,
    }


def _verify(path: Path, expected: bytes) -> None:
    actual = _normalized(path.read_bytes()) if path.is_file() else b""
    if actual != _normalized(expected):
        raise RuntimeError(f"Frozen final holdout drift: {path.relative_to(REPOSITORY_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    cases = build_cases()
    freeze = build_freeze(cases)
    cases_bytes, freeze_bytes = _canonical_json(cases), _canonical_json(freeze)
    if args.verify:
        _verify(BENCHMARK_ROOT / "cases.json", cases_bytes)
        _verify(BENCHMARK_ROOT / "freeze.json", freeze_bytes)
        print("Face-smartphone v3 independent final holdout freeze: PASS")
        return 0
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "cases.json").write_bytes(cases_bytes)
    (BENCHMARK_ROOT / "freeze.json").write_bytes(freeze_bytes)
    print("Face-smartphone v3 independent final holdout freeze generated: 40 cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
