from __future__ import annotations

"""Discover, build and verify the independent V4 final release holdout.

Benchmark-only tooling. It never imports or modifies production restoration code.
Network access is used only by --discover-sources. Once sources.json is committed,
normal build/verify is fully deterministic and offline.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts import freeze_face_smartphone_benchmark as v1

BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v4-final-holdout"
CONSUMED_ROOTS = {
    "v1": REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v1",
    "v2": REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v2-final-holdout",
    "v3": REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v3-final-holdout",
}
CANVAS_SIZE = 384
BASE_SEED = 202608170001
BENCHMARK_ID = "cfs-face-smartphone-v4-final-holdout"
FEMALE_IDENTITY_COUNT = 19
CONTROL_IDENTITY_COUNT = 1
TOTAL_IDENTITIES = FEMALE_IDENTITY_COUNT + CONTROL_IDENTITY_COUNT
CASE_COUNT = 40

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

FEMALE_CATEGORIES = (
    "Selfies of women",
    "Selfies of women smiling",
    "Selfies of standing women",
    "Selfies of sitting women",
)
CONTROL_CATEGORIES = ("Selfies of men",)
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ConservativeFaceStudio-V4Freeze/1.0 (release benchmark provenance)"

ALLOWED_LICENSE_PREFIXES = (
    "CC BY ",
    "CC BY-SA ",
    "CC0",
    "Public domain",
    "PD-",
)


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _normalized(payload: bytes) -> bytes:
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _api(params: dict[str, str | int]) -> dict[str, Any]:
    query = urlencode({"format": "json", "formatversion": "2", **params})
    request = Request(f"{COMMONS_API}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def _category_files(category: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    continuation: dict[str, Any] = {}
    while True:
        payload = _api({
            "action": "query",
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": 100,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
            **continuation,
        })
        results.extend(payload.get("query", {}).get("pages", []))
        if "continue" not in payload:
            break
        continuation = payload["continue"]
        if len(results) >= 300:
            break
    return results


def _download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _largest_face_bbox(image_bytes: bytes) -> list[float] | None:
    raw = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    if min(h, w) < 320:
        return None
    scale = min(1.0, 1600.0 / max(h, w))
    probe = image if scale == 1.0 else cv2.resize(
        image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    gray = cv2.cvtColor(probe, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(70, 70),
    )
    if len(faces) == 0:
        return None
    px, py, pw, ph = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    if pw * ph < 0.01 * probe.shape[0] * probe.shape[1]:
        return None
    inv = 1.0 / scale
    x0, y0 = px * inv, py * inv
    x1, y1 = (px + pw) * inv, (py + ph) * inv
    margin_x, margin_y = 0.16 * (x1 - x0), 0.22 * (y1 - y0)
    x0, x1 = max(0.0, x0 - margin_x), min(float(w), x1 + margin_x)
    y0, y1 = max(0.0, y0 - margin_y), min(float(h), y1 + margin_y)
    return [round(x0 / w, 6), round(y0 / h, 6), round(x1 / w, 6), round(y1 / h, 6)]


def _ext(meta: dict[str, Any], key: str) -> str:
    item = meta.get(key)
    return str(item.get("value", "")) if isinstance(item, dict) else ""


def _old_identity_evidence() -> tuple[set[str], set[str], set[str], set[str]]:
    ids: set[str] = set()
    hashes: set[str] = set()
    pages: set[str] = set()
    identities: set[str] = set()
    for root in CONSUMED_ROOTS.values():
        path = root / "sources.json"
        if not path.is_file():
            continue
        for item in _load(path).get("sources", []):
            ids.add(str(item.get("source_id", "")))
            hashes.add(str(item.get("clean_source_sha256", "")))
            pages.add(str(item.get("page_url", "")))
            identities.add(str(item.get("identity_key", item.get("author", ""))).casefold())
    return ids, hashes, pages, identities


def _candidate_source(page: dict[str, Any], *, domain: str, ordinal: int) -> dict[str, Any] | None:
    title = str(page.get("title", ""))
    infos = page.get("imageinfo")
    if not title.startswith("File:") or not isinstance(infos, list) or not infos:
        return None
    info = infos[0]
    meta = info.get("extmetadata", {})
    if not isinstance(meta, dict):
        return None
    media_type = _ext(meta, "FileType").casefold()
    url = str(info.get("url", ""))
    if not url or (media_type and media_type not in {"jpeg", "png", "jpg"}):
        return None
    license_name = _plain(_ext(meta, "LicenseShortName"))
    if not license_name or not license_name.startswith(ALLOWED_LICENSE_PREFIXES):
        return None
    author = _plain(_ext(meta, "Artist"))
    if not author or author.casefold() in {"unknown", "anonymous", "various"}:
        return None
    image_bytes = _download(url)
    sha256 = _sha(image_bytes)
    bbox = _largest_face_bbox(image_bytes)
    if bbox is None:
        return None
    width, height = int(info.get("width", 0)), int(info.get("height", 0))
    if width <= 0 or height <= 0:
        return None
    identity_key = re.sub(r"\s+", " ", author).strip().casefold()
    file_name = title[5:]
    return {
        "author": author,
        "capture_notes": "Commons self-portrait/selfie source selected before V4 candidate modification.",
        "clean_source_sha256": sha256,
        "download_url": url,
        "face_bbox_normalized": bbox,
        "filename": file_name,
        "identity_key": identity_key,
        "license": license_name,
        "license_url": _plain(_ext(meta, "LicenseUrl")),
        "original_dimensions": [width, height],
        "page_url": f"https://commons.wikimedia.org/wiki/{quote(title.replace(' ', '_'), safe=':/')}",
        "primary_domain": domain == "female",
        "redistribution_status": "allowed_under_recorded_commons_license",
        "source_id": f"finalholdout4_{ordinal:02d}_{'f' if domain == 'female' else 'c'}",
        "source_category": "Selfies of women" if domain == "female" else "Selfies of men",
        "subject_domain": domain,
    }


def discover_sources() -> dict[str, Any]:
    old_ids, old_hashes, old_pages, old_identities = _old_identity_evidence()
    selected: list[dict[str, Any]] = []
    used_hashes: set[str] = set()
    used_pages: set[str] = set()
    used_identities: set[str] = set()

    def collect(categories: tuple[str, ...], domain: str, needed: int) -> None:
        nonlocal selected
        seen_titles: set[str] = set()
        for category in categories:
            pages = sorted(_category_files(category), key=lambda item: str(item.get("title", "")).casefold())
            for page in pages:
                title = str(page.get("title", ""))
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                candidate = _candidate_source(page, domain=domain, ordinal=len(selected) + 1)
                if candidate is None:
                    continue
                identity = candidate["identity_key"]
                if (
                    candidate["source_id"] in old_ids
                    or candidate["clean_source_sha256"] in old_hashes
                    or candidate["page_url"] in old_pages
                    or identity in old_identities
                    or candidate["clean_source_sha256"] in used_hashes
                    or candidate["page_url"] in used_pages
                    or identity in used_identities
                ):
                    continue
                selected.append(candidate)
                used_hashes.add(candidate["clean_source_sha256"])
                used_pages.add(candidate["page_url"])
                used_identities.add(identity)
                if sum(item["subject_domain"] == domain for item in selected) >= needed:
                    return
        raise RuntimeError(f"Not enough independent {domain} identities for V4: need {needed}")

    collect(FEMALE_CATEGORIES, "female", FEMALE_IDENTITY_COUNT)
    collect(CONTROL_CATEGORIES, "control", CONTROL_IDENTITY_COUNT)

    if len(selected) != TOTAL_IDENTITIES:
        raise RuntimeError(f"V4 source count drift: {len(selected)} != {TOTAL_IDENTITIES}")
    if len({item["identity_key"] for item in selected}) != TOTAL_IDENTITIES:
        raise RuntimeError("V4 identity registry contains duplicates")
    if sum(bool(item["primary_domain"]) for item in selected) != FEMALE_IDENTITY_COUNT:
        raise RuntimeError("V4 female-primary-domain ratio drift")
    return {
        "benchmark_id": BENCHMARK_ID,
        "download_date_utc": "2026-08-17",
        "identity_disjointness": "SHA-256, source page and declared self-portrait author identity keys are disjoint from V1, V2 and consumed V3.",
        "identity_registry": [
            {
                "identity_key": item["identity_key"],
                "source_id": item["source_id"],
                "subject_domain": item["subject_domain"],
            }
            for item in selected
        ],
        "primary_domain_identity_ratio": FEMALE_IDENTITY_COUNT / TOTAL_IDENTITIES,
        "sources": selected,
        "version": 1,
    }


def _assert_independent(sources_payload: dict[str, Any]) -> dict[str, str]:
    sources = list(sources_payload.get("sources", []))
    if len(sources) != TOTAL_IDENTITIES:
        raise RuntimeError(f"Expected {TOTAL_IDENTITIES} V4 identities, got {len(sources)}")
    old_ids, old_hashes, old_pages, old_identities = _old_identity_evidence()
    new_ids = {str(item["source_id"]) for item in sources}
    new_hashes = {str(item["clean_source_sha256"]) for item in sources}
    new_pages = {str(item["page_url"]) for item in sources}
    new_identities = {str(item["identity_key"]).casefold() for item in sources}
    collisions = {
        "source_id": sorted(new_ids & old_ids),
        "sha256": sorted(new_hashes & old_hashes),
        "page_url": sorted(new_pages & old_pages),
        "identity_key": sorted(new_identities & old_identities),
    }
    if any(collisions.values()):
        raise RuntimeError(f"V4 identity/source overlap with consumed data: {collisions!r}")
    if len(new_ids) != TOTAL_IDENTITIES or len(new_hashes) != TOTAL_IDENTITIES or len(new_identities) != TOTAL_IDENTITIES:
        raise RuntimeError("V4 sources are not one-to-one with independent identities")
    female = sum(str(item.get("subject_domain")) == "female" for item in sources)
    if female != FEMALE_IDENTITY_COUNT:
        raise RuntimeError(f"V4 primary-domain identity count drift: {female}")
    consumed: dict[str, str] = {}
    for name, root in CONSUMED_ROOTS.items():
        path = root / "sources.json"
        consumed[name] = _sha(_canonical_json(_load(path)))
        if len(consumed[name]) != 64:
            raise RuntimeError(f"Consumed {name} source-manifest checksum missing")
    return consumed


def build_cases() -> dict[str, Any]:
    sources_payload = _load(BENCHMARK_ROOT / "sources.json")
    sources = list(sources_payload["sources"])
    _assert_independent(sources_payload)
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
            if pixels <= 0:
                raise RuntimeError(f"Empty V4 damage mask: {source_id}/{style}")
            overlap = float(np.count_nonzero((mask > 0) & (face > 0)) / pixels)
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
                "case_id": f"cfsfs4-fin-{global_index + 1:03d}-{style}",
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
                "predeclared_abstention_expected": bool(extreme and reference_count == 0),
                "primary_face_case": True,
                "provenance_contract": "GENERATED_OR_RESTORED_NEVER_BECOMES_OBSERVED_EVIDENCE",
                "recoverability_pre_score": "LOW_EVIDENCE_ABSTAIN" if extreme else (
                    "REFERENCE_RECOVERABLE" if target95 else "SEVERE_RECOVERABLE"
                ),
                "reference_ids": references,
                "target95_applicable_pre_score": target95,
                "target95_policy": "FROZEN_APPLICABILITY",
                "wrong_person_source_ids": sorted({
                    ref.split(":", 1)[0] for ref in references if ref.split(":", 1)[0] != source_id
                }),
            })
            global_index += 1

    counts = {source_id: sum(c["main_source_id"] == source_id for c in cases) for source_id in source_ids}
    if global_index != CASE_COUNT or any(value != 2 for value in counts.values()):
        raise RuntimeError(f"V4 allocation drift: cases={global_index}, identities={counts!r}")
    ordinary = [c for c in cases if c["damage_type"] != "extreme_low_evidence"]
    if any(float(c["face_overlap_ratio"]) < 0.95 for c in ordinary):
        raise RuntimeError("V4 ordinary damage falls below 95% facial-domain intersection")
    return {
        "benchmark_id": BENCHMARK_ID,
        "cases": cases,
        "generation_algorithm": "freeze_face_smartphone_v4_final_holdout.py:v1",
        "mask_checksum_encoding": f"uint8_{CANVAS_SIZE}x{CANVAS_SIZE}_raw",
        "split": "final_holdout",
        "version": 1,
    }


def build_contract() -> dict[str, Any]:
    return {
        "benchmark_id": BENCHMARK_ID,
        "benchmark_status": "FROZEN_INDEPENDENT_FINAL_HOLDOUT_BEFORE_CANDIDATE_MODIFICATION",
        "canvas_size": CANVAS_SIZE,
        "case_count": CASE_COUNT,
        "category_counts": CATEGORY_COUNTS,
        "evaluation_priority": [
            "identity_preservation",
            "main_geometry_preservation",
            "observed_evidence_fidelity",
            "provenance_correctness",
            "face_damage_recovery",
            "undamaged_region_preservation",
            "artifact_minimization",
            "visual_attractiveness",
        ],
        "frozen_guardrails": {
            "sface_same_identity_minimum": 0.363,
            "outside_region_mae_max": 8.0,
            "provenance_valid_required": True,
            "wrong_person_final_pixels_max": 0,
            "calibration_required": "60/60",
        },
        "holdout_policy": "One execution only after a candidate source tree is frozen. A failed V4 is consumed and cannot be rerun for certification.",
        "identity_count": TOTAL_IDENTITIES,
        "identity_policy": "19 female primary-domain selfie identities plus one non-primary safety control; identity registry disjoint from V1/V2/V3.",
        "ordinary_damage_face_overlap_target": 0.95,
        "predeclared_abstention_policy": "Opaque or extreme cases with zero adequate same-person evidence may ABSTAIN; abstention is not restoration PASS.",
        "primary_domain_identity_count": FEMALE_IDENTITY_COUNT,
        "primary_domain_identity_ratio": FEMALE_IDENTITY_COUNT / TOTAL_IDENTITIES,
        "reference_count_range": [0, 9],
        "release_ready": False,
        "runner_policy": {
            "changed_manifest_refused": True,
            "mismatched_candidate_tree_refused": True,
            "overwrite_refused": True,
            "second_execution_refused": True,
            "unfrozen_candidate_refused": True,
        },
        "split_counts": {"final_holdout": CASE_COUNT},
        "target95_policy": "FROZEN_APPLICABILITY_BEFORE_CANDIDATE_EXECUTION",
        "tuning_started": False,
        "version": 1,
    }


def build_freeze(cases_payload: dict[str, Any], contract_payload: dict[str, Any]) -> dict[str, Any]:
    sources_payload = _load(BENCHMARK_ROOT / "sources.json")
    consumed_sources = _assert_independent(sources_payload)
    sources_bytes = _canonical_json(sources_payload)
    cases_bytes = _canonical_json(cases_payload)
    contract_bytes = _canonical_json(contract_payload)
    masks = [{"case_id": c["case_id"], "damage_mask_checksum": c["damage_mask_checksum"]} for c in cases_payload["cases"]]
    refs = [{"case_id": c["case_id"], "reference_ids": c["reference_ids"]} for c in cases_payload["cases"]]
    identities = list(sources_payload["identity_registry"])
    source_hashes = [
        {"source_id": item["source_id"], "sha256": item["clean_source_sha256"]}
        for item in sources_payload["sources"]
    ]
    return {
        "benchmark_id": BENCHMARK_ID,
        "cases_manifest_sha256": _sha(cases_bytes),
        "category_distribution": CATEGORY_COUNTS,
        "consumed_sources_manifest_sha256": consumed_sources,
        "contract_sha256": _sha(contract_bytes),
        "damage_seed_base": BASE_SEED,
        "frozen_before_candidate_modification": True,
        "identity_registry_sha256": _sha(_canonical_json(identities)),
        "mask_hashes_sha256": _sha(_canonical_json(masks)),
        "production_pipeline_changed_by_freeze": False,
        "reference_assignment_sha256": _sha(_canonical_json(refs)),
        "source_hash_registry_sha256": _sha(_canonical_json(source_hashes)),
        "sources_manifest_sha256": _sha(sources_bytes),
        "split": {"final_holdout": CASE_COUNT},
        "target95_policy": "FROZEN_APPLICABILITY",
        "version": 1,
    }


def _verify(path: Path, expected: bytes) -> None:
    actual = _normalized(path.read_bytes()) if path.is_file() else b""
    if actual != _normalized(expected):
        raise RuntimeError(f"Frozen V4 drift: {path.relative_to(REPOSITORY_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-sources", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    if args.discover_sources:
        sources_path = BENCHMARK_ROOT / "sources.json"
        if sources_path.exists():
            raise RuntimeError("Refusing to rediscover already frozen V4 sources")
        sources_path.write_bytes(_canonical_json(discover_sources()))

    if not (BENCHMARK_ROOT / "sources.json").is_file():
        raise RuntimeError("V4 sources.json missing; run --discover-sources exactly once before candidate modification")

    cases = build_cases()
    contract = build_contract()
    freeze = build_freeze(cases, contract)
    expected = {
        "cases.json": _canonical_json(cases),
        "contract.json": _canonical_json(contract),
        "freeze.json": _canonical_json(freeze),
    }

    if args.verify:
        for name, payload in expected.items():
            _verify(BENCHMARK_ROOT / name, payload)
        print("Face-smartphone V4 independent final holdout freeze: PASS")
        return 0

    for name, payload in expected.items():
        path = BENCHMARK_ROOT / name
        if path.exists():
            raise RuntimeError(f"Refusing to overwrite frozen V4 file: {name}")
        path.write_bytes(payload)
    print(f"Face-smartphone V4 freeze generated: {CASE_COUNT} cases / {TOTAL_IDENTITIES} identities")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
