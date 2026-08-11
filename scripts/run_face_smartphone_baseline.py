from __future__ import annotations

"""Run the unchanged production pipeline on the frozen face-smartphone benchmark."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import urllib.request
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import app  # noqa: F401  # Apply the same OpenCV process policy as production.
import cv2
import numpy as np

from app.practical_benchmark import Scenario, _fit_portrait, _motion_blur, _mosaic, evaluate_scenario
from app.production_model_smoke import PRODUCTION_MODEL_KEYS
from app.model_catalog import all_model_manifests
from scripts import freeze_face_smartphone_benchmark as freeze


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def acquire_sources(cache: Path, *, offline: bool = False) -> dict[str, Path]:
    payload = json.loads((freeze.BENCHMARK_ROOT / "sources.json").read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for source in payload["sources"]:
        target = cache / source["filename"]
        expected = source["clean_source_sha256"]
        if target.is_file() and _sha256(target) != expected:
            raise RuntimeError(f"Frozen clean source checksum mismatch: {source['source_id']}")
        if not target.is_file():
            if offline:
                raise RuntimeError(f"Frozen clean source absent in offline mode: {source['source_id']}")
            temporary = target.with_suffix(target.suffix + ".tmp")
            request = urllib.request.Request(source["download_url"], headers={"User-Agent": "ConservativeFaceStudio-face-smartphone-benchmark/1.0"})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload_bytes = response.read()
                temporary.write_bytes(payload_bytes)
                if _sha256(temporary) != expected:
                    raise RuntimeError(f"Downloaded frozen clean source checksum mismatch: {source['source_id']}")
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        resolved[source["source_id"]] = target
    return resolved


def load_clean_images(paths: dict[str, Path]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for source_id, path in paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Frozen source decode failed: {source_id}")
        result[source_id] = _fit_portrait(image, size=freeze.CANVAS_SIZE)
    return result


def production_model_paths(root: Path) -> dict[str, Path]:
    registry = {manifest.key: manifest for manifest in all_model_manifests()}
    paths: dict[str, Path] = {}
    for key in PRODUCTION_MODEL_KEYS:
        manifest = registry[key]
        candidate = root / manifest.destination
        if not candidate.is_file():
            raise RuntimeError(f"Production model missing for frozen baseline: {key}: {candidate}")
        actual = _sha256(candidate)
        if manifest.expected_sha256 and actual != manifest.expected_sha256:
            raise RuntimeError(f"Production model checksum mismatch for frozen baseline: {key}")
        paths[key] = candidate
    return paths


def _component_mask(source: dict[str, Any], variant: str) -> np.ndarray:
    x0, y0, x1, y1 = freeze._fitted_face_bbox(source)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    mask = np.zeros((freeze.CANVAS_SIZE, freeze.CANVAS_SIZE), dtype=np.uint8)
    boxes = {
        "left_half_observed": (0.00, 0.00, 0.56, 1.00),
        "right_half_observed": (0.44, 0.00, 1.00, 1.00),
        "left_eye_only": (0.05, 0.17, 0.51, 0.45),
        "right_eye_only": (0.49, 0.17, 0.95, 0.45),
        "nose_dominant": (0.30, 0.30, 0.70, 0.72),
        "mouth_chin_only": (0.19, 0.64, 0.81, 1.00),
    }
    rx0, ry0, rx1, ry1 = boxes[variant]
    cv2.rectangle(
        mask,
        (int(round(x0 + rx0 * width)), int(round(y0 + ry0 * height))),
        (int(round(x0 + rx1 * width)), int(round(y0 + ry1 * height))),
        255,
        -1,
    )
    return cv2.bitwise_and(mask, freeze._face_domain_mask(source))


def _partial_reference(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    reference = np.zeros_like(clean)
    reference[mask > 0] = clean[mask > 0]
    return reference


def materialize_reference(reference_id: str, sources: dict[str, dict[str, Any]], clean_images: dict[str, np.ndarray]) -> np.ndarray:
    source_id, variant = reference_id.split(":", 1)
    clean = clean_images[source_id]
    source = sources[source_id]
    if variant in {"full_observed", "duplicate_full", "wrong_person_full"}:
        return clean.copy()
    if variant in {"left_half_observed", "right_half_observed", "left_eye_only", "right_eye_only", "nose_dominant", "mouth_chin_only"}:
        return _partial_reference(clean, _component_mask(source, variant))
    if variant in {"blurred_low_quality", "wrong_person_blurred"}:
        return cv2.GaussianBlur(clean, (15, 15), 3.2)
    if variant == "useless_blank":
        return np.zeros_like(clean)
    raise ValueError(f"Unknown frozen reference variant: {variant}")


def _mixed_phone_degradation(clean: np.ndarray, style: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if style == "face_blur_plus_low_light":
        return np.clip(clean.astype(np.float32) * 0.46 + rng.normal(0.0, 5.0, clean.shape), 0, 255).astype(np.uint8)
    if style == "mosaic_plus_resize":
        h, w = clean.shape[:2]
        small = cv2.resize(clean, (max(2, w // 3), max(2, h // 3)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    quality = 34 if style == "face_sticker_plus_jpeg" else 46
    ok, encoded = cv2.imencode(".jpg", clean, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else clean.copy()


def render_primary(clean: np.ndarray, source: dict[str, Any], case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mask = freeze._raw_mask(source, case["mask_shape"], case["mask_region"], int(case["damage_seed"]))
    if hashlib.sha256(mask.tobytes(order="C")).hexdigest() != case["damage_mask_checksum"]:
        raise RuntimeError(f"Frozen damage mask checksum mismatch: {case['case_id']}")
    primary = clean.copy()
    category = case["damage_type"]
    style = case["damage_style"]
    if category == "face_blur_censor":
        if style == "local_motion_blur":
            candidate = _motion_blur(clean, 21)
        else:
            kernel = 31 if "near_full" in style else 21
            candidate = cv2.GaussianBlur(clean, (kernel, kernel), 7.0 if kernel == 31 else 4.0)
        primary[mask > 0] = candidate[mask > 0]
    elif category == "face_mosaic_pixelation":
        factor = 24 if "large_block" in style else (9 if "small_block" in style else 16)
        candidate = _mosaic(clean, factor)
        primary[mask > 0] = candidate[mask > 0]
    else:
        if category == "mixed_smartphone":
            primary = _mixed_phone_degradation(clean, style, int(case["damage_seed"]))
        color = (8, 8, 8)
        if "star" in style:
            color = (20, 35, 225)
        elif "emoji" in style:
            color = (40, 210, 245)
        elif "hand" in style:
            color = (120, 160, 205)
        elif "goggles" in style or "eye_mask" in style:
            color = (35, 20, 55)
        elif "hair" in style:
            color = (20, 20, 28)
        primary[mask > 0] = color
    return primary, mask


def materialize_scenario(case: dict[str, Any], sources: dict[str, dict[str, Any]], clean_images: dict[str, np.ndarray]) -> tuple[np.ndarray, Scenario]:
    source = sources[case["main_source_id"]]
    clean = clean_images[case["main_source_id"]]
    primary, mask = render_primary(clean, source, case)
    references = tuple(materialize_reference(item, sources, clean_images) for item in case["reference_ids"])
    recoverable = case["recoverability_pre_score"] != "LOW_EVIDENCE_ABSTAIN"
    scenario = Scenario(
        name=case["case_id"],
        primary=primary,
        references=references,
        damage_mask=mask,
        recoverable=recoverable,
        opaque_without_evidence=not bool(case["target95_applicable_pre_score"]),
    )
    return clean, scenario


def _peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def select_cases(cases: list[dict[str, Any]], split: str, case_ids: set[str] | None) -> list[dict[str, Any]]:
    selected = [item for item in cases if split == "all" or item["calibration_or_holdout"] == split]
    if case_ids is None:
        return selected
    available = {str(item["case_id"]) for item in selected}
    missing = sorted(case_ids - available)
    if missing:
        raise ValueError(f"Frozen case IDs not found in selected split: {', '.join(missing)}")
    return [item for item in selected if item["case_id"] in case_ids]


def run_baseline(
    output: Path,
    *,
    cache: Path,
    model_root: Path,
    split: str = "calibration",
    offline_sources: bool = False,
    case_ids: set[str] | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    cases_payload = freeze.build_cases()
    freeze_payload = freeze.build_freeze(cases_payload)
    sources_payload = json.loads((freeze.BENCHMARK_ROOT / "sources.json").read_text(encoding="utf-8"))
    sources = {item["source_id"]: item for item in sources_payload["sources"]}
    source_paths = acquire_sources(cache, offline=offline_sources)
    clean_images = load_clean_images(source_paths)
    model_paths = production_model_paths(model_root)
    selected = select_cases(cases_payload["cases"], split, case_ids)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "benchmark_id": cases_payload["benchmark_id"],
        "benchmark_freeze": freeze_payload,
        "baseline_id": None if candidate_id else "production-unchanged-368f00c",
        "candidate_id": candidate_id,
        "production_sha": "368f00c122520f471e7ef310f9daf8781b51f111",
        "production_pipeline_changed": candidate_id is not None,
        "split": split,
        "holdout_used_for_tuning": False,
        "target95_policy": "REPORT_ONLY",
        "model_sha256": {key: _sha256(path) for key, path in model_paths.items()},
        "source_sha256": {key: _sha256(path) for key, path in source_paths.items()},
        "cases": [],
    }
    for index, case in enumerate(selected, start=1):
        started = time.perf_counter()
        try:
            clean, scenario = materialize_scenario(case, sources, clean_images)
            metrics = evaluate_scenario(clean, scenario, output / "cases", core_paths=model_paths)
            metrics.update({
                "case_id": case["case_id"],
                "calibration_or_holdout": case["calibration_or_holdout"],
                "damage_type": case["damage_type"],
                "damage_style": case["damage_style"],
                "face_overlap_ratio": case["face_overlap_ratio"],
                "reference_ids": case["reference_ids"],
                "target95_applicable_pre_score": case["target95_applicable_pre_score"],
                "target95_policy": "REPORT_ONLY",
                "runtime_seconds": time.perf_counter() - started,
                "process_peak_rss_mib": _peak_rss_mib(),
            })
            provenance_sum = sum(float(metrics.get(key, 0.0)) for key in ("primary_fraction", "reference_fraction", "symmetry_fraction", "generated_fraction"))
            metrics["provenance_fraction_sum"] = provenance_sum
            metrics["provenance_valid"] = bool(abs(provenance_sum - 1.0) <= 1e-6)
            metrics["hard_guardrail_pass"] = bool(metrics["provenance_valid"] and float(metrics.get("outside_region_mae", 255.0)) <= 8.0)
            report["cases"].append(metrics)
        except Exception as exc:
            report["cases"].append({
                "case_id": case["case_id"],
                "calibration_or_holdout": case["calibration_or_holdout"],
                "damage_type": case["damage_type"],
                "runtime_seconds": time.perf_counter() - started,
                "failure_reason": str(exc),
            })
        print(f"[{index}/{len(selected)}] {case['case_id']}", flush=True)
        (output / "baseline.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    completed = [item for item in report["cases"] if "conservative_recovery_score" in item]
    failures = [item for item in report["cases"] if "failure_reason" in item]
    report["summary"] = {
        "completed_cases": len(completed),
        "error_cases": len(failures),
        "hard_guardrail_passes": sum(item.get("hard_guardrail_pass") is True for item in completed),
        "mean_conservative_recovery_score_report_only": float(np.mean([item["conservative_recovery_score"] for item in completed])) if completed else None,
        "mean_identity_similarity": float(np.mean([item["identity_similarity"] for item in completed])) if completed else None,
        "mean_outside_region_mae": float(np.mean([item["outside_region_mae"] for item in completed])) if completed else None,
        "target95_applicable_pre_score": sum(item.get("target95_applicable_pre_score") is True for item in completed),
        "target95_pass_report_only": sum(item.get("target95_passed") is True and item.get("target95_applicable_pre_score") is True for item in completed),
    }
    (output / "baseline.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="face-smartphone-baseline")
    parser.add_argument("--cache", default=".benchmark-cache/face-smartphone-v1")
    parser.add_argument("--model-root", default=".")
    parser.add_argument("--split", choices=("calibration", "holdout", "all"), default="calibration")
    parser.add_argument("--offline-sources", action="store_true")
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--candidate-id", default=None)
    args = parser.parse_args()
    report = run_baseline(
        Path(args.output),
        cache=Path(args.cache),
        model_root=Path(args.model_root),
        split=args.split,
        offline_sources=args.offline_sources,
        case_ids=set(args.case_id) if args.case_id else None,
        candidate_id=args.candidate_id,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 2 if args.fail_on_errors and report["summary"]["error_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
