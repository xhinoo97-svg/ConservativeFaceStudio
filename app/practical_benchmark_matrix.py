from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core_models import ensure_core_pretrained_models
from app.practical_benchmark import (
    PortraitSource,
    Scenario,
    _fit_portrait,
    _partial_reference,
    download_public_portraits,
    evaluate_scenario,
)


# Real same-identity references with a different session/pose. Both are NASA works
# documented as public domain on Wikimedia Commons. They are downloaded at benchmark
# time and are never redistributed with the application.
REAL_POSE_REFERENCES: dict[str, PortraitSource] = {
    "mae_jemison": PortraitSource(
        "mae_jemison_pose",
        "Mae Carol Jemison.jpg",
        "https://commons.wikimedia.org/wiki/File:Mae_Carol_Jemison.jpg",
    ),
    "sally_ride": PortraitSource(
        "sally_ride_pose",
        "Sally Ride (1984).jpg",
        "https://commons.wikimedia.org/wiki/File:Sally_Ride_(1984).jpg",
    ),
}


def _direct_upload_url(filename: str) -> str:
    normalized = filename.replace(" ", "_")
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    quoted = urllib.parse.quote(normalized, safe="()_,.-")
    return f"https://upload.wikimedia.org/wikipedia/commons/{digest[0]}/{digest[:2]}/{quoted}"


def _download_pose_reference(source: PortraitSource, root: Path, *, attempts: int = 3, timeout: int = 60) -> dict[str, str]:
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{source.key}.jpg"
    if target.is_file() and cv2.imread(str(target), cv2.IMREAD_COLOR) is not None:
        return {
            "key": source.key,
            "filename": source.filename,
            "page_url": source.page_url,
            "download_url": "cache",
            "license": source.license,
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "local_path": str(target),
        }

    candidates = (_direct_upload_url(source.filename), source.download_url)
    last_error: Exception | None = None
    for url in candidates:
        for attempt in range(attempts):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "ConservativeFaceStudio-benchmark/1.2 (+https://github.com/xhinoo97-svg/ConservativeFaceStudio)",
                        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    },
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                target.write_bytes(payload)
                image = cv2.imread(str(target), cv2.IMREAD_COLOR)
                if image is None or image.size == 0:
                    target.unlink(missing_ok=True)
                    raise RuntimeError("downloaded pose reference is not a decodable image")
                return {
                    "key": source.key,
                    "filename": source.filename,
                    "page_url": source.page_url,
                    "download_url": url,
                    "license": source.license,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "local_path": str(target),
                }
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
                last_error = exc
                target.unlink(missing_ok=True)
                if attempt + 1 < attempts:
                    retry_after = 0.0
                    if isinstance(exc, urllib.error.HTTPError):
                        try:
                            retry_after = float(exc.headers.get("Retry-After", "0"))
                        except (TypeError, ValueError):
                            retry_after = 0.0
                    time.sleep(max(retry_after, min(6.0, 1.0 * (2**attempt))))
    raise RuntimeError(f"unable to download real pose reference {source.key}: {last_error}")


def _rect_mask(shape: tuple[int, int], x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), 255, -1)
    return mask


def _disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    kernel = ((xx * xx + yy * yy) <= radius * radius).astype(np.float32)
    kernel /= max(float(kernel.sum()), 1.0)
    return cv2.filter2D(image, -1, kernel)


def _opaque_damage(clean: np.ndarray, mask: np.ndarray, value: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    damaged = clean.copy()
    damaged[mask > 0] = value
    return damaged


def make_extended_scenarios(clean: np.ndarray) -> tuple[Scenario, ...]:
    h, w = clean.shape[:2]
    full = np.full((h, w), 255, dtype=np.uint8)
    left_half = _rect_mask((h, w), 0.0, 0.0, 0.52, 1.0)
    eye_band = _rect_mask((h, w), 0.18, 0.25, 0.82, 0.48)
    nose = _rect_mask((h, w), 0.38, 0.40, 0.62, 0.66)
    mouth_chin = _rect_mask((h, w), 0.25, 0.58, 0.75, 0.90)
    upper_crop = _rect_mask((h, w), 0.05, 0.05, 0.95, 0.58)
    lower_crop = _rect_mask((h, w), 0.05, 0.42, 0.95, 0.95)
    center_crop = _rect_mask((h, w), 0.22, 0.20, 0.78, 0.84)

    half_damaged = _opaque_damage(clean, left_half)
    central_damage = _rect_mask((h, w), 0.28, 0.30, 0.72, 0.78)
    central_opaque = _opaque_damage(clean, central_damage)

    return (
        Scenario("defocus_mild_single", _disk_blur(clean, 3), (), full, True),
        Scenario("defocus_heavy_single", _disk_blur(clean, 7), (), full, True),
        Scenario("half_face_opaque_single", half_damaged, (), left_half, False, True),
        Scenario("eye_only_reference", central_opaque, (_partial_reference(clean, eye_band),), central_damage, True),
        Scenario("nose_only_reference", central_opaque, (_partial_reference(clean, nose),), central_damage, True),
        Scenario("mouth_chin_only_reference", central_opaque, (_partial_reference(clean, mouth_chin),), central_damage, True),
        Scenario("two_partial_crops", central_opaque, (_partial_reference(clean, upper_crop), _partial_reference(clean, lower_crop)), central_damage, True),
        Scenario("multi_reference_complementary", central_opaque, (_partial_reference(clean, upper_crop), _partial_reference(clean, lower_crop), _partial_reference(clean, center_crop)), central_damage, True),
    )


def _real_pose_scenario(clean: np.ndarray, reference: np.ndarray) -> Scenario:
    h, w = clean.shape[:2]
    central_damage = _rect_mask((h, w), 0.28, 0.30, 0.72, 0.78)
    central_opaque = _opaque_damage(clean, central_damage)
    return Scenario("real_same_identity_pose_reference", central_opaque, (reference,), central_damage, True)


def run_matrix(output: Path, *, cache: Path, limit: int = 10, size: int = 320) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sources = download_public_portraits(cache, limit=limit)
    bootstrap = ensure_core_pretrained_models(output / "core-models", timeout_seconds=60)

    pose_sources: dict[str, dict[str, str]] = {}
    pose_download_errors: dict[str, str] = {}
    pose_cache = cache / "real-pose-references"
    for identity_key, source in REAL_POSE_REFERENCES.items():
        try:
            pose_sources[identity_key] = _download_pose_reference(source, pose_cache)
        except Exception as exc:
            pose_download_errors[identity_key] = str(exc)

    report: dict[str, Any] = {
        "format": "ConservativeFaceStudio extended practical scenario matrix",
        "version": 2,
        "portrait_count": len(sources),
        "base_scenario_count_per_portrait": 8,
        "real_pose_scenario_identity_count": len(pose_sources),
        "note": "Scores remain decomposed metrics; no universal 95% claim. Opaque single-image half-face cases are explicitly non-recoverable ground-truth cases. Real-pose scenarios use separately photographed public-domain images of the same identity, never synthetic crops of the ground truth.",
        "sources": sources,
        "real_pose_sources": pose_sources,
        "real_pose_download_errors": pose_download_errors,
        "core_models_ready": bootstrap.ready,
        "core_model_errors": bootstrap.errors,
        "cases": [],
    }

    for portrait_index, item in enumerate(sources):
        image = cv2.imread(item["local_path"], cv2.IMREAD_COLOR)
        if image is None:
            report["cases"].append({"portrait": item["key"], "error": "decode failed"})
            continue
        clean = _fit_portrait(image, size=size)
        portrait_dir = output / item["key"]
        scenarios = list(make_extended_scenarios(clean))

        pose_item = pose_sources.get(item["key"])
        if pose_item is not None:
            pose_image = cv2.imread(pose_item["local_path"], cv2.IMREAD_COLOR)
            if pose_image is not None and pose_image.size:
                scenarios.append(_real_pose_scenario(clean, _fit_portrait(pose_image, size=size)))

        for scenario in scenarios:
            try:
                metrics = evaluate_scenario(clean, scenario, portrait_dir, core_paths=bootstrap.paths if bootstrap.ready else None)
                metrics["portrait"] = item["key"]
                report["cases"].append(metrics)
            except Exception as exc:
                report["cases"].append({
                    "portrait": item["key"],
                    "scenario": scenario.name,
                    "recoverable": scenario.recoverable,
                    "error": str(exc),
                })

    valid = [item for item in report["cases"] if "conservative_recovery_score" in item]
    errors = [item for item in report["cases"] if "error" in item]
    by_scenario: dict[str, dict[str, float | int | None]] = {}
    names = sorted({str(item.get("scenario")) for item in valid if item.get("scenario")})
    for name in names:
        cases = [item for item in valid if item.get("scenario") == name]
        by_scenario[name] = {
            "count": len(cases),
            "mean_score": float(np.mean([item["conservative_recovery_score"] for item in cases])) if cases else None,
            "mean_damage_mae_after": float(np.mean([item["damage_mae_after"] for item in cases])) if cases else None,
            "mean_identity_similarity": float(np.mean([item["identity_similarity"] for item in cases])) if cases else None,
            "mean_reference_fraction": float(np.mean([item["reference_fraction"] for item in cases])) if cases else None,
        }
    report["summary"] = {
        "completed_cases": len(valid),
        "error_cases": len(errors),
        "by_scenario": by_scenario,
    }

    (output / "practical-matrix.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    fields = [
        "portrait", "scenario", "recoverable", "reference_count", "conservative_recovery_score",
        "psnr_before", "psnr_after", "ssim_after", "damage_mae_before", "damage_mae_after",
        "identity_similarity", "landmark_nme", "occlusion_iou", "occlusion_precision", "occlusion_recall",
        "reference_fraction", "symmetry_fraction", "generated_fraction", "abstention_correct", "error",
    ]
    with (output / "practical-matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="practical-matrix")
    parser.add_argument("--cache", default=".benchmark-cache/public-portraits")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args()
    report = run_matrix(Path(args.output), cache=Path(args.cache), limit=args.limit, size=args.size)
    print(json.dumps(report.get("summary", {}), indent=2, sort_keys=True))
    if args.fail_on_errors and int(report.get("summary", {}).get("error_cases", 0)) > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
