from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.core_models import ensure_core_pretrained_models
from app.practical_benchmark import Scenario, _fit_portrait, _partial_reference, download_public_portraits, evaluate_scenario


def _rect_mask(shape: tuple[int, int], x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), 255, -1)
    return mask


def _disk_blur(image: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    size = radius * 2 + 1
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


def run_matrix(output: Path, *, cache: Path, limit: int = 10, size: int = 320) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sources = download_public_portraits(cache, limit=limit)
    bootstrap = ensure_core_pretrained_models(output / "core-models", timeout_seconds=60)
    report: dict[str, Any] = {
        "format": "ConservativeFaceStudio extended practical scenario matrix",
        "version": 1,
        "portrait_count": len(sources),
        "scenario_count_per_portrait": 8,
        "note": "Scores remain decomposed metrics; no universal 95% claim. Opaque single-image half-face cases are explicitly non-recoverable ground-truth cases.",
        "sources": sources,
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
        for scenario in make_extended_scenarios(clean):
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
