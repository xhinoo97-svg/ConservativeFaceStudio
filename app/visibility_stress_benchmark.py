from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.practical_benchmark import (
    Scenario,
    _fit_portrait,
    download_public_portraits,
    evaluate_scenario,
)
from app.core_models import ensure_core_pretrained_models


@dataclass(frozen=True)
class VisibilityCase:
    scenario: Scenario
    mode: str
    visible_fraction: float
    expected_union_support: np.ndarray


def _partial_reference(clean: np.ndarray, support: np.ndarray) -> np.ndarray:
    out = np.zeros_like(clean)
    out[support > 0] = clean[support > 0]
    return out


def _opaque_damage(clean: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = clean.copy()
    out[mask > 0] = (12, 12, 12)
    return out


def _mask_fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) / max(1, mask.size))


def _visible_masks(shape: tuple[int, int]) -> dict[str, np.ndarray]:
    h, w = shape
    masks: dict[str, np.ndarray] = {}

    left = np.zeros((h, w), np.uint8)
    left[:, : int(round(w * 0.40))] = 255
    masks["left40"] = left

    right = np.zeros((h, w), np.uint8)
    right[:, w - int(round(w * 0.40)) :] = 255
    masks["right40"] = right

    upper = np.zeros((h, w), np.uint8)
    upper[: int(round(h * 0.40)), :] = 255
    masks["upper40"] = upper

    lower = np.zeros((h, w), np.uint8)
    lower[h - int(round(h * 0.40)) :, :] = 255
    masks["lower40"] = lower

    center = np.zeros((h, w), np.uint8)
    x0, x1 = int(round(w * 0.30)), int(round(w * 0.70))
    center[:, x0:x1] = 255
    masks["center40"] = center
    return masks


def _split_support(mask: np.ndarray, count: int) -> tuple[np.ndarray, ...]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return tuple(np.zeros_like(mask) for _ in range(count))
    xmin, xmax = int(xs.min()), int(xs.max()) + 1
    edges = np.linspace(xmin, xmax, count + 1, dtype=int)
    supports: list[np.ndarray] = []
    for index in range(count):
        support = np.zeros_like(mask)
        support[:, edges[index] : edges[index + 1]] = mask[:, edges[index] : edges[index + 1]]
        supports.append(support)
    return tuple(supports)


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def make_visibility_cases(clean: np.ndarray) -> tuple[VisibilityCase, ...]:
    h, w = clean.shape[:2]
    full = np.full((h, w), 255, np.uint8)
    cases: list[VisibilityCase] = []

    for label, visible in _visible_masks((h, w)).items():
        damage = cv2.bitwise_not(visible)
        primary = _opaque_damage(clean, damage)
        visible_fraction = _mask_fraction(visible)

        # SINGLE: deliberately unsupported hidden area. Diagnostic only; no >=95 claim.
        cases.append(
            VisibilityCase(
                Scenario(f"single_{label}", primary, (), damage, False, True),
                "single",
                visible_fraction,
                np.zeros_like(damage),
            )
        )

        # MULTI: complementary real-pixel donors cover the entire hidden area.
        for donor_count in (2, 3, 5):
            supports = _split_support(damage, donor_count)
            references = tuple(_partial_reference(clean, support) for support in supports)
            union = np.zeros_like(damage)
            for support in supports:
                union = cv2.bitwise_or(union, support)
            cases.append(
                VisibilityCase(
                    Scenario(
                        f"multi{donor_count}_{label}",
                        primary,
                        references,
                        damage,
                        True,
                    ),
                    "multi",
                    visible_fraction,
                    union,
                )
            )

    # Mixed difficulty: only 40% visible, plus blur and rotated donor evidence.
    visible = _visible_masks((h, w))["left40"]
    damage = cv2.bitwise_not(visible)
    primary = _opaque_damage(cv2.GaussianBlur(clean, (11, 11), 2.6), damage)
    supports = _split_support(damage, 3)
    rotated_refs = tuple(_rotate(_partial_reference(clean, support), angle) for support, angle in zip(supports, (-15.0, 15.0, 30.0)))
    union = np.zeros_like(damage)
    for support in supports:
        union = cv2.bitwise_or(union, support)
    cases.append(
        VisibilityCase(
            Scenario("multi3_left40_blur_rotated_donors", primary, rotated_refs, damage, True),
            "multi",
            _mask_fraction(visible),
            union,
        )
    )
    return tuple(cases)


def _damage_reference_coverage(provenance: np.ndarray, damage: np.ndarray, expected_support: np.ndarray) -> float | None:
    target = (damage > 0) & (expected_support > 0)
    denominator = int(np.count_nonzero(target))
    if denominator == 0:
        return None
    reference_pixel = (provenance > 0) & (provenance < np.uint16(65534))
    return float(np.count_nonzero(reference_pixel & target) / denominator)


def run_visibility_benchmark(output: Path, *, cache: Path, limit: int = 3, size: int = 256) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    sources = download_public_portraits(cache, limit=limit)
    bootstrap = ensure_core_pretrained_models(output / "core-models", timeout_seconds=60)
    report: dict[str, Any] = {
        "format": "ConservativeFaceStudio 40-percent-visible stress benchmark",
        "version": 1,
        "target_visible_fraction": 0.40,
        "multi_reference_target_score": 95.0,
        "single_image_note": "Unsupported hidden pixels are diagnostic only and are not claimed recoverable at 95 percent fidelity.",
        "cases": [],
    }

    for portrait_index, source in enumerate(sources):
        image = cv2.imread(source["local_path"], cv2.IMREAD_COLOR)
        if image is None:
            report["cases"].append({"portrait": source["key"], "error": "decode failed"})
            continue
        clean = _fit_portrait(image, size=size)
        for case in make_visibility_cases(clean):
            case_dir = output / source["key"]
            try:
                metrics = evaluate_scenario(
                    clean,
                    case.scenario,
                    case_dir,
                    core_paths=bootstrap.paths if bootstrap.ready else None,
                )
                provenance_path = Path(metrics["blocks_zip"]).parent / "provenance-map.npy"
                coverage = None
                if provenance_path.is_file():
                    provenance = np.load(provenance_path)
                    coverage = _damage_reference_coverage(provenance, case.scenario.damage_mask, case.expected_union_support)
                metrics.update(
                    {
                        "portrait": source["key"],
                        "mode": case.mode,
                        "visible_fraction": case.visible_fraction,
                        "expected_union_support_fraction": _mask_fraction(case.expected_union_support),
                        "damage_reference_coverage": coverage,
                        "strict95_applicable": case.mode == "multi",
                        "strict95_passed": bool(metrics.get("conservative_recovery_score", 0.0) >= 95.0) if case.mode == "multi" else None,
                    }
                )
                report["cases"].append(metrics)
            except Exception as exc:
                report["cases"].append({"portrait": source["key"], "scenario": case.scenario.name, "mode": case.mode, "error": str(exc)})

    valid = [item for item in report["cases"] if "conservative_recovery_score" in item]
    multi = [item for item in valid if item.get("mode") == "multi"]
    single = [item for item in valid if item.get("mode") == "single"]
    report["summary"] = {
        "completed_cases": len(valid),
        "error_cases": len(report["cases"]) - len(valid),
        "multi_cases": len(multi),
        "single_cases": len(single),
        "multi_mean_score": float(np.mean([x["conservative_recovery_score"] for x in multi])) if multi else None,
        "multi_pass95_count": int(sum(x.get("strict95_passed") is True for x in multi)),
        "multi_pass95_rate": float(sum(x.get("strict95_passed") is True for x in multi) / len(multi)) if multi else None,
        "single_mean_score": float(np.mean([x["conservative_recovery_score"] for x in single])) if single else None,
    }
    (output / "visibility-stress.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    fields = ["portrait", "scenario", "mode", "visible_fraction", "reference_count", "conservative_recovery_score", "identity_similarity", "ssim_after", "damage_mae_after", "outside_region_mae", "reference_fraction", "generated_fraction", "damage_reference_coverage", "strict95_applicable", "strict95_passed", "error"]
    with (output / "visibility-stress.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["cases"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="visibility-stress")
    parser.add_argument("--cache", default=".benchmark-cache/public-portraits")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--require-multi-95", action="store_true")
    args = parser.parse_args()
    report = run_visibility_benchmark(Path(args.output), cache=Path(args.cache), limit=args.limit, size=args.size)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    if args.fail_on_errors and report["summary"]["error_cases"]:
        return 2
    if args.require_multi_95 and report["summary"]["multi_pass95_rate"] != 1.0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
