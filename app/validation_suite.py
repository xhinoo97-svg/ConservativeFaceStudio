from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from app.reference_inpainting import verified_reference_repair
from app.restoration import DeblurSettings, conservative_deblur, quality_enhance
from app.validation import synthetic_degradations, validation_metrics


def _reference_image(width: int = 320, height: int = 320) -> np.ndarray:
    image = np.zeros((height, width, 3), np.uint8)
    cv2.ellipse(image, (width // 2, height // 2), (width // 3, height * 2 // 5), 0, 0, 360, (135, 165, 195), -1)
    cv2.circle(image, (width * 2 // 5, height * 2 // 5), max(3, width // 45), (25, 25, 25), -1)
    cv2.circle(image, (width * 3 // 5, height * 2 // 5), max(3, width // 45), (25, 25, 25), -1)
    cv2.line(image, (width // 2, height * 9 // 20), (width // 2, height * 3 // 5), (70, 80, 90), 3)
    cv2.ellipse(image, (width // 2, height * 7 // 10), (width // 8, height // 25), 0, 0, 180, (40, 40, 80), 3)
    return image


def _critical_occlusion_case(ground_truth: np.ndarray) -> dict[str, object]:
    h, w = ground_truth.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, (w * 2 // 5, h * 2 // 5), (w * 3 // 5, h * 3 // 5), 255, -1)

    damaged = ground_truth.copy()
    damaged[mask > 0] = (245, 25, 180)  # sticker-like obstruction

    reference_a = ground_truth.copy()
    reference_b = ground_truth.copy()
    repaired = verified_reference_repair(
        damaged,
        [reference_a, reference_b],
        mask,
        identity_scores=[0.90, 0.90],
        identity_verification_available=True,
        minimum_context_score=0.20,
        agreement_threshold=4.0,
        feather_sigma=0.0,
    )

    before = validation_metrics(damaged, ground_truth)
    after = validation_metrics(repaired.image, ground_truth)
    outside_untouched = bool(np.array_equal(repaired.image[mask == 0], damaged[mask == 0]))
    improved = after.psnr > before.psnr and after.identity_score >= before.identity_score
    complete = repaired.repaired_pixels > 0 and repaired.unresolved_pixels == 0
    passed = bool(outside_untouched and improved and complete)
    return {
        "before": asdict(before),
        "after": asdict(after),
        "requested_pixels": repaired.requested_pixels,
        "repaired_pixels": repaired.repaired_pixels,
        "unresolved_pixels": repaired.unresolved_pixels,
        "outside_mask_unchanged": outside_untouched,
        "psnr_delta": float(after.psnr - before.psnr),
        "identity_delta": float(after.identity_score - before.identity_score),
        "source_pixel_counts": list(repaired.source_pixel_counts),
        "passed": passed,
    }


def run_validation_suite(
    *,
    minimum_psnr_delta: float = -0.05,
    minimum_identity_delta: float = -0.05,
) -> dict[str, object]:
    ground_truth = _reference_image()
    cases = synthetic_degradations(ground_truth)
    report: dict[str, object] = {
        "format": "ConservativeFaceStudio validation suite",
        "version": 3,
        "thresholds": {
            "minimum_psnr_delta": float(minimum_psnr_delta),
            "minimum_identity_delta": float(minimum_identity_delta),
        },
        "passed": True,
        "cases": {},
        "critical_occlusion_repair": {},
    }
    failures: list[str] = []
    settings = DeblurSettings()
    for name, degraded in cases.items():
        restored = quality_enhance(conservative_deblur(degraded, settings))
        before = validation_metrics(degraded, ground_truth)
        after = validation_metrics(restored, ground_truth)
        identity_delta = float(after.identity_score - before.identity_score)
        psnr_delta = float(after.psnr - before.psnr)
        passed = psnr_delta >= minimum_psnr_delta and identity_delta >= minimum_identity_delta
        if not passed:
            failures.append(name)
        report["cases"][name] = {
            "before": asdict(before),
            "after": asdict(after),
            "identity_delta": identity_delta,
            "psnr_delta": psnr_delta,
            "passed": passed,
        }

    critical = _critical_occlusion_case(ground_truth)
    report["critical_occlusion_repair"] = critical
    if not bool(critical["passed"]):
        failures.append("critical_occlusion_repair")

    report["passed"] = not failures
    report["failed_cases"] = failures
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="validation-report.json")
    parser.add_argument("--fail-on-regression", action="store_true")
    args = parser.parse_args()
    report = run_validation_suite()
    target = Path(args.output)
    target.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.fail_on_regression and not bool(report["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
