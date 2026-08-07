from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

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


def run_validation_suite() -> dict[str, object]:
    ground_truth = _reference_image()
    cases = synthetic_degradations(ground_truth)
    report: dict[str, object] = {"format": "ConservativeFaceStudio validation suite", "version": 1, "cases": {}}
    for name, degraded in cases.items():
        restored = quality_enhance(conservative_deblur(degraded, DeblurSettings(denoise=4, sharpen=0.7, contrast=1.0)))
        before = validation_metrics(degraded, ground_truth)
        after = validation_metrics(restored, ground_truth)
        report["cases"][name] = {
            "before": asdict(before),
            "after": asdict(after),
            "identity_delta": after.identity_score - before.identity_score,
            "psnr_delta": after.psnr - before.psnr,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="validation-report.json")
    args = parser.parse_args()
    target = Path(args.output)
    target.write_text(json.dumps(run_validation_suite(), indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
