from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


EVALUATION_DAMAGE_TYPES: tuple[str, ...] = (
    "OPAQUE_STICKER",
    "TRANSLUCENT_STICKER",
    "EMOJI",
    "TEXT",
    "SCRIBBLE_THIN_BLACK",
    "SCRIBBLE_THICK_BLACK",
    "SCRIBBLE_THIN_COLOR",
    "SCRIBBLE_THICK_COLOR",
    "BLUR_LOCAL",
    "BLUR_GLOBAL",
    "MOTION_BLUR",
    "DEFOCUS",
    "BLOCK_MOSAIC",
    "PIXELATION",
    "JPEG_ARTIFACT",
    "NOISE",
    "MIXED_DAMAGE",
    "HEALTHY",
)

POSITIONS: tuple[str, ...] = (
    "LEFT_EYE",
    "RIGHT_EYE",
    "NOSE",
    "MOUTH",
    "LEFT_CHEEK",
    "RIGHT_CHEEK",
    "FOREHEAD",
)
SIZES: tuple[str, ...] = ("SMALL", "MEDIUM", "LARGE")
SEVERITIES: tuple[str, ...] = ("LIGHT", "MEDIUM", "SEVERE")

# Legacy 12-class LR-ASPP can be measured for binary localization on every row.
# Exact class scoring is only authoritative where the old taxonomy has a direct semantic match.
LEGACY_CLASS_FOR_TYPE: Mapping[str, str | None] = {
    "OPAQUE_STICKER": "STICKER",
    "TRANSLUCENT_STICKER": None,
    "EMOJI": "STICKER",
    "TEXT": "SCRIBBLE",
    "SCRIBBLE_THIN_BLACK": "SCRIBBLE",
    "SCRIBBLE_THICK_BLACK": "SCRIBBLE",
    "SCRIBBLE_THIN_COLOR": "SCRIBBLE",
    "SCRIBBLE_THICK_COLOR": "SCRIBBLE",
    "BLUR_LOCAL": "BLUR",
    "BLUR_GLOBAL": "BLUR",
    "MOTION_BLUR": "MOTION_BLUR",
    "DEFOCUS": "BLUR",
    "BLOCK_MOSAIC": "BLOCK_MOSAIC",
    "PIXELATION": "PIXELATION",
    "JPEG_ARTIFACT": "JPEG_ARTIFACT",
    "NOISE": None,
    "MIXED_DAMAGE": None,
    "HEALTHY": "HEALTHY",
}


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    damage_type: str
    position: str
    size: str
    severity: str
    opacity: str
    expected_legacy_class: str | None
    binary_damage_expected: bool


def build_matrix() -> list[EvaluationCase]:
    rows: list[EvaluationCase] = []
    position_index = 0
    for damage_type in EVALUATION_DAMAGE_TYPES:
        if damage_type == "HEALTHY":
            rows.append(
                EvaluationCase(
                    case_id="HEALTHY-001",
                    damage_type=damage_type,
                    position="GLOBAL",
                    size="NONE",
                    severity="NONE",
                    opacity="NONE",
                    expected_legacy_class="HEALTHY",
                    binary_damage_expected=False,
                )
            )
            continue
        for profile_index, (size, severity) in enumerate(zip(SIZES, SEVERITIES), start=1):
            global_type = damage_type in {"BLUR_GLOBAL", "JPEG_ARTIFACT", "NOISE"}
            position = "GLOBAL" if global_type else POSITIONS[position_index % len(POSITIONS)]
            position_index += 1
            if damage_type == "OPAQUE_STICKER":
                opacity = "OPAQUE"
            elif damage_type == "TRANSLUCENT_STICKER":
                opacity = ("LOW", "MEDIUM", "HIGH")[profile_index - 1]
            else:
                opacity = "N/A"
            rows.append(
                EvaluationCase(
                    case_id=f"{damage_type}-{profile_index:03d}",
                    damage_type=damage_type,
                    position=position,
                    size=size,
                    severity=severity,
                    opacity=opacity,
                    expected_legacy_class=LEGACY_CLASS_FOR_TYPE[damage_type],
                    binary_damage_expected=True,
                )
            )
    return rows


def binary_metrics(*, true_positive: int, false_positive: int, false_negative: int, true_negative: int) -> dict[str, float | int]:
    tp = int(true_positive)
    fp = int(false_positive)
    fn = int(false_negative)
    tn = int(true_negative)
    if min(tp, fp, fn, tn) < 0:
        raise ValueError("confusion counts must be non-negative")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
    }


def phase04_gate(report: Mapping[str, object]) -> dict[str, object]:
    binary = report.get("binary")
    groups = report.get("groups")
    if not isinstance(binary, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("report must contain binary and groups mappings")

    def metric(mapping: Mapping[str, object], name: str) -> float:
        value = mapping.get(name)
        if not isinstance(value, (int, float)):
            raise ValueError(f"missing numeric metric: {name}")
        return float(value)

    sticker = groups.get("STICKER")
    scribble = groups.get("SCRIBBLE")
    motion = groups.get("MOTION_BLUR")
    local_blur = groups.get("BLUR_LOCAL")
    critical = groups.get("CRITICAL_MIN")
    if not all(isinstance(value, Mapping) for value in (sticker, scribble, motion, local_blur, critical)):
        raise ValueError("report is missing required Phase04 metric groups")

    checks = {
        "binary_precision_gte_0_95": metric(binary, "precision") >= 0.95,
        "binary_recall_gte_0_90": metric(binary, "recall") >= 0.90,
        "sticker_f1_gte_0_90": metric(sticker, "f1") >= 0.90,
        "scribble_f1_gte_0_90": metric(scribble, "f1") >= 0.90,
        "motion_blur_f1_gte_0_85": metric(motion, "f1") >= 0.85,
        "local_blur_f1_gte_0_85": metric(local_blur, "f1") >= 0.85,
        "critical_min_f1_gt_0": metric(critical, "f1") > 0.0,
    }
    return {
        "thresholds_frozen_before_measurement": True,
        "checks": checks,
        "passed": all(checks.values()),
    }


def matrix_payload() -> dict[str, object]:
    rows = build_matrix()
    return {
        "phase": "PHASE_04_DAMAGE_MASK",
        "purpose": "predeclared evaluation matrix; no model scores are fabricated here",
        "final_holdout_used": False,
        "v3_used": False,
        "v4_used": False,
        "production_qualified": False,
        "required_gate": {
            "binary_precision_min": 0.95,
            "binary_recall_min": 0.90,
            "sticker_f1_min": 0.90,
            "scribble_f1_min": 0.90,
            "motion_blur_f1_min": 0.85,
            "local_blur_f1_min": 0.85,
            "critical_class_f1_must_be_nonzero": True,
        },
        "coverage": {
            "damage_types": list(EVALUATION_DAMAGE_TYPES),
            "positions": list(POSITIONS) + ["GLOBAL"],
            "sizes": list(SIZES),
            "severities": list(SEVERITIES),
            "translucent_opacity_levels": ["LOW", "MEDIUM", "HIGH"],
        },
        "case_count": len(rows),
        "cases": [asdict(row) for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = matrix_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"case_count": payload["case_count"], "gate": payload["required_gate"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
