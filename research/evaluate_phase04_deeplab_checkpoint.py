from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from damage_mask_dataset import SourceRecord, load_face_crop  # noqa: E402
from phase04_damage_evaluation import build_matrix, phase04_gate  # noqa: E402
from phase04_deeplab_challenger import ARCHITECTURE, Phase04DeepLabDamageModel  # noqa: E402
from phase04_training_dataset import (  # noqa: E402
    PHASE04_HEALTHY_INDEX,
    PHASE04_TRAINING_CLASSES,
    build_training_sample,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_records(manifest: Path) -> tuple[list[SourceRecord], dict[str, Any]]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("final_holdout_used") is True:
        raise RuntimeError("Final holdout is forbidden for Phase04 challenger development evaluation")
    if payload.get("v3_used") is True or payload.get("v4_used") is True:
        raise RuntimeError("V3/V4 material is forbidden for Phase04 challenger development evaluation")
    rows = payload.get("sources")
    if not isinstance(rows, list):
        raise ValueError("source manifest has no sources list")

    train_ids: set[str] = set()
    validation_ids: set[str] = set()
    selected: list[SourceRecord] = []
    validation_source_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        split = str(row.get("dataset_split", ""))
        identity = str(row.get("identity_key", row.get("source_id", "")))
        if not identity:
            raise ValueError("source row has no identity key")
        if split == "train":
            train_ids.add(identity)
        elif split == "validation":
            validation_ids.add(identity)
        if split != "validation":
            continue
        bbox = row.get("face_bbox_normalized")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid face bbox for {row.get('source_id')}")
        source_id = str(row["source_id"])
        validation_source_ids.append(source_id)
        selected.append(
            SourceRecord(
                source_id=source_id,
                filename=str(row["filename"]),
                clean_source_sha256=str(row["clean_source_sha256"]).lower(),
                face_bbox_normalized=tuple(float(value) for value in bbox),
            )
        )
    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError(f"Identity leakage between train and validation: {sorted(overlap)}")
    if not selected:
        raise RuntimeError("No validation sources in manifest")
    meta = {
        "train_identity_count": len(train_ids),
        "validation_identity_count": len(validation_ids),
        "validation_source_ids": validation_source_ids,
        "identity_disjoint": True,
    }
    return selected, meta


def _as_input(image: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    array = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...])
    return torch.from_numpy(array).float()


def _confusion(predicted: np.ndarray, truth: np.ndarray) -> tuple[int, int, int, int]:
    pred = predicted.astype(bool)
    target = truth.astype(bool)
    tp = int(np.count_nonzero(pred & target))
    fp = int(np.count_nonzero(pred & ~target))
    fn = int(np.count_nonzero(~pred & target))
    tn = int(np.count_nonzero(~pred & ~target))
    return tp, fp, fn, tn


def _add_counts(target: dict[str, int], counts: tuple[int, int, int, int]) -> None:
    for key, value in zip(("tp", "fp", "fn", "tn"), counts):
        target[key] += int(value)


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (int(counts[key]) for key in ("tp", "fp", "fn", "tn"))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    union = tp + fp + fn
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(tp / union) if union else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
    }


def _group_for_type(damage_type: str) -> tuple[str, ...]:
    groups: list[str] = []
    if damage_type in {"OPAQUE_STICKER", "TRANSLUCENT_STICKER", "EMOJI"}:
        groups.append("STICKER")
    if damage_type.startswith("SCRIBBLE_"):
        groups.append("SCRIBBLE")
    if damage_type == "MOTION_BLUR":
        groups.append("MOTION_BLUR")
    if damage_type == "BLUR_LOCAL":
        groups.append("BLUR_LOCAL")
    return tuple(groups)


def evaluate(
    *,
    checkpoint: Path,
    source_dir: Path,
    manifest: Path,
    output: Path,
    threshold: float,
    image_size: int,
    seed: int,
) -> dict[str, Any]:
    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be in (0,1)")
    if image_size < 64:
        raise ValueError("image_size must be >= 64")
    records, split_meta = _validation_records(manifest)
    checkpoint_sha256 = _sha256(checkpoint)
    checkpoint_payload = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint_payload, dict):
        raise RuntimeError("DeepLab checkpoint payload is invalid")
    training_meta = checkpoint_payload.get("training")
    if not isinstance(training_meta, dict):
        raise RuntimeError("DeepLab checkpoint has no training provenance")
    if training_meta.get("final_holdout_used") is True:
        raise RuntimeError("Checkpoint provenance indicates final holdout use")
    if training_meta.get("v3_used") is True or training_meta.get("v4_used") is True:
        raise RuntimeError("Checkpoint provenance indicates forbidden V3/V4 use")

    model = Phase04DeepLabDamageModel.from_trained_checkpoint(
        checkpoint=checkpoint,
        expected_sha256=checkpoint_sha256,
    ).eval()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    faces = {
        record.source_id: load_face_crop(record, source_dir, size=image_size)
        for record in records
    }
    cases = build_matrix()
    overall_counts: dict[str, int] = defaultdict(int)
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    case_rows: list[dict[str, Any]] = []
    total_inference_seconds = 0.0

    with torch.inference_mode():
        for source_index, record in enumerate(records):
            clean = faces[record.source_id]
            for case_index, case in enumerate(cases):
                sample_seed = int(seed + source_index * 100_000 + case_index * 101)
                sample = build_training_sample(
                    clean,
                    case,
                    seed=sample_seed,
                    source_id=record.source_id,
                )
                tensor = _as_input(sample.image)
                started = time.perf_counter()
                logits = model(tensor)
                total_inference_seconds += time.perf_counter() - started
                expected_shape = (1, len(PHASE04_TRAINING_CLASSES), image_size, image_size)
                if tuple(logits.shape) != expected_shape:
                    raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)} != {expected_shape}")
                probabilities = torch.softmax(logits, dim=1)[0]
                confidence, class_map = torch.max(probabilities, dim=0)
                predicted = (
                    (class_map.cpu().numpy() != int(PHASE04_HEALTHY_INDEX))
                    & (confidence.cpu().numpy() >= float(threshold))
                )
                truth = sample.binary_mask > 0
                counts = _confusion(predicted, truth)
                _add_counts(overall_counts, counts)
                _add_counts(type_counts[case.damage_type], counts)
                for group in _group_for_type(case.damage_type):
                    _add_counts(group_counts[group], counts)
                case_metrics = _metrics(
                    {key: value for key, value in zip(("tp", "fp", "fn", "tn"), counts)}
                )
                case_rows.append(
                    {
                        "source_id": record.source_id,
                        "case_id": case.case_id,
                        "damage_type": case.damage_type,
                        "position": case.position,
                        "size": case.size,
                        "severity": case.severity,
                        "opacity": case.opacity,
                        "metrics": case_metrics,
                    }
                )

    overall = _metrics(overall_counts)
    per_type = {key: _metrics(value) for key, value in sorted(type_counts.items())}
    groups = {key: _metrics(value) for key, value in sorted(group_counts.items())}
    critical_types = [key for key in per_type if key != "HEALTHY"]
    if not critical_types:
        raise RuntimeError("No damage types were measured")
    critical_min_type = min(critical_types, key=lambda key: float(per_type[key]["f1"]))
    groups["CRITICAL_MIN"] = dict(per_type[critical_min_type])
    groups["CRITICAL_MIN"]["damage_type"] = critical_min_type
    gate = phase04_gate({"binary": overall, "groups": groups})

    report: dict[str, Any] = {
        "experiment": "phase04_deeplab_identity_disjoint_development_v1",
        "qualification_scope": "development_measurement_not_final_holdout",
        "production_qualified": False,
        "phase04_gate_passed_on_this_development_set": bool(gate["passed"]),
        "disposition": "DEVELOPMENT_GATE_PASS" if gate["passed"] else "DEVELOPMENT_GATE_FAIL",
        "model": {
            "architecture": ARCHITECTURE,
            "checkpoint": str(checkpoint),
            "sha256": checkpoint_sha256,
            "healthy_index": int(PHASE04_HEALTHY_INDEX),
            "classes": list(PHASE04_TRAINING_CLASSES),
            "runtime_damage_confidence_threshold": float(threshold),
            "training": training_meta,
        },
        "data": {
            **split_meta,
            "matrix_case_count_per_identity": len(cases),
            "completed_cases": len(case_rows),
            "expected_cases": len(records) * len(cases),
            "error_cases": 0,
            "v3_used": False,
            "v4_used": False,
            "v5_used": False,
            "final_holdout_used": False,
            "training_or_tuning_authorized": False,
        },
        "metrics": {
            "binary": overall,
            "groups": groups,
            "per_damage_type": per_type,
        },
        "frozen_phase04_gate": gate,
        "runtime": {
            "device": "torch_cpu",
            "total_inference_seconds": float(total_inference_seconds),
            "seconds_per_case": float(total_inference_seconds / max(1, len(case_rows))),
            "network_required_after_checkpoint_and_source_acquisition": False,
        },
        "safety": {
            "model_output_type": "mask_logits_only",
            "can_modify_image_pixels": False,
            "wrong_person_final_pixels": 0,
            "provenance_violations": 0,
            "restoration_pass_count": 0,
        },
        "production_blockers": [
            "development validation set is not the final 300-400 identity benchmark",
            "checkpoint is not production-qualified",
            "physical HP EliteBook execution not measured",
        ],
        "cases": case_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=240905)
    args = parser.parse_args()
    report = evaluate(
        checkpoint=args.checkpoint,
        source_dir=args.source_dir,
        manifest=args.manifest,
        output=args.output,
        threshold=args.threshold,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "disposition": report["disposition"],
                "binary": report["metrics"]["binary"],
                "groups": report["metrics"]["groups"],
                "gate": report["frozen_phase04_gate"],
                "completed_cases": report["data"]["completed_cases"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
