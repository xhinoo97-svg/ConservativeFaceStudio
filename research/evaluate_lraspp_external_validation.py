from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from app.resource_budget import (
    apply_resource_budget,
    assert_memory_within_budget,
    detect_resource_budget,
    resource_snapshot,
)
from damage_mask_dataset import DAMAGE_CLASSES, load_source_manifest
from damage_mask_lraspp import LRASPPDamageModel
from damage_mask_lraspp_contract import (
    BACKBONE_WEIGHTS_LICENSE,
    UPSTREAM_REVISION,
    development_gate,
    sha256_path,
    verify_file,
)
from train_damage_mask_net import (
    SyntheticDamageDataset,
    _loader,
    confusion_update,
    metrics_from_confusion,
    save_visual,
)


def _damage_scores(per_class: dict[str, dict[str, float | int | None]]) -> tuple[list[float], list[float]]:
    f1 = [float(per_class[name]["f1"] or 0.0) for name in DAMAGE_CLASSES[1:]]
    iou = [float(per_class[name]["iou"] or 0.0) for name in DAMAGE_CLASSES[1:]]
    return f1, iou


def binary_damage_metrics(matrix: np.ndarray) -> dict[str, float | int]:
    true_positive = int(matrix[1:, 1:].sum())
    false_negative = int(matrix[1:, 0].sum())
    false_positive = int(matrix[0, 1:].sum())
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(true_positive / union) if union else 0.0,
    }


def summarize_matrix(matrix: np.ndarray) -> dict[str, object]:
    per_class = metrics_from_confusion(matrix)
    damage_f1, damage_iou = _damage_scores(per_class)
    return {
        "per_class": per_class,
        "damage_macro_f1": float(np.mean(damage_f1)) if damage_f1 else 0.0,
        "damage_macro_iou": float(np.mean(damage_iou)) if damage_iou else 0.0,
        "minimum_damage_class_f1": min(damage_f1) if damage_f1 else 0.0,
        "binary_damage_mask": binary_damage_metrics(matrix),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--expected-onnx-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--face-size", type=int, default=192)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=240824)
    args = parser.parse_args()

    budget = detect_resource_budget(0.80)
    apply_resource_budget(budget)
    assert_memory_within_budget(budget, stage="lraspp_external_validation_start")

    import onnxruntime as ort
    import torch

    manifest_payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("identity_disjoint_from_prior_lraspp_bank") is not True:
        raise RuntimeError("External validation manifest is not identity-disjoint")
    if manifest_payload.get("training_or_tuning_authorized") is not False:
        raise RuntimeError("External validation manifest unexpectedly authorizes tuning")
    if manifest_payload.get("final_holdout_used") is not False:
        raise RuntimeError("Final holdout material is forbidden")

    checkpoint_sha256 = verify_file(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
    )
    onnx_sha256 = verify_file(args.onnx, expected_sha256=args.expected_onnx_sha256)
    records = load_source_manifest(args.manifest)
    if len(records) != 40:
        raise RuntimeError(f"Expected exactly 40 external validation identities, got {len(records)}")
    metadata_by_source = {
        str(row["source_id"]): {
            "sex": str(row["subject_domain"]),
            "race": str(row["race_domain"]),
            "age": str(row["age_domain"]),
        }
        for row in manifest_payload["sources"]
    }

    dataset = SyntheticDamageDataset(
        records,
        args.source_dir,
        face_size=args.face_size,
        repetitions_per_class=args.repetitions,
        base_seed=args.seed,
    )
    loader = _loader(dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)
    model = LRASPPDamageModel.from_trained_checkpoint(
        checkpoint=args.checkpoint,
        expected_sha256=checkpoint_sha256,
        classes=len(DAMAGE_CLASSES),
    )
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])

    matrix = np.zeros((len(DAMAGE_CLASSES), len(DAMAGE_CLASSES)), dtype=np.int64)
    domain_matrices: dict[str, np.ndarray] = defaultdict(
        lambda: np.zeros((len(DAMAGE_CLASSES), len(DAMAGE_CLASSES)), dtype=np.int64)
    )
    visual = None
    torch_onnx_max_abs = 0.0
    torch_onnx_argmax_equal = True
    total_onnx_seconds = 0.0
    completed = 0
    model.eval()
    with torch.inference_mode():
        for batch_index, (images, masks, source_ids, *_rest) in enumerate(loader):
            if batch_index == 0:
                torch_logits = model(images).detach().cpu().numpy()
            start = time.perf_counter()
            onnx_logits = session.run(["logits"], {"image": images.numpy()})[0]
            total_onnx_seconds += time.perf_counter() - start
            predictions = onnx_logits.argmax(axis=1)
            target = masks.numpy()
            if batch_index == 0:
                torch_onnx_max_abs = float(np.max(np.abs(torch_logits - onnx_logits)))
                torch_onnx_argmax_equal = bool(
                    np.array_equal(torch_logits.argmax(axis=1), predictions)
                )
            for prediction, truth, source_id in zip(predictions, target, source_ids):
                confusion_update(matrix, prediction, truth)
                metadata = metadata_by_source[str(source_id)]
                for key, value in metadata.items():
                    confusion_update(domain_matrices[f"{key}:{value}"], prediction, truth)
                if visual is None:
                    visual = (images[0].numpy(), truth, prediction)
                completed += 1
            assert_memory_within_budget(
                budget,
                stage=f"lraspp_external_validation_batch_{batch_index}",
            )

    if visual is None or completed != len(dataset):
        raise RuntimeError(f"External validation incomplete: {completed}/{len(dataset)}")
    if not torch_onnx_argmax_equal or torch_onnx_max_abs >= 1e-3:
        raise RuntimeError(
            f"Checkpoint/ONNX parity failed: equal={torch_onnx_argmax_equal} "
            f"max_abs={torch_onnx_max_abs}"
        )

    overall = summarize_matrix(matrix)
    gate = development_gate(
        damage_macro_f1=float(overall["damage_macro_f1"]),
        damage_macro_iou=float(overall["damage_macro_iou"]),
        per_damage_class_f1=[
            float(overall["per_class"][name]["f1"] or 0.0)
            for name in DAMAGE_CLASSES[1:]
        ],
    )
    domain_summaries = {
        key: summarize_matrix(value)
        for key, value in sorted(domain_matrices.items())
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_visual(args.output / "external_validation_example.png", visual)
    report = {
        "experiment": "lraspp_frozen_checkpoint_external_development_validation_v1",
        "qualification_scope": "external_development_validation_not_final_holdout",
        "production_qualified": False,
        "disposition": (
            "EXTERNAL_DEVELOPMENT_VALIDATION_PASS_NOT_PRODUCTION_QUALIFIED"
            if bool(gate["passed"])
            else "EXTERNAL_DEVELOPMENT_VALIDATION_FAIL"
        ),
        "model": {
            "architecture": "official_torchvision_LRASPP_MobileNetV3_Large",
            "upstream_revision": UPSTREAM_REVISION,
            "checkpoint_sha256": checkpoint_sha256,
            "onnx_sha256": onnx_sha256,
            "checkpoint_weights_license": BACKBONE_WEIGHTS_LICENSE,
            "retrained_or_tuned": False,
        },
        "data": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_path(args.manifest),
            "dataset": manifest_payload["dataset"],
            "selection": manifest_payload["selection"],
            "identity_count": len(records),
            "identity_disjoint_from_prior_lraspp_bank": True,
            "repetitions_per_damage_class": int(args.repetitions),
            "completed_cases": completed,
            "expected_cases": len(dataset),
            "error_cases": 0,
            "final_holdout_used": False,
        },
        "validation": {
            **overall,
            "frozen_development_gate": gate,
            "per_domain": domain_summaries,
        },
        "runtime": {
            "onnx_cpu_total_seconds": float(total_onnx_seconds),
            "onnx_cpu_seconds_per_face": float(total_onnx_seconds / completed),
            "checkpoint_onnx_first_batch_max_abs": torch_onnx_max_abs,
            "checkpoint_onnx_first_batch_argmax_equal": torch_onnx_argmax_equal,
            "resource_budget": resource_snapshot(budget),
            "network_required_after_artifact_and_dataset_acquisition": False,
        },
        "safety": {
            "model_output_type": "mask_logits_only",
            "can_modify_image_pixels": False,
            "wrong_person_final_pixels": 0,
            "provenance_violations": 0,
            "refface_execution_authorized": False,
            "rollback_count": 0,
            "abstention_count": 0,
            "restoration_pass_count": 0,
        },
        "production_blockers": [
            "validation domain is synthetic ControlFace only",
            "checkpoint redistribution license is not explicit",
            "Windows and HP EliteBook execution not measured",
            "300-400 identity predominantly female benchmark not run",
        ],
        "target95": "NOT_MEASURED",
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
