from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from damage_mask_dataset import SourceRecord, load_face_crop
from phase04_balanced_sampler import balanced_training_pairs
from phase04_damage_evaluation import build_matrix
from phase04_deeplab_challenger import ARCHITECTURE, Phase04DeepLabDamageModel
from phase04_training_dataset import (
    PHASE04_HEALTHY_INDEX,
    PHASE04_TRAINING_CLASSES,
    build_training_sample,
)

LOSS_VERSION = "region_balanced_ce_binary_focal_dice_v2"


def _records_for_split(manifest: Path, split: str) -> list[SourceRecord]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("final_holdout_used") is True:
        raise RuntimeError("Final holdout is forbidden for Phase04 challenger training")
    if payload.get("v3_used") is True or payload.get("v4_used") is True:
        raise RuntimeError("V3/V4 material is forbidden for Phase04 challenger training")
    rows = payload.get("sources")
    if not isinstance(rows, list):
        raise ValueError("source manifest has no sources list")

    selected: list[SourceRecord] = []
    train_identity_keys: set[str] = set()
    validation_identity_keys: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        identity_key = str(row.get("identity_key", row.get("source_id", "")))
        row_split = str(row.get("dataset_split", ""))
        if row_split == "train":
            train_identity_keys.add(identity_key)
        elif row_split == "validation":
            validation_identity_keys.add(identity_key)
        if row_split != split:
            continue
        bbox = row.get("face_bbox_normalized")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid face bbox for {row.get('source_id')}")
        selected.append(
            SourceRecord(
                source_id=str(row["source_id"]),
                filename=str(row["filename"]),
                clean_source_sha256=str(row["clean_source_sha256"]).lower(),
                face_bbox_normalized=tuple(float(value) for value in bbox),
            )
        )
    if train_identity_keys & validation_identity_keys:
        raise RuntimeError("Identity leakage between train and validation source records")
    if not selected:
        raise RuntimeError(f"No source records for split={split}")
    return selected


def _as_input(images: list[np.ndarray]) -> torch.Tensor:
    arrays = []
    for image in images:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        arrays.append(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
    return torch.from_numpy(np.stack(arrays, axis=0)).float()


def _region_mean(values: torch.Tensor, selected: torch.Tensor) -> torch.Tensor | None:
    if values.shape != selected.shape:
        raise ValueError("values and selected masks must have the same shape")
    count = int(torch.count_nonzero(selected).detach().cpu())
    if count == 0:
        return None
    return values[selected].mean()


def _region_balanced_mean(values: torch.Tensor, damage: torch.Tensor) -> torch.Tensor:
    damage_mean = _region_mean(values, damage)
    healthy_mean = _region_mean(values, ~damage)
    if damage_mean is not None and healthy_mean is not None:
        return 0.5 * (damage_mean + healthy_mean)
    if damage_mean is not None:
        return damage_mean
    if healthy_mean is not None:
        return healthy_mean
    raise RuntimeError("segmentation target contains no pixels")


def _training_case_type_weights(cases) -> dict[str, float]:
    counts = Counter(case.damage_type for case in cases if case.damage_type != "HEALTHY")
    if not counts:
        raise ValueError("Phase04 matrix has no damage cases")
    ordered = sorted(int(value) for value in counts.values())
    reference = float(ordered[len(ordered) // 2])
    weights: dict[str, float] = {"HEALTHY": 1.0}
    for damage_type, count in counts.items():
        value = math.sqrt(reference / float(count))
        weights[str(damage_type)] = float(min(3.0, max(0.5, value)))
    return weights


def _weighted_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("invalid segmentation tensor rank")
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("logits and target spatial shapes do not match")
    batch = int(target.shape[0])
    if sample_weights is None:
        sample_weights = torch.ones((batch,), dtype=logits.dtype, device=logits.device)
    else:
        sample_weights = sample_weights.to(device=logits.device, dtype=logits.dtype).reshape(-1)
        if int(sample_weights.numel()) != batch:
            raise ValueError("sample_weights length must match batch size")
        if torch.any(sample_weights <= 0):
            raise ValueError("sample_weights must be positive")

    per_pixel_ce = F.cross_entropy(logits, target, reduction="none")
    probabilities = torch.softmax(logits, dim=1)
    damage_probability = (1.0 - probabilities[:, int(PHASE04_HEALTHY_INDEX)]).clamp(1e-6, 1.0 - 1e-6)
    damage_target = target != int(PHASE04_HEALTHY_INDEX)

    sample_losses: list[torch.Tensor] = []
    for index in range(batch):
        damage = damage_target[index]
        region_ce = _region_balanced_mean(per_pixel_ce[index], damage)

        binary_target = damage.to(dtype=logits.dtype)
        binary_ce = F.binary_cross_entropy(damage_probability[index], binary_target, reduction="none")
        pt = torch.where(damage, damage_probability[index], 1.0 - damage_probability[index])
        focal = ((1.0 - pt).pow(2.0) * binary_ce)
        region_focal = _region_balanced_mean(focal, damage)

        intersection = torch.sum(damage_probability[index] * binary_target)
        denominator = torch.sum(damage_probability[index]) + torch.sum(binary_target)
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0))

        sample_loss = region_ce + 0.5 * region_focal + 0.5 * dice_loss
        sample_losses.append(sample_loss)

    stacked = torch.stack(sample_losses)
    return torch.sum(stacked * sample_weights) / torch.sum(sample_weights).clamp_min(1e-6)


def train(
    *,
    source_dir: Path,
    manifest: Path,
    backbone: Path,
    output: Path,
    report_path: Path,
    image_size: int,
    batch_size: int,
    max_steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    if image_size < 64:
        raise ValueError("image_size must be >= 64")
    if batch_size < 2:
        raise ValueError(
            "batch_size must be >= 2 for DeepLab training because the official ASPP head "
            "contains BatchNorm over a 1x1 pooled feature map"
        )
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

    records = _records_for_split(manifest, "train")
    faces = {
        record.source_id: load_face_crop(record, source_dir, size=image_size)
        for record in records
    }
    cases = build_matrix()
    case_type_weights = _training_case_type_weights(cases)
    sample_count = int(max_steps) * int(batch_size)
    training_pairs = balanced_training_pairs(
        [record.source_id for record in records],
        case_count=len(cases),
        seed=int(seed),
        sample_count=sample_count,
    )

    model = Phase04DeepLabDamageModel(backbone_checkpoint=backbone).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)

    losses: list[float] = []
    started = time.perf_counter()
    consumed = 0
    for step in range(max_steps):
        batch_pairs = training_pairs[consumed : consumed + batch_size]
        consumed += batch_size
        if len(batch_pairs) != batch_size:
            raise RuntimeError("balanced sampler returned an incomplete training batch")
        images: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        batch_weights: list[float] = []
        for local_index, pair in enumerate(batch_pairs):
            source_id = pair.source_id
            case_index = int(pair.case_index)
            case = cases[case_index]
            sample_seed = (
                int(seed)
                + step * 100_003
                + local_index * 1_009
                + case_index
                + int(pair.epoch) * 10_000_019
            )
            sample = build_training_sample(
                faces[source_id],
                case,
                seed=sample_seed,
                source_id=source_id,
            )
            images.append(sample.image)
            targets.append(sample.target)
            batch_weights.append(float(case_type_weights[case.damage_type]))
        image_tensor = _as_input(images)
        target_tensor = torch.from_numpy(np.stack(targets, axis=0)).long()
        weight_tensor = torch.tensor(batch_weights, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        loss = _weighted_segmentation_loss(logits, target_tensor, sample_weights=weight_tensor)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {float(loss.detach())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    elapsed = time.perf_counter() - started
    unique_case_count = len({int(pair.case_index) for pair in training_pairs})
    source_sample_counts = Counter(pair.source_id for pair in training_pairs)
    complete_matrix_epochs = sample_count // len(cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "architecture": ARCHITECTURE,
        "classes": list(PHASE04_TRAINING_CLASSES),
        "backbone_sha256": model.backbone_sha256,
        "state_dict": model.state_dict(),
        "training": {
            "qualification_scope": "DEVELOPMENT_TRAINING_NOT_FINAL_HOLDOUT",
            "final_holdout_used": False,
            "v3_used": False,
            "v4_used": False,
            "image_size": int(image_size),
            "batch_size": int(batch_size),
            "max_steps": int(max_steps),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "source_count": len(records),
            "batchnorm_safe_batch": True,
            "balanced_full_matrix_sampler": True,
            "loss_version": LOSS_VERSION,
            "case_type_weights": dict(sorted(case_type_weights.items())),
            "training_sample_count": int(sample_count),
            "unique_matrix_cases_seen": int(unique_case_count),
            "complete_matrix_epochs": int(complete_matrix_epochs),
            "source_sample_counts": dict(sorted(source_sample_counts.items())),
        },
    }
    torch.save(checkpoint, output)

    report: dict[str, object] = {
        "architecture": ARCHITECTURE,
        "classes": list(PHASE04_TRAINING_CLASSES),
        "source_count": len(records),
        "matrix_case_count": len(cases),
        "steps": int(max_steps),
        "batch_size": int(batch_size),
        "batchnorm_safe_batch": True,
        "balanced_full_matrix_sampler": True,
        "loss_version": LOSS_VERSION,
        "case_type_weights": dict(sorted(case_type_weights.items())),
        "training_sample_count": int(sample_count),
        "unique_matrix_cases_seen": int(unique_case_count),
        "complete_matrix_epochs": int(complete_matrix_epochs),
        "source_sample_counts": dict(sorted(source_sample_counts.items())),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "minimum_loss": float(min(losses)),
        "elapsed_seconds": float(elapsed),
        "checkpoint": str(output),
        "checkpoint_bytes": int(output.stat().st_size),
        "final_holdout_used": False,
        "v3_used": False,
        "v4_used": False,
        "production_qualified": False,
        "quality_gate_measured": False,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--backbone", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=240905)
    args = parser.parse_args()
    report = train(
        source_dir=args.source_dir,
        manifest=args.manifest,
        backbone=args.backbone,
        output=args.output,
        report_path=args.report,
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
