from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from damage_mask_dataset import SourceRecord, load_face_crop
from phase04_damage_evaluation import build_matrix
from phase04_deeplab_challenger import ARCHITECTURE, Phase04DeepLabDamageModel
from phase04_training_dataset import (
    PHASE04_HEALTHY_INDEX,
    PHASE04_TRAINING_CLASSES,
    build_training_sample,
)


def _records_for_split(manifest: Path, split: str) -> list[SourceRecord]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("final_holdout_used") is True:
        raise RuntimeError("Final holdout is forbidden for Phase04 challenger training")
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


def _weighted_segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or target.ndim != 3:
        raise ValueError("invalid segmentation tensor rank")
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    damage = target != int(PHASE04_HEALTHY_INDEX)
    # Healthy pixels dominate every local-damage image. Preserve them in the loss,
    # while giving damaged pixels enough authority to learn thin scribbles/edges.
    weights = torch.where(damage, torch.full_like(per_pixel, 6.0), torch.ones_like(per_pixel))
    return (per_pixel * weights).sum() / weights.sum().clamp_min(1.0)


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
    if batch_size < 1 or max_steps < 1:
        raise ValueError("batch_size and max_steps must be positive")
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
    pairs = [(record.source_id, case_index) for record in records for case_index in range(len(cases))]
    rng = random.Random(seed)
    rng.shuffle(pairs)

    model = Phase04DeepLabDamageModel(backbone_checkpoint=backbone).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)

    losses: list[float] = []
    started = time.perf_counter()
    consumed = 0
    for step in range(max_steps):
        batch_pairs = []
        for _ in range(batch_size):
            batch_pairs.append(pairs[consumed % len(pairs)])
            consumed += 1
        images: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for local_index, (source_id, case_index) in enumerate(batch_pairs):
            case = cases[case_index]
            sample_seed = seed + step * 100_003 + local_index * 1_009 + case_index
            sample = build_training_sample(
                faces[source_id],
                case,
                seed=sample_seed,
                source_id=source_id,
            )
            images.append(sample.image)
            targets.append(sample.target)
        image_tensor = _as_input(images)
        target_tensor = torch.from_numpy(np.stack(targets, axis=0)).long()

        optimizer.zero_grad(set_to_none=True)
        logits = model(image_tensor)
        loss = _weighted_segmentation_loss(logits, target_tensor)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {float(loss.detach())}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    elapsed = time.perf_counter() - started
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
