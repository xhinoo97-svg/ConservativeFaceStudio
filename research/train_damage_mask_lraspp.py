from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import numpy as np

from app.resource_budget import (
    apply_resource_budget,
    assert_memory_within_budget,
    detect_resource_budget,
    resource_snapshot,
)
from damage_mask_dataset import DAMAGE_CLASSES, load_source_manifest
from damage_mask_lraspp import LRASPPDamageModel, parameter_count
from damage_mask_lraspp_contract import (
    BACKBONE_BYTES,
    BACKBONE_SHA256,
    BACKBONE_URL,
    BACKBONE_WEIGHTS_LICENSE,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    UPSTREAM_CODE_LICENSE,
    UPSTREAM_LICENSE_SHA256,
    UPSTREAM_REPOSITORY,
    UPSTREAM_REVISION,
    UPSTREAM_TAG,
    development_gate,
    sha256_path,
)
from train_damage_mask_net import (
    SyntheticDamageDataset,
    _loader,
    evaluate,
    metrics_from_confusion,
    save_visual,
    train_epoch,
)


def set_deterministic(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _damage_scores(per_class: dict[str, dict[str, float | int | None]]) -> tuple[list[float], list[float]]:
    f1 = [
        float(per_class[name]["f1"] or 0.0)
        for name in DAMAGE_CLASSES[1:]
    ]
    iou = [
        float(per_class[name]["iou"] or 0.0)
        for name in DAMAGE_CLASSES[1:]
    ]
    return f1, iou


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--backbone-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--face-size", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-repetitions", type=int, default=6)
    parser.add_argument("--val-repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=240823)
    args = parser.parse_args()

    set_deterministic(args.seed)
    budget = detect_resource_budget(0.80)
    apply_resource_budget(budget)
    assert_memory_within_budget(budget, stage="damage_mask_lraspp_start")

    import onnx
    import onnxruntime as ort
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    records = load_source_manifest(args.manifest)
    if len(records) < 4:
        raise RuntimeError("Need at least four identities for a development split")
    split_index = max(2, len(records) - 2)
    train_records = records[:split_index]
    validation_records = records[split_index:]
    if {row.source_id for row in train_records} & {row.source_id for row in validation_records}:
        raise RuntimeError("Identity leakage in LR-ASPP development split")

    train_dataset = SyntheticDamageDataset(
        train_records,
        args.source_dir,
        face_size=args.face_size,
        repetitions_per_class=args.train_repetitions,
        base_seed=args.seed,
    )
    validation_dataset = SyntheticDamageDataset(
        validation_records,
        args.source_dir,
        face_size=args.face_size,
        repetitions_per_class=args.val_repetitions,
        base_seed=args.seed + 90_000_000,
    )
    train_loader = _loader(train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = _loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )

    model = LRASPPDamageModel(
        classes=len(DAMAGE_CLASSES),
        backbone_checkpoint=args.backbone_checkpoint,
    ).to("cpu")
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.network.backbone.parameters(), "lr": 2e-4},
            {"params": model.network.classifier.parameters(), "lr": 2e-3},
        ],
        weight_decay=1e-4,
    )
    class_weights = torch.tensor(
        [0.12] + [2.0] * (len(DAMAGE_CLASSES) - 1),
        dtype=torch.float32,
    )

    history: list[dict[str, float | int]] = []
    best_state = None
    best_metrics = None
    best_visual = None
    best_damage_macro_f1 = -1.0
    training_start = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_epoch(model, train_loader, optimizer, class_weights)
        validation_loss, matrix, visual = evaluate(model, validation_loader)
        per_class = metrics_from_confusion(matrix)
        damage_f1, _ = _damage_scores(per_class)
        damage_macro_f1 = float(np.mean(damage_f1)) if damage_f1 else 0.0
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "damage_macro_f1": damage_macro_f1,
            }
        )
        if damage_macro_f1 > best_damage_macro_f1:
            best_damage_macro_f1 = damage_macro_f1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_metrics = per_class
            best_visual = visual
        assert_memory_within_budget(budget, stage=f"damage_mask_lraspp_epoch_{epoch}")
    training_seconds = time.perf_counter() - training_start

    if best_state is None or best_metrics is None or best_visual is None:
        raise RuntimeError("LR-ASPP training produced no best checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()

    checkpoint_path = args.output / "damage_mask_lraspp_dev.pth"
    torch.save(
        {
            "state_dict": best_state,
            "classes": list(DAMAGE_CLASSES),
            "face_size": int(args.face_size),
            "backbone_sha256": BACKBONE_SHA256,
            "upstream_revision": UPSTREAM_REVISION,
            "development_only": True,
        },
        checkpoint_path,
    )
    checkpoint_sha256 = sha256_path(checkpoint_path)
    save_visual(args.output / "validation_example.png", best_visual)

    image_chw, _, _ = best_visual
    sample_tensor = torch.from_numpy(image_chw).unsqueeze(0).float()
    reloaded = LRASPPDamageModel.from_trained_checkpoint(
        checkpoint=checkpoint_path,
        expected_sha256=checkpoint_sha256,
        classes=len(DAMAGE_CLASSES),
    )
    with torch.inference_mode():
        original_logits = model(sample_tensor)
        reload_start = time.perf_counter()
        reload_logits = reloaded(sample_tensor)
        torch_cpu_seconds = time.perf_counter() - reload_start
    reload_max_abs = float(
        np.max(
            np.abs(
                original_logits.detach().cpu().numpy()
                - reload_logits.detach().cpu().numpy()
            )
        )
    )
    if reload_max_abs != 0.0:
        raise RuntimeError(f"Offline trained-checkpoint reload drift: {reload_max_abs}")
    assert_memory_within_budget(budget, stage="damage_mask_lraspp_torch_inference")

    onnx_path = args.output / "damage_mask_lraspp_dev.onnx"
    torch.onnx.export(
        reloaded,
        sample_tensor,
        str(onnx_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    )
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    onnx_sha256 = sha256_path(onnx_path)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_start = time.perf_counter()
    ort_logits = session.run(["logits"], {"image": sample_tensor.numpy()})[0]
    ort_seconds = time.perf_counter() - ort_start
    torch_logits = reload_logits.detach().cpu().numpy()
    max_abs = float(np.max(np.abs(torch_logits - ort_logits)))
    mean_abs = float(np.mean(np.abs(torch_logits - ort_logits)))
    segmentation_equal = bool(
        np.array_equal(torch_logits.argmax(axis=1), ort_logits.argmax(axis=1))
    )

    damage_f1, damage_iou = _damage_scores(best_metrics)
    damage_macro_f1 = float(np.mean(damage_f1)) if damage_f1 else 0.0
    damage_macro_iou = float(np.mean(damage_iou)) if damage_iou else 0.0
    gate = development_gate(
        damage_macro_f1=damage_macro_f1,
        damage_macro_iou=damage_macro_iou,
        per_damage_class_f1=damage_f1,
    )
    parity_pass = bool(segmentation_equal and max_abs < 1e-3)
    disposition = (
        "DEVELOPMENT_MASK_ADEQUACY_PASS_NOT_PRODUCTION_QUALIFIED"
        if bool(gate["passed"]) and parity_pass
        else "MODEL_DATA_QUALITY_FAIL"
    )

    report = {
        "experiment": "damage_mask_lraspp_mobilenetv3_development_comparison_v1",
        "qualification_scope": "development_only_not_final_holdout",
        "production_qualified": False,
        "disposition": disposition,
        "architecture": {
            "name": "LRASPP_MobileNet_V3_Large",
            "source": "official_torchvision",
            "parameter_count": int(params),
            "input_face_size": int(args.face_size),
            "classes": list(DAMAGE_CLASSES),
            "loaded_backbone_tensor_count": int(model.loaded_backbone_tensor_count),
        },
        "upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "tag": UPSTREAM_TAG,
            "revision": UPSTREAM_REVISION,
            "code_license": UPSTREAM_CODE_LICENSE,
            "license_sha256": UPSTREAM_LICENSE_SHA256,
            "torch": TORCH_VERSION,
            "torchvision": TORCHVISION_VERSION,
            "backbone_url": BACKBONE_URL,
            "backbone_bytes": BACKBONE_BYTES,
            "backbone_sha256": BACKBONE_SHA256,
            "backbone_weights_license": BACKBONE_WEIGHTS_LICENSE,
            "redistribution_qualified": False,
        },
        "data": {
            "manifest": str(args.manifest),
            "train_identities": [row.source_id for row in train_records],
            "validation_identities": [row.source_id for row in validation_records],
            "identity_disjoint": True,
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "exact_synthetic_masks": True,
            "final_holdout_used": False,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "training_seconds": float(training_seconds),
            "history": history,
            "best_damage_macro_f1": float(best_damage_macro_f1),
        },
        "validation": {
            "per_class": best_metrics,
            "damage_macro_iou": damage_macro_iou,
            "damage_macro_f1": damage_macro_f1,
            "minimum_damage_class_f1": min(damage_f1) if damage_f1 else 0.0,
            "development_gate": gate,
        },
        "runtime": {
            "torch_cpu_seconds_single_face": float(torch_cpu_seconds),
            "onnxruntime_cpu_seconds_single_face_first_call": float(ort_seconds),
            "resource_budget": resource_snapshot(budget),
            "offline_trained_checkpoint_reload_max_abs": reload_max_abs,
            "network_required_after_artifact_acquisition": False,
        },
        "onnx_parity": {
            "opset": 17,
            "fixed_spatial_input": [int(args.face_size), int(args.face_size)],
            "max_abs_logit_difference": max_abs,
            "mean_abs_logit_difference": mean_abs,
            "argmax_segmentation_equal": segmentation_equal,
            "passed": parity_pass,
        },
        "safety": {
            "model_output_type": "mask_logits_only",
            "wrong_person_final_pixels": 0,
            "provenance_violations": 0,
            "can_modify_image_pixels": False,
            "refface_execution_authorized": False,
            "reason": "Development comparison only; two validation identities and weights license unresolved",
        },
        "artifacts": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha256,
            "onnx": onnx_path.name,
            "onnx_sha256": onnx_sha256,
            "validation_example": "validation_example.png",
        },
        "next_gate": (
            "If DEVELOPMENT adequacy passes, repeat on a substantially larger identity-disjoint "
            "validation bank and resolve checkpoint licensing before any RefFace or product integration."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not parity_pass:
        raise RuntimeError(
            f"ONNX parity failed: equal={segmentation_equal} max_abs={max_abs}"
        )

    del session, reloaded, model
    gc.collect()
    assert_memory_within_budget(budget, stage="damage_mask_lraspp_end")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
