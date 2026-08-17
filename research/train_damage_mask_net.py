from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from app.resource_budget import (
    apply_resource_budget,
    assert_memory_within_budget,
    detect_resource_budget,
    resource_snapshot,
)
from damage_mask_dataset import (
    CLASS_TO_INDEX,
    DAMAGE_CLASSES,
    SourceRecord,
    apply_exact_damage,
    load_face_crop,
    load_source_manifest,
)
from damage_mask_net import DamageMaskUNet, parameter_count


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


class SyntheticDamageDataset:
    def __init__(
        self,
        records: list[SourceRecord],
        source_dir: Path,
        *,
        face_size: int,
        repetitions_per_class: int,
        base_seed: int,
    ) -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            pass

        self._dataset_base = _Dataset
        self.records = records
        self.face_size = int(face_size)
        self.repetitions_per_class = int(repetitions_per_class)
        self.base_seed = int(base_seed)
        self.faces = {
            record.source_id: load_face_crop(record, source_dir, size=self.face_size)
            for record in records
        }
        self.items: list[tuple[SourceRecord, str, int]] = []
        for source_index, record in enumerate(records):
            for class_index, damage_class in enumerate(DAMAGE_CLASSES[1:], start=1):
                for repetition in range(self.repetitions_per_class):
                    seed = (
                        self.base_seed
                        + source_index * 1000003
                        + class_index * 10007
                        + repetition * 997
                    )
                    self.items.append((record, damage_class, seed))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        import torch

        record, damage_class, seed = self.items[index]
        face = self.faces[record.source_id]
        sample = apply_exact_damage(face, damage_class, seed)
        rgb = cv2.cvtColor(sample.image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        mask = torch.from_numpy(sample.mask.astype(np.int64, copy=False))
        return image, mask, record.source_id, damage_class, int(seed)


def _loader(dataset: SyntheticDamageDataset, *, batch_size: int, shuffle: bool, seed: int):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )


def dice_loss(logits, target) -> object:
    import torch
    import torch.nn.functional as F

    probabilities = F.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=len(DAMAGE_CLASSES)).permute(0, 3, 1, 2).float()
    # Damage-only dice prevents the huge healthy background from hiding small masks.
    probs = probabilities[:, 1:]
    truth = one_hot[:, 1:]
    intersection = (probs * truth).sum(dim=(0, 2, 3))
    denominator = probs.sum(dim=(0, 2, 3)) + truth.sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def confusion_update(matrix: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> None:
    classes = len(DAMAGE_CLASSES)
    flat = target.reshape(-1).astype(np.int64) * classes + prediction.reshape(-1).astype(np.int64)
    counts = np.bincount(flat, minlength=classes * classes).reshape(classes, classes)
    matrix += counts


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(DAMAGE_CLASSES):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        union = tp + fp + fn
        denom_f1 = 2 * tp + fp + fn
        result[name] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'iou': float(tp / union) if union else None,
            'f1': float(2 * tp / denom_f1) if denom_f1 else None,
        }
    return result


def train_epoch(model, loader, optimizer, class_weights) -> float:
    import torch
    import torch.nn.functional as F

    model.train()
    total = 0.0
    batches = 0
    for image, mask, *_ in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        ce = F.cross_entropy(logits, mask, weight=class_weights)
        loss = ce + 0.75 * dice_loss(logits, mask)
        loss.backward()
        optimizer.step()
        total += float(loss.detach().cpu().item())
        batches += 1
    return total / max(1, batches)


def evaluate(model, loader) -> tuple[float, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray] | None]:
    import torch
    import torch.nn.functional as F

    model.eval()
    matrix = np.zeros((len(DAMAGE_CLASSES), len(DAMAGE_CLASSES)), dtype=np.int64)
    total_loss = 0.0
    batches = 0
    visual = None
    with torch.inference_mode():
        for image, mask, *_ in loader:
            logits = model(image)
            loss = F.cross_entropy(logits, mask)
            prediction = logits.argmax(dim=1)
            total_loss += float(loss.cpu().item())
            batches += 1
            pred_np = prediction.cpu().numpy()
            mask_np = mask.cpu().numpy()
            for p, t in zip(pred_np, mask_np):
                confusion_update(matrix, p, t)
            if visual is None:
                visual = (
                    image[0].cpu().numpy(),
                    mask_np[0],
                    pred_np[0],
                )
    return total_loss / max(1, batches), matrix, visual


def colourize_mask(mask: np.ndarray) -> np.ndarray:
    palette = np.array([
        [0, 0, 0], [255, 120, 0], [255, 0, 160], [0, 200, 255],
        [80, 80, 255], [0, 255, 100], [255, 255, 0], [180, 0, 255],
        [255, 0, 0], [80, 80, 80], [0, 160, 160], [255, 180, 180],
    ], dtype=np.uint8)
    return palette[np.clip(mask.astype(np.int64), 0, len(palette) - 1)]


def save_visual(path: Path, visual: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    image_chw, truth, prediction = visual
    rgb = np.clip(image_chw.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    truth_bgr = cv2.cvtColor(colourize_mask(truth), cv2.COLOR_RGB2BGR)
    pred_bgr = cv2.cvtColor(colourize_mask(prediction), cv2.COLOR_RGB2BGR)
    panel = np.hstack((bgr, truth_bgr, pred_bgr))
    cv2.putText(panel, 'INPUT', (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    width = bgr.shape[1]
    cv2.putText(panel, 'GT MASK', (width + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, 'PRED MASK', (2 * width + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), panel):
        raise RuntimeError('Could not save validation visual')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--face-size', type=int, default=192)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--train-repetitions', type=int, default=6)
    parser.add_argument('--val-repetitions', type=int, default=3)
    parser.add_argument('--seed', type=int, default=240817)
    parser.add_argument('--base-channels', type=int, default=12)
    args = parser.parse_args()

    set_deterministic(args.seed)
    budget = detect_resource_budget(0.80)
    apply_resource_budget(budget)
    assert_memory_within_budget(budget, stage='damage_mask_net_start')

    import torch
    import torch.nn.functional as F

    args.output.mkdir(parents=True, exist_ok=True)
    records = load_source_manifest(args.manifest)
    if len(records) < 4:
        raise RuntimeError('Need at least four identities for a development split')
    # Frozen deterministic identity-disjoint development split for this research slice.
    split_index = max(2, len(records) - 2)
    train_records = records[:split_index]
    val_records = records[split_index:]
    if {r.source_id for r in train_records} & {r.source_id for r in val_records}:
        raise RuntimeError('Identity leakage in DamageMaskNet split')

    train_dataset = SyntheticDamageDataset(
        train_records,
        args.source_dir,
        face_size=args.face_size,
        repetitions_per_class=args.train_repetitions,
        base_seed=args.seed,
    )
    val_dataset = SyntheticDamageDataset(
        val_records,
        args.source_dir,
        face_size=args.face_size,
        repetitions_per_class=args.val_repetitions,
        base_seed=args.seed + 90000000,
    )
    train_loader = _loader(train_dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed)
    val_loader = _loader(val_dataset, batch_size=args.batch_size, shuffle=False, seed=args.seed)

    model = DamageMaskUNet(classes=len(DAMAGE_CLASSES), base_channels=args.base_channels).to('cpu')
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    class_weights = torch.tensor([0.12] + [2.0] * (len(DAMAGE_CLASSES) - 1), dtype=torch.float32)

    history = []
    training_start = time.perf_counter()
    best_state = None
    best_damage_mean_f1 = -1.0
    best_metrics = None
    best_visual = None
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = train_epoch(model, train_loader, optimizer, class_weights)
        val_loss, matrix, visual = evaluate(model, val_loader)
        per_class = metrics_from_confusion(matrix)
        damage_f1 = [
            float(per_class[name]['f1'])
            for name in DAMAGE_CLASSES[1:]
            if per_class[name]['f1'] is not None
        ]
        damage_mean_f1 = float(np.mean(damage_f1)) if damage_f1 else 0.0
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'damage_macro_f1': damage_mean_f1,
        })
        if damage_mean_f1 > best_damage_mean_f1:
            best_damage_mean_f1 = damage_mean_f1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_metrics = per_class
            best_visual = visual
        assert_memory_within_budget(budget, stage=f'damage_mask_net_epoch_{epoch}')

    training_seconds = time.perf_counter() - training_start
    if best_state is None or best_metrics is None or best_visual is None:
        raise RuntimeError('Training produced no best checkpoint')
    model.load_state_dict(best_state, strict=True)
    model.eval()

    checkpoint_path = args.output / 'damage_mask_unet_dev.pth'
    torch.save({
        'state_dict': best_state,
        'classes': list(DAMAGE_CLASSES),
        'base_channels': int(args.base_channels),
        'face_size': int(args.face_size),
        'development_only': True,
    }, checkpoint_path)

    save_visual(args.output / 'validation_example.png', best_visual)

    # Measure one CPU inference after warm-up.
    image_chw, _, _ = best_visual
    sample_tensor = torch.from_numpy(image_chw).unsqueeze(0).float()
    with torch.inference_mode():
        _ = model(sample_tensor)
        start = time.perf_counter()
        torch_logits = model(sample_tensor)
        inference_seconds = time.perf_counter() - start
    assert_memory_within_budget(budget, stage='damage_mask_net_torch_inference')

    onnx_path = args.output / 'damage_mask_unet_dev.onnx'
    torch.onnx.export(
        model,
        sample_tensor,
        str(onnx_path),
        input_names=['image'],
        output_names=['logits'],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            'image': {0: 'batch', 2: 'height', 3: 'width'},
            'logits': {0: 'batch', 2: 'height', 3: 'width'},
        },
    )

    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    ort_start = time.perf_counter()
    ort_logits = session.run(['logits'], {'image': sample_tensor.numpy()})[0]
    ort_seconds = time.perf_counter() - ort_start
    torch_np = torch_logits.detach().cpu().numpy()
    max_abs = float(np.max(np.abs(torch_np - ort_logits)))
    mean_abs = float(np.mean(np.abs(torch_np - ort_logits)))
    torch_seg = torch_np.argmax(axis=1)
    ort_seg = ort_logits.argmax(axis=1)
    segmentation_equal = bool(np.array_equal(torch_seg, ort_seg))

    damage_iou = [
        float(best_metrics[name]['iou'])
        for name in DAMAGE_CLASSES[1:]
        if best_metrics[name]['iou'] is not None
    ]
    damage_f1 = [
        float(best_metrics[name]['f1'])
        for name in DAMAGE_CLASSES[1:]
        if best_metrics[name]['f1'] is not None
    ]

    report = {
        'experiment': 'damage_mask_net_unet_development_vertical_slice_v1',
        'qualification_scope': 'development_only_not_final_holdout',
        'production_qualified': False,
        'architecture': {
            'name': 'DamageMaskUNet',
            'base_channels': int(args.base_channels),
            'parameter_count': int(params),
            'input_face_size_training': int(args.face_size),
            'classes': list(DAMAGE_CLASSES),
        },
        'data': {
            'manifest': str(args.manifest),
            'train_identities': [r.source_id for r in train_records],
            'validation_identities': [r.source_id for r in val_records],
            'identity_disjoint': True,
            'train_samples': len(train_dataset),
            'validation_samples': len(val_dataset),
            'exact_synthetic_masks': True,
            'final_holdout_used': False,
            'limitations': [
                'Only the legacy V1 development source bank is used in this vertical slice',
                'The intended 300-400 identity 95-99% female primary-domain bank is not built yet',
                'This run proves the training/export path; it does not qualify DamageMaskNet',
            ],
        },
        'training': {
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'training_seconds': float(training_seconds),
            'history': history,
            'best_damage_macro_f1': float(best_damage_mean_f1),
        },
        'validation': {
            'per_class': best_metrics,
            'damage_macro_iou': float(np.mean(damage_iou)) if damage_iou else 0.0,
            'damage_macro_f1': float(np.mean(damage_f1)) if damage_f1 else 0.0,
        },
        'runtime': {
            'torch_cpu_seconds_single_face': float(inference_seconds),
            'onnxruntime_cpu_seconds_single_face_first_call': float(ort_seconds),
            'resource_budget': resource_snapshot(budget),
        },
        'onnx_parity': {
            'opset': 17,
            'max_abs_logit_difference': max_abs,
            'mean_abs_logit_difference': mean_abs,
            'argmax_segmentation_equal': segmentation_equal,
        },
        'artifacts': {
            'checkpoint': checkpoint_path.name,
            'onnx': onnx_path.name,
            'validation_example': 'validation_example.png',
        },
        'next_gate': (
            'Architecture comparison on a much larger identity-disjoint primary-domain bank; '
            'then Windows/EliteBook ONNX inference qualification.'
        ),
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    assert segmentation_equal, 'ONNX argmax segmentation changed from PyTorch'
    assert max_abs < 1e-3, f'ONNX parity drift too high: {max_abs}'

    del session, model
    gc.collect()
    assert_memory_within_budget(budget, stage='damage_mask_net_end')
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
