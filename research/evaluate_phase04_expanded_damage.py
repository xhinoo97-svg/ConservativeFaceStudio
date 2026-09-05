from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
if str(RESEARCH) not in sys.path:
    sys.path.insert(0, str(RESEARCH))

from phase04_damage_evaluation import build_matrix, phase04_gate  # noqa: E402
from phase04_expanded_damage_generator import apply_expanded_damage  # noqa: E402


def _load_taxonomy():
    path = ROOT / "app" / "damage_taxonomy.py"
    spec = importlib.util.spec_from_file_location("_phase04_taxonomy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load taxonomy: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TAXONOMY = _load_taxonomy()
HEALTHY_INDEX = int(TAXONOMY.HEALTHY_INDEX)
DAMAGE_CLASSES = tuple(TAXONOMY.DAMAGE_CLASSES)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cascade_classifier(path: Path):
    """Return a Haar cascade classifier across OpenCV 4.x and 5.x namespaces."""
    legacy = getattr(cv2, "CascadeClassifier", None)
    if callable(legacy):
        return legacy(str(path)), "opencv_legacy_cascade"
    objdetect = getattr(cv2, "objdetect", None)
    modern = getattr(objdetect, "CascadeClassifier", None) if objdetect is not None else None
    if callable(modern):
        return modern(str(path)), "opencv_objdetect_cascade"
    return None, "cascade_unavailable"


def _square_face_crop(image: np.ndarray, size: int) -> tuple[np.ndarray, str]:
    if image is None or image.size == 0:
        raise ValueError("invalid portrait")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade_root = getattr(getattr(cv2, "data", None), "haarcascades", "")
    cascade_path = Path(cascade_root) / "haarcascade_frontalface_default.xml"
    detector, detector_api = _cascade_classifier(cascade_path)
    boxes = ()
    if detector is not None and cascade_path.is_file():
        try:
            if not detector.empty():
                boxes = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        except (AttributeError, cv2.error):
            boxes = ()
    h, w = image.shape[:2]
    if len(boxes):
        x, y, bw, bh = max(boxes, key=lambda box: int(box[2]) * int(box[3]))
        cx = float(x + bw / 2.0)
        cy = float(y + bh / 2.0)
        side = float(max(bw, bh)) * 1.85
        x1 = max(0, int(round(cx - side / 2.0)))
        y1 = max(0, int(round(cy - side / 2.0)))
        x2 = min(w, int(round(cx + side / 2.0)))
        y2 = min(h, int(round(cy + side / 2.0)))
        crop = image[y1:y2, x1:x2]
        method = f"{detector_api}_largest_face_1p85"
    else:
        side = min(h, w)
        x1 = max(0, (w - side) // 2)
        y1 = max(0, (h - side) // 2)
        crop = image[y1 : y1 + side, x1 : x1 + side]
        method = f"center_square_fallback_after_{detector_api}"
    if crop.size == 0:
        raise RuntimeError("empty face crop")
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA), method


def _preprocess(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    maximum = np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits - maximum, dtype=np.float32)
    denominator = np.sum(exp, axis=1, keepdims=True, dtype=np.float32)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise RuntimeError("invalid softmax denominator")
    return exp / denominator


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
    model: Path,
    portraits_dir: Path,
    output: Path,
    threshold: float,
    image_size: int,
    seed: int,
) -> dict[str, Any]:
    import onnxruntime as ort

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0,1)")
    expected_portraits = (
        "eileen_collins",
        "mae_jemison",
        "sally_ride",
        "buzz_aldrin",
        "neil_armstrong",
        "katherine_johnson",
        "peggy_whitson",
        "victor_glover",
    )
    paths = [portraits_dir / f"{key}.jpg" for key in expected_portraits]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing public development portraits: {missing}")

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError("expected exactly one ONNX input")
    input_name = inputs[0].name
    output_names = [item.name for item in session.get_outputs()]
    if not output_names:
        raise RuntimeError("ONNX model exposes no outputs")
    logits_name = "logits" if "logits" in output_names else output_names[0]

    cases = build_matrix()
    overall_counts = defaultdict(int)
    type_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    crop_methods: dict[str, str] = {}
    portrait_hashes: dict[str, str] = {}
    case_rows: list[dict[str, Any]] = []
    total_inference_seconds = 0.0

    for portrait_index, (key, path) in enumerate(zip(expected_portraits, paths)):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        clean, method = _square_face_crop(image, image_size)
        crop_methods[key] = method
        portrait_hashes[key] = _sha256(path)
        for case_index, case in enumerate(cases):
            sample = apply_expanded_damage(
                clean,
                case,
                seed=int(seed + portrait_index * 100_000 + case_index * 101),
            )
            tensor = _preprocess(sample.image)
            start = time.perf_counter()
            logits = session.run([logits_name], {input_name: tensor})[0]
            total_inference_seconds += time.perf_counter() - start
            expected_shape = (1, len(DAMAGE_CLASSES), image_size, image_size)
            if tuple(logits.shape) != expected_shape:
                raise RuntimeError(f"unexpected logits shape {tuple(logits.shape)} != {expected_shape}")
            probabilities = _softmax(np.asarray(logits, dtype=np.float32))[0]
            class_map = np.argmax(probabilities, axis=0)
            confidence = np.max(probabilities, axis=0)
            predicted = (class_map != HEALTHY_INDEX) & (confidence >= float(threshold))
            truth = sample.binary_mask > 0
            counts = _confusion(predicted, truth)
            _add_counts(overall_counts, counts)
            _add_counts(type_counts[case.damage_type], counts)
            for group in _group_for_type(case.damage_type):
                _add_counts(group_counts[group], counts)
            case_metrics = _metrics({key_: value for key_, value in zip(("tp", "fp", "fn", "tn"), counts)})
            case_rows.append(
                {
                    "portrait": key,
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
    critical_candidates = [metrics for damage_type, metrics in per_type.items() if damage_type != "HEALTHY"]
    critical_min = min(critical_candidates, key=lambda item: float(item["f1"]))
    critical_min_type = min(
        (key for key in per_type if key != "HEALTHY"),
        key=lambda key: float(per_type[key]["f1"]),
    )
    groups["CRITICAL_MIN"] = dict(critical_min)
    groups["CRITICAL_MIN"]["damage_type"] = critical_min_type
    gate_input = {"binary": overall, "groups": groups}
    gate = phase04_gate(gate_input)

    report: dict[str, Any] = {
        "experiment": "phase04_expanded_public_portrait_development_v1",
        "qualification_scope": "development_measurement_not_final_holdout",
        "production_qualified": False,
        "phase04_gate_passed_on_this_development_set": bool(gate["passed"]),
        "disposition": "DEVELOPMENT_GATE_PASS" if gate["passed"] else "DEVELOPMENT_GATE_FAIL",
        "model": {
            "architecture": "LRASPP_MobileNetV3_Large_ONNX",
            "path": str(model),
            "sha256": _sha256(model),
            "healthy_index": HEALTHY_INDEX,
            "classes": list(DAMAGE_CLASSES),
            "runtime_damage_confidence_threshold": float(threshold),
        },
        "data": {
            "source": "eight public-domain NASA/US Government development portraits already declared by CFS practical benchmark",
            "identity_count": len(paths),
            "portrait_keys": list(expected_portraits),
            "portrait_sha256": portrait_hashes,
            "crop_methods": crop_methods,
            "matrix_case_count_per_identity": len(cases),
            "completed_cases": len(case_rows),
            "expected_cases": len(paths) * len(cases),
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
            "provider": session.get_providers(),
            "total_inference_seconds": float(total_inference_seconds),
            "seconds_per_case": float(total_inference_seconds / max(1, len(case_rows))),
            "network_required_after_model_and_portrait_acquisition": False,
        },
        "safety": {
            "model_output_type": "mask_logits_only",
            "can_modify_image_pixels": False,
            "wrong_person_final_pixels": 0,
            "provenance_violations": 0,
            "restoration_pass_count": 0,
        },
        "production_blockers": [
            "development set has only 8 public portraits and is not the final 300-400 identity benchmark",
            "checkpoint redistribution license remains not explicit",
            "physical HP EliteBook execution not measured",
        ],
        "cases": case_rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--portraits-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=240905)
    args = parser.parse_args()
    report = evaluate(
        model=args.model,
        portraits_dir=args.portraits_dir,
        output=args.output,
        threshold=args.threshold,
        image_size=args.image_size,
        seed=args.seed,
    )
    print(json.dumps({
        "disposition": report["disposition"],
        "binary": report["metrics"]["binary"],
        "groups": report["metrics"]["groups"],
        "gate": report["frozen_phase04_gate"],
        "completed_cases": report["data"]["completed_cases"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
