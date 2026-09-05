from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2

from phase04_face_crop import build_yunet_cropper, crop_main_face


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(
    *,
    source_dir: Path,
    prefetch_manifest: Path,
    yunet_path: Path,
    output_dir: Path,
    output_manifest: Path,
    output_size: int,
    train_count: int,
    minimum_validation_count: int,
    minimum_detector_score: float,
) -> dict[str, object]:
    payload = json.loads(prefetch_manifest.read_text(encoding="utf-8"))
    resolved = payload.get("resolved")
    download_failures = payload.get("failures")
    if not isinstance(resolved, list) or not resolved:
        raise RuntimeError("public portrait prefetch manifest has no resolved images")
    if download_failures:
        raise RuntimeError(f"public portrait prefetch contains failures: {download_failures}")
    if int(train_count) < 1 or int(minimum_validation_count) < 1:
        raise ValueError("train_count and minimum_validation_count must be positive")

    output_dir.mkdir(parents=True, exist_ok=True)
    engine = build_yunet_cropper(yunet_path, score_threshold=float(minimum_detector_score))
    detected_rows: list[dict[str, object]] = []
    detector_failures: list[dict[str, str]] = []

    for original_index, row in enumerate(resolved):
        if not isinstance(row, dict):
            raise ValueError("invalid resolved portrait row")
        key = str(row.get("key", "")).strip()
        if not key:
            raise ValueError("resolved portrait has no key")
        candidate = Path(str(row.get("path", ""))) if row.get("path") else source_dir / f"{key}.jpg"
        if not candidate.is_file():
            candidate = source_dir / f"{key}.jpg"
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"cannot decode portrait: {candidate}")
        try:
            result = crop_main_face(
                image,
                engine,
                output_size=int(output_size),
                context_scale=1.35,
                minimum_detector_score=float(minimum_detector_score),
            )
        except (RuntimeError, ValueError) as exc:
            detector_failures.append({"key": key, "error": str(exc)})
            continue

        detected_index = len(detected_rows)
        filename = f"{detected_index + 1:02d}_{key}_yunet_face.jpg"
        target = output_dir / filename
        if not cv2.imwrite(str(target), result.image, [int(cv2.IMWRITE_JPEG_QUALITY), 98]):
            raise RuntimeError(f"cannot write face crop: {target}")
        detected_rows.append(
            {
                "source_id": f"public_yunet_{key}",
                "filename": filename,
                "clean_source_sha256": _sha256(target),
                "face_bbox_normalized": [0.0, 0.0, 1.0, 1.0],
                "identity_key": f"public-development:{key}",
                "license": "US_GOVERNMENT_PUBLIC_DOMAIN",
                "original_index": int(original_index),
                "original_filename": candidate.name,
                "original_sha256": _sha256(candidate),
                "detector": {
                    "backend": result.detector_backend,
                    "score": result.detector_score,
                    "source_bbox": list(result.source_bbox),
                    "crop_bbox": list(result.crop_bbox),
                    "minimum_score": float(minimum_detector_score),
                },
            }
        )

    required_total = int(train_count) + int(minimum_validation_count)
    if len(detected_rows) < required_total:
        raise RuntimeError(
            f"YuNet accepted only {len(detected_rows)}/{len(resolved)} portraits at score >= "
            f"{minimum_detector_score}; need at least {required_total}; failures={detector_failures}"
        )

    rows: list[dict[str, object]] = []
    for accepted_index, row in enumerate(detected_rows):
        copied = dict(row)
        copied["dataset_split"] = "train" if accepted_index < int(train_count) else "validation"
        rows.append(copied)

    train_ids = {str(row["identity_key"]) for row in rows if row["dataset_split"] == "train"}
    validation_ids = {str(row["identity_key"]) for row in rows if row["dataset_split"] == "validation"}
    if train_ids & validation_ids:
        raise RuntimeError("identity leakage in YuNet portrait bank")
    manifest: dict[str, object] = {
        "version": 2,
        "purpose": "Phase04 YuNet-cropped public real portraits for DEVELOPMENT only",
        "detector": {
            "backend": "OpenCV Zoo YuNet",
            "asset": str(yunet_path),
            "asset_sha256": _sha256(yunet_path),
            "minimum_score": float(minimum_detector_score),
            "context_scale": 1.35,
            "rejected_portraits_are_excluded_without_threshold_relaxation": True,
        },
        "output_size": int(output_size),
        "identity_disjoint": True,
        "final_holdout_used": False,
        "v3_used": False,
        "v4_used": False,
        "counts": {
            "requested": len(resolved),
            "accepted": len(rows),
            "rejected": len(detector_failures),
            "train": sum(row["dataset_split"] == "train" for row in rows),
            "validation": sum(row["dataset_split"] == "validation" for row in rows),
            "total": len(rows),
        },
        "detector_failures": detector_failures,
        "sources": rows,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--prefetch-manifest", required=True, type=Path)
    parser.add_argument("--yunet", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument("--train-count", type=int, default=6)
    parser.add_argument("--minimum-validation-count", type=int, default=2)
    parser.add_argument("--minimum-detector-score", type=float, default=0.75)
    args = parser.parse_args()
    report = prepare(
        source_dir=args.source_dir,
        prefetch_manifest=args.prefetch_manifest,
        yunet_path=args.yunet,
        output_dir=args.output_dir,
        output_manifest=args.output_manifest,
        output_size=args.output_size,
        train_count=args.train_count,
        minimum_validation_count=args.minimum_validation_count,
        minimum_detector_score=args.minimum_detector_score,
    )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "detector": report["detector"],
                "detector_failures": report["detector_failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
