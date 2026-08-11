from __future__ import annotations

"""Build or verify the immutable face-smartphone v1 benchmark manifests.

This is benchmark-only tooling. It does not import, patch, or route the production
restoration pipeline. Mask checksums are over canonical 384x384 uint8 bytes.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = REPOSITORY_ROOT / "benchmarks" / "face-smartphone-v1"
CANVAS_SIZE = 384
BASE_SEED = 202608110000

CATEGORY_COUNTS = {
    "opaque_graphic_face_occlusion": 25,
    "face_blur_censor": 20,
    "face_mosaic_pixelation": 20,
    "black_paint_scribble": 10,
    "natural_facial_occluder": 10,
    "mixed_smartphone": 10,
    "extreme_low_evidence": 5,
}

SPLIT_CATEGORY_COUNTS = {
    "calibration": {
        "opaque_graphic_face_occlusion": 15,
        "face_blur_censor": 12,
        "face_mosaic_pixelation": 12,
        "black_paint_scribble": 6,
        "natural_facial_occluder": 6,
        "mixed_smartphone": 6,
        "extreme_low_evidence": 3,
    },
    "holdout": {
        "opaque_graphic_face_occlusion": 10,
        "face_blur_censor": 8,
        "face_mosaic_pixelation": 8,
        "black_paint_scribble": 4,
        "natural_facial_occluder": 4,
        "mixed_smartphone": 4,
        "extreme_low_evidence": 2,
    },
}

STYLES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "opaque_graphic_face_occlusion": (
        ("large_star_sticker", "star", "central_face"),
        ("emoji_sticker", "ellipse", "central_face"),
        ("central_face_sticker", "rounded_rect", "central_face"),
        ("eyes_nose_sticker", "rounded_rect", "eyes_nose"),
        ("eyes_nose_mouth_sticker", "star", "eyes_nose_mouth"),
        ("near_full_face_sticker", "ellipse", "near_full_face"),
    ),
    "face_blur_censor": (
        ("horizontal_blur_bar", "rounded_rect", "eyes_nose"),
        ("central_rect_blur", "rect", "central_face"),
        ("near_full_face_blur", "ellipse", "near_full_face"),
        ("shifted_partial_blur", "rect", "left_face"),
        ("local_motion_blur", "ellipse", "eyes_nose_mouth"),
        ("local_defocus_blur", "ellipse", "right_face"),
    ),
    "face_mosaic_pixelation": (
        ("small_block_mosaic", "rounded_rect", "central_face"),
        ("medium_block_mosaic", "rounded_rect", "eyes_nose_mouth"),
        ("large_block_mosaic", "rect", "near_full_face"),
        ("eyes_only_mosaic", "rounded_rect", "eyes_brows"),
        ("eyes_nose_mosaic", "rounded_rect", "eyes_nose"),
        ("central_face_mosaic", "ellipse", "central_face"),
        ("near_full_face_mosaic", "ellipse", "near_full_face"),
    ),
    "black_paint_scribble": (
        ("eye_band_scribble", "scribble", "eyes_brows"),
        ("irregular_face_stroke", "scribble", "eyes_nose_mouth"),
        ("partial_face_black_paint", "scribble", "left_face"),
        ("near_full_face_black_paint", "scribble", "near_full_face"),
    ),
    "natural_facial_occluder": (
        ("decorative_eye_mask", "eye_mask", "eyes_nose"),
        ("ski_goggles", "goggles", "eyes_brows"),
        ("hair_crossing_face", "hair_strand", "right_face"),
        ("hand_object_crossing_face", "hand_band", "eyes_nose_mouth"),
        ("partial_face_crop", "crop_edge", "face_edge"),
    ),
    "mixed_smartphone": (
        ("face_sticker_plus_jpeg", "star", "central_face"),
        ("face_blur_plus_low_light", "rounded_rect", "eyes_nose_mouth"),
        ("mosaic_plus_resize", "ellipse", "central_face"),
        ("occlusion_plus_compression", "hand_band", "eyes_nose"),
        ("crop_plus_facial_occlusion", "crop_edge", "face_edge"),
    ),
    "extreme_low_evidence": (
        ("near_full_opaque_no_reference", "ellipse", "near_full_face"),
        ("near_full_black_paint_wrong_reference", "scribble", "near_full_face"),
        ("face_crop_useless_reference", "crop_edge", "near_full_face"),
    ),
}

REGION_BOXES = {
    "eyes_brows": (0.10, 0.19, 0.90, 0.43),
    "eyes_nose": (0.08, 0.20, 0.92, 0.64),
    "eyes_nose_mouth": (0.07, 0.19, 0.93, 0.82),
    "central_face": (0.12, 0.18, 0.88, 0.84),
    "near_full_face": (0.02, 0.02, 0.98, 0.98),
    "left_face": (0.00, 0.12, 0.58, 0.93),
    "right_face": (0.42, 0.12, 1.00, 0.93),
    "face_edge": (0.00, 0.05, 0.42, 0.98),
}

REGION_COMPONENTS = {
    "eyes_brows": ["left_eye", "right_eye", "left_brow", "right_brow"],
    "eyes_nose": ["left_eye", "right_eye", "left_brow", "right_brow", "nose", "left_cheek", "right_cheek"],
    "eyes_nose_mouth": ["left_eye", "right_eye", "left_brow", "right_brow", "nose", "philtrum", "mouth", "lips", "left_cheek", "right_cheek"],
    "central_face": ["left_eye", "right_eye", "nose", "philtrum", "mouth", "lips", "left_cheek", "right_cheek"],
    "near_full_face": ["left_eye", "right_eye", "left_brow", "right_brow", "nose", "philtrum", "mouth", "lips", "left_cheek", "right_cheek", "chin", "jaw", "forehead", "face_contour", "hairline"],
    "left_face": ["left_eye", "left_brow", "nose", "left_cheek", "mouth", "chin", "jaw", "face_contour", "hairline"],
    "right_face": ["right_eye", "right_brow", "nose", "right_cheek", "mouth", "chin", "jaw", "face_contour", "hairline"],
    "face_edge": ["left_cheek", "right_cheek", "jaw", "forehead", "face_contour", "hairline"],
}


def _canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_sources() -> dict[str, Any]:
    return json.loads((BENCHMARK_ROOT / "sources.json").read_text(encoding="utf-8"))


def _fitted_face_bbox(source: dict[str, Any]) -> tuple[int, int, int, int]:
    width, height = (int(value) for value in source["original_dimensions"])
    scale = min(1.0, CANVAS_SIZE / max(width, height))
    fitted_w = max(1, int(round(width * scale)))
    fitted_h = max(1, int(round(height * scale)))
    pad_x = (CANVAS_SIZE - fitted_w) // 2
    pad_y = (CANVAS_SIZE - fitted_h) // 2
    x0, y0, x1, y1 = (float(value) for value in source["face_bbox_normalized"])
    return (
        int(round(pad_x + x0 * fitted_w)),
        int(round(pad_y + y0 * fitted_h)),
        int(round(pad_x + x1 * fitted_w)),
        int(round(pad_y + y1 * fitted_h)),
    )


def _face_domain_mask(source: dict[str, Any]) -> np.ndarray:
    x0, y0, x1, y1 = _fitted_face_bbox(source)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    center = (int(round((x0 + x1) * 0.5)), int(round(y0 + height * 0.52)))
    axes = (max(1, int(round(width * 0.54))), max(1, int(round(height * 0.55))))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _box_for_region(source: dict[str, Any], region: str, rng: np.random.Generator) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = _fitted_face_bbox(source)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    rx0, ry0, rx1, ry1 = REGION_BOXES[region]
    jitter_x = float(rng.uniform(-0.025, 0.025))
    jitter_y = float(rng.uniform(-0.025, 0.025))
    scale = float(rng.uniform(0.94, 1.06))
    cx = (rx0 + rx1) * 0.5 + jitter_x
    cy = (ry0 + ry1) * 0.5 + jitter_y
    rw = (rx1 - rx0) * scale
    rh = (ry1 - ry0) * scale
    return (
        int(round(x0 + (cx - rw * 0.5) * width)),
        int(round(y0 + (cy - rh * 0.5) * height)),
        int(round(x0 + (cx + rw * 0.5) * width)),
        int(round(y0 + (cy + rh * 0.5) * height)),
    )


def _star_points(box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    outer = min(x1 - x0, y1 - y0) * 0.50
    inner = outer * 0.43
    points: list[tuple[int, int]] = []
    for index in range(10):
        angle = -np.pi / 2.0 + index * np.pi / 5.0
        radius = outer if index % 2 == 0 else inner
        points.append((int(round(cx + radius * np.cos(angle))), int(round(cy + radius * np.sin(angle)))))
    return np.asarray(points, dtype=np.int32)


def _raw_mask(source: dict[str, Any], shape: str, region: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    box = _box_for_region(source, region, rng)
    x0, y0, x1, y1 = box
    mask = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
    if shape == "rect":
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    elif shape == "rounded_rect":
        radius = max(2, int(round(min(x1 - x0, y1 - y0) * 0.16)))
        cv2.rectangle(mask, (x0 + radius, y0), (x1 - radius, y1), 255, -1)
        cv2.rectangle(mask, (x0, y0 + radius), (x1, y1 - radius), 255, -1)
        for point in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius), (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
            cv2.circle(mask, point, radius, 255, -1)
    elif shape == "ellipse":
        cv2.ellipse(mask, ((x0 + x1) // 2, (y0 + y1) // 2), (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)), 0, 0, 360, 255, -1)
    elif shape == "star":
        cv2.fillPoly(mask, [_star_points(box)], 255)
    elif shape == "scribble":
        count = 5 if region != "near_full_face" else 9
        thickness = max(3, int(round((x1 - x0) * (0.055 if count < 9 else 0.075))))
        for _ in range(count):
            points = np.asarray([
                (int(rng.integers(x0, max(x0 + 1, x1))), int(rng.integers(y0, max(y0 + 1, y1))))
                for _ in range(4)
            ], dtype=np.int32)
            cv2.polylines(mask, [points], False, 255, thickness, lineType=cv2.LINE_8)
    elif shape == "eye_mask":
        inset = max(1, int(round((x1 - x0) * 0.08)))
        points = np.asarray([(x0, (y0 + y1) // 2), (x0 + inset, y0), ((x0 + x1) // 2, y0 + inset), (x1 - inset, y0), (x1, (y0 + y1) // 2), (x1 - inset, y1), ((x0 + x1) // 2, y1 - inset), (x0 + inset, y1)], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif shape == "goggles":
        cy = (y0 + y1) // 2
        half = max(2, (x1 - x0) // 5)
        ay = max(2, (y1 - y0) // 2)
        cv2.ellipse(mask, (x0 + half, cy), (half, ay), 0, 0, 360, 255, -1)
        cv2.ellipse(mask, (x1 - half, cy), (half, ay), 0, 0, 360, 255, -1)
        cv2.rectangle(mask, (x0 + 2 * half - 2, cy - 3), (x1 - 2 * half + 2, cy + 3), 255, -1)
    elif shape == "hair_strand":
        thickness = max(4, int(round((x1 - x0) * 0.10)))
        for offset in (0.18, 0.42, 0.66):
            points = np.asarray([(int(x0 + offset * (x1 - x0)), y0), (int(x0 + (offset - 0.18) * (x1 - x0)), (y0 + y1) // 2), (int(x0 + (offset + 0.10) * (x1 - x0)), y1)], dtype=np.int32)
            cv2.polylines(mask, [points], False, 255, thickness, lineType=cv2.LINE_8)
    elif shape == "hand_band":
        points = np.asarray([(x0, int(y0 + 0.18 * (y1 - y0))), (int(x0 + 0.17 * (x1 - x0)), y0), (x1, int(y0 + 0.78 * (y1 - y0))), (int(x0 + 0.83 * (x1 - x0)), y1)], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
    elif shape == "crop_edge":
        cv2.rectangle(mask, (x0, y0), (int(x0 + 0.68 * (x1 - x0)), y1), 255, -1)
    else:
        raise ValueError(f"Unsupported mask shape: {shape}")
    return cv2.bitwise_and(mask, _face_domain_mask(source))


def _reference_ids(source_id: str, wrong_source_id: str, count: int, *, extreme: bool) -> list[str]:
    if count <= 0:
        return []
    if extreme:
        options = [f"{wrong_source_id}:wrong_person_full", f"{source_id}:useless_blank", f"{wrong_source_id}:wrong_person_blurred"]
        return options[:count]
    variants = [
        "full_observed",
        "left_half_observed",
        "right_half_observed",
        "left_eye_only",
        "right_eye_only",
        "nose_dominant",
        "mouth_chin_only",
        "blurred_low_quality",
        "duplicate_full",
    ]
    result = [f"{source_id}:{variant}" for variant in variants[:count]]
    if count >= 4:
        result[-1] = f"{wrong_source_id}:wrong_person_full"
    return result


def _case_severity(category: str, style_index: int) -> str:
    if category == "extreme_low_evidence":
        return "EXTREME"
    if category in {"opaque_graphic_face_occlusion", "black_paint_scribble"}:
        return "SEVERE" if style_index % 3 else "MEDIUM"
    if category in {"face_blur_censor", "face_mosaic_pixelation", "mixed_smartphone"}:
        return ("MEDIUM", "SEVERE", "SEVERE")[style_index % 3]
    return "SEVERE_RECOVERABLE"


def build_cases() -> dict[str, Any]:
    source_payload = _load_sources()
    sources = source_payload["sources"]
    by_id = {item["source_id"]: item for item in sources}
    calibration_sources = [item["source_id"] for item in sources[:5]]
    holdout_sources = [item["source_id"] for item in sources[5:]]
    split_source_quotas = {
        "calibration": dict.fromkeys(calibration_sources, 12),
        "holdout": {holdout_sources[0]: 14, holdout_sources[1]: 13, holdout_sources[2]: 13},
    }
    cases: list[dict[str, Any]] = []
    global_index = 0
    for split in ("calibration", "holdout"):
        source_pool = calibration_sources if split == "calibration" else holdout_sources
        used = dict.fromkeys(source_pool, 0)
        for category, count in SPLIT_CATEGORY_COUNTS[split].items():
            styles = STYLES[category]
            for category_index in range(count):
                available = [source_id for source_id in source_pool if used[source_id] < split_source_quotas[split][source_id]]
                source_id = min(available, key=lambda item: (used[item], source_pool.index(item)))
                used[source_id] += 1
                source = by_id[source_id]
                style, shape, region = styles[category_index % len(styles)]
                seed = BASE_SEED + global_index
                mask = _raw_mask(source, shape, region, seed)
                face = _face_domain_mask(source)
                mask_pixels = int(np.count_nonzero(mask))
                overlap = float(np.count_nonzero((mask > 0) & (face > 0)) / max(1, mask_pixels))
                reference_count = global_index % 10
                extreme = category == "extreme_low_evidence"
                if extreme:
                    reference_count = category_index % 3
                wrong_source_id = sources[(sources.index(source) + 1) % len(sources)]["source_id"]
                references = _reference_ids(source_id, wrong_source_id, reference_count, extreme=extreme)
                target95_applicable = bool(not extreme and any(reference.startswith(f"{source_id}:") and "useless" not in reference for reference in references))
                case_id = f"cfsfs1-{split[:3]}-{global_index + 1:03d}-{style}"
                cases.append({
                    "case_id": case_id,
                    "calibration_or_holdout": split,
                    "clean_source_checksum": source["clean_source_sha256"],
                    "damage_mask_checksum": _sha256_bytes(mask.tobytes(order="C")),
                    "damage_mask_encoding": f"uint8_{CANVAS_SIZE}x{CANVAS_SIZE}_raw",
                    "damage_seed": seed,
                    "damage_severity": _case_severity(category, category_index),
                    "damage_style": style,
                    "damage_type": category,
                    "evaluated_face_regions": REGION_COMPONENTS[region],
                    "face_overlap_ratio": round(overlap, 8),
                    "main_contract": "SOURCE0_IMMUTABLE_TARGET_CANVAS",
                    "main_source_id": source_id,
                    "mask_region": region,
                    "mask_shape": shape,
                    "primary_face_case": True,
                    "provenance_contract": "GENERATED_OR_RESTORED_NEVER_BECOMES_OBSERVED_EVIDENCE",
                    "recoverability_pre_score": "LOW_EVIDENCE_ABSTAIN" if extreme else ("REFERENCE_RECOVERABLE" if target95_applicable else "SEVERE_RECOVERABLE"),
                    "reference_ids": references,
                    "target95_applicable_pre_score": target95_applicable,
                    "target95_policy": "REPORT_ONLY",
                })
                global_index += 1
        if used != split_source_quotas[split]:
            raise RuntimeError(f"Split/source allocation drift for {split}: {used!r}")
    if global_index != 100:
        raise RuntimeError(f"Expected 100 cases, got {global_index}")
    return {
        "benchmark_id": "cfs-face-smartphone-v1",
        "cases": cases,
        "generation_algorithm": "freeze_face_smartphone_benchmark.py:v1",
        "mask_checksum_encoding": f"uint8_{CANVAS_SIZE}x{CANVAS_SIZE}_raw",
        "version": 1,
    }


def build_freeze(cases_payload: dict[str, Any]) -> dict[str, Any]:
    sources_bytes = _canonical_json(_load_sources())
    cases_bytes = _canonical_json(cases_payload)
    contract_bytes = (BENCHMARK_ROOT / "contract.json").read_bytes()
    return {
        "architecture_freeze_base_sha": "368f00c122520f471e7ef310f9daf8781b51f111",
        "benchmark_id": "cfs-face-smartphone-v1",
        "cases_manifest_sha256": _sha256_bytes(cases_bytes),
        "contract_sha256": _sha256_bytes(contract_bytes),
        "frozen_before_tuning": True,
        "production_pipeline_changed": False,
        "sources_manifest_sha256": _sha256_bytes(sources_bytes),
        "split": {"calibration": 60, "holdout": 40},
        "target95_policy": "REPORT_ONLY",
        "version": 1,
    }


def _verify_file(path: Path, expected: bytes) -> None:
    actual = path.read_bytes() if path.is_file() else b""
    if actual != expected:
        raise RuntimeError(f"Frozen benchmark drift: {path.relative_to(REPOSITORY_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify committed manifests without writing")
    args = parser.parse_args()
    cases = build_cases()
    freeze = build_freeze(cases)
    cases_bytes = _canonical_json(cases)
    freeze_bytes = _canonical_json(freeze)
    if args.verify:
        _verify_file(BENCHMARK_ROOT / "cases.json", cases_bytes)
        _verify_file(BENCHMARK_ROOT / "freeze.json", freeze_bytes)
        print("Face-smartphone benchmark freeze: PASS")
        return 0
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "cases.json").write_bytes(cases_bytes)
    (BENCHMARK_ROOT / "freeze.json").write_bytes(freeze_bytes)
    print("Face-smartphone benchmark freeze generated: 100 cases (60 calibration / 40 holdout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
