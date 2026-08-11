from __future__ import annotations

"""Face-focused practical benchmark for ConservativeFaceStudio.

All synthetic damage is anchored to the facial region rather than whole-image
coordinates. The release matrix exercises one primary image plus zero through nine
references. Target-95 applicability is fixed from the scenario evidence *before* the
restoration output is scored; it is never changed afterwards to excuse a poor result.
"""

from pathlib import Path
import shutil
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import app  # noqa: F401  # Apply the packaged OpenCV boot policy before importing cv2.
import cv2
import numpy as np

import app.practical_benchmark as pb


_ORIGINAL_EVALUATE_SCENARIO = pb.evaluate_scenario


def _largest_face_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    """Return a deterministic face-focused ROI without depending on legacy Haar APIs."""
    h, w = image.shape[:2]
    return int(0.22 * w), int(0.12 * h), int(0.78 * w), int(0.84 * h)


def _ellipse_in_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int], center: tuple[float, float], axes: tuple[float, float]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    cx = int(round(x0 + center[0] * fw))
    cy = int(round(y0 + center[1] * fh))
    ax = max(1, int(round(axes[0] * fw)))
    ay = max(1, int(round(axes[1] * fh)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)
    return mask


def _rect_in_bbox(shape: tuple[int, int], bbox: tuple[int, int, int, int], p0: tuple[float, float], p1: tuple[float, float]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    a = (int(round(x0 + p0[0] * fw)), int(round(y0 + p0[1] * fh)))
    b = (int(round(x0 + p1[0] * fw)), int(round(y0 + p1[1] * fh)))
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(mask, a, b, 255, -1)
    return mask


def _masked_variant(clean: np.ndarray, variant: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = clean.copy()
    result[mask > 0] = variant[mask > 0]
    return result


def _split_support(mask: np.ndarray, count: int) -> tuple[np.ndarray, ...]:
    """Split one facial support mask into complementary overlapping donor regions."""
    count = max(1, min(9, int(count)))
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return tuple(np.zeros_like(mask) for _ in range(count))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    width = max(1, x1 - x0)
    overlap = max(1, width // max(30, count * 8))
    result: list[np.ndarray] = []
    for index in range(count):
        a = x0 + int(round(index * width / count))
        b = x0 + int(round((index + 1) * width / count))
        a = max(x0, a - overlap)
        b = min(x1, b + overlap)
        part = np.zeros_like(mask)
        part[:, a:b] = mask[:, a:b]
        result.append(part)
    return tuple(result)


def _partial_reference_set(clean: np.ndarray, support: np.ndarray, count: int) -> tuple[np.ndarray, ...]:
    return tuple(pb._partial_reference(clean, part) for part in _split_support(support, count))


def make_face_anchored_scenarios(clean: np.ndarray, *, seed: int = 20260808, profile: str = "full") -> tuple[pb.Scenario, ...]:
    rng = np.random.default_rng(seed)
    h, w = clean.shape[:2]
    bbox = _largest_face_bbox(clean)

    face = _ellipse_in_bbox((h, w), bbox, (0.50, 0.52), (0.48, 0.50))
    central = _ellipse_in_bbox((h, w), bbox, (0.50, 0.56), (0.36, 0.30))
    severe = _ellipse_in_bbox((h, w), bbox, (0.50, 0.54), (0.46, 0.42))
    eye_band = _rect_in_bbox((h, w), bbox, (0.08, 0.27), (0.92, 0.49))
    nose = _ellipse_in_bbox((h, w), bbox, (0.50, 0.57), (0.20, 0.24))
    mouth = _ellipse_in_bbox((h, w), bbox, (0.50, 0.78), (0.34, 0.14))

    opaque = clean.copy()
    opaque[central > 0] = (18, 18, 18)
    opaque_severe = clean.copy()
    opaque_severe[severe > 0] = (12, 12, 12)

    component_support = cv2.bitwise_or(cv2.bitwise_or(eye_band, nose), mouth)
    component_damage = cv2.bitwise_and(central, component_support)
    component_opaque = clean.copy()
    component_opaque[component_damage > 0] = (18, 18, 18)

    scribble = clean.copy()
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)
    line_x0 = int(round(x0 + 0.15 * fw))
    line_x1 = int(round(x0 + 0.85 * fw))
    base_y0 = int(round(y0 + 0.67 * fh))
    base_y1 = int(round(y0 + 0.43 * fh))
    thickness = max(3, int(round(fw / 25)))
    for offset in (-22, -11, 0, 11, 22):
        scaled = int(round(offset * max(0.45, fh / 220.0)))
        cv2.line(scribble, (line_x0, base_y0 + scaled), (line_x1, base_y1 + scaled), (8, 8, 8), thickness)
    scribble_mask = np.any(scribble != clean, axis=2).astype(np.uint8) * 255
    scribble_mask = cv2.bitwise_and(scribble_mask, face)
    scribble = _masked_variant(clean, scribble, face)

    translucent = clean.copy()
    overlay = np.empty_like(clean)
    overlay[:] = (210, 60, 180)
    alpha_mask = _ellipse_in_bbox((h, w), bbox, (0.50, 0.55), (0.42, 0.34))
    blended = cv2.addWeighted(clean, 0.55, overlay, 0.45, 0)
    translucent[alpha_mask > 0] = blended[alpha_mask > 0]

    mild = _masked_variant(clean, cv2.GaussianBlur(clean, (5, 5), 0.9), face)
    medium = _masked_variant(clean, cv2.GaussianBlur(clean, (11, 11), 2.4), face)
    heavy = _masked_variant(clean, cv2.GaussianBlur(clean, (19, 19), 5.0), face)
    motion = _masked_variant(clean, pb._motion_blur(clean, 17), face)
    noisy = _masked_variant(clean, pb._jpeg_noise(clean, rng), face)
    mosaic = _masked_variant(clean, pb._mosaic(clean, 14), face)

    ref1 = _partial_reference_set(clean, central, 1)
    ref2 = _partial_reference_set(clean, scribble_mask, 2)
    ref3 = _partial_reference_set(clean, component_damage, 3)
    ref4 = _partial_reference_set(clean, severe, 4)
    ref5 = _partial_reference_set(clean, severe, 5)
    ref6 = _partial_reference_set(clean, severe, 6)
    ref7 = _partial_reference_set(clean, severe, 7)
    ref8 = _partial_reference_set(clean, severe, 8)
    ref9 = _partial_reference_set(clean, severe, 9)

    all_cases = (
        # MAIN-only degradations remain important quality tests, but destructive blur/
        # mosaic are not pre-declared >=95% evidence-recoverable without references.
        pb.Scenario("face_blur_mild_single", mild, (), face, True),
        pb.Scenario("face_blur_medium_single", medium, (), face, True),
        pb.Scenario("face_blur_heavy_single", heavy, (), face, True),
        pb.Scenario("face_motion_blur_single", motion, (), face, True),
        pb.Scenario("face_noise_jpeg_single", noisy, (), face, True),
        pb.Scenario("face_mosaic_single", mosaic, (), face, True),
        pb.Scenario("face_translucent_single", translucent, (), alpha_mask, True),
        pb.Scenario("face_opaque_single_no_evidence", opaque, (), central, False, True),
        # Reference-complete cases. Each donor set is complementary and together
        # covers the known synthetic damage, so target95 is objectively applicable.
        pb.Scenario("face_opaque_ref1", opaque, ref1, central, True),
        pb.Scenario("face_scribble_ref2", scribble, ref2, scribble_mask, True),
        pb.Scenario("face_component_ref3", component_opaque, ref3, component_damage, True),
        pb.Scenario("face_severe_ref4", opaque_severe, ref4, severe, True),
        pb.Scenario("face_severe_ref5", opaque_severe, ref5, severe, True),
        pb.Scenario("face_severe_ref6", opaque_severe, ref6, severe, True),
        pb.Scenario("face_severe_ref7", opaque_severe, ref7, severe, True),
        pb.Scenario("face_severe_ref8", opaque_severe, ref8, severe, True),
        pb.Scenario("face_severe_ref9", opaque_severe, ref9, severe, True),
    )
    if profile == "quick":
        # Release quick profile intentionally covers every allowed reference count 0..9.
        chosen = {
            "face_blur_medium_single",          # MAIN + 0
            "face_blur_heavy_single",           # MAIN + 0 severe blur diagnostic
            "face_opaque_single_no_evidence",   # MAIN + 0 non-recoverable control
            "face_opaque_ref1",                 # MAIN + 1
            "face_scribble_ref2",               # MAIN + 2
            "face_component_ref3",              # MAIN + 3
            "face_severe_ref4",                 # MAIN + 4
            "face_severe_ref5",                 # MAIN + 5
            "face_severe_ref6",                 # MAIN + 6
            "face_severe_ref7",                 # MAIN + 7
            "face_severe_ref8",                 # MAIN + 8
            "face_severe_ref9",                 # MAIN + 9
        }
        return tuple(item for item in all_cases if item.name in chosen)
    return all_cases


def _export_canonical_visual_evidence(clean: np.ndarray, scenario: pb.Scenario, output_dir) -> None:
    case_dir = output_dir / scenario.name
    case_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(case_dir / "00_ground_truth_original.png"), clean)
    cv2.imwrite(str(case_dir / "01_primary_degraded.png"), scenario.primary)
    for index, reference in enumerate(scenario.references[:9], start=1):
        cv2.imwrite(str(case_dir / f"02_reference_{index:02d}.png"), reference)
    cv2.imwrite(str(case_dir / "05_damage_mask.png"), scenario.damage_mask)


def _reference_union_coverage(scenario: pb.Scenario) -> float:
    active = scenario.damage_mask > 0
    denominator = int(np.count_nonzero(active))
    if denominator == 0 or not scenario.references:
        return 0.0
    support = np.zeros_like(active)
    for reference in scenario.references:
        # Synthetic partial references use exact black outside known donor support.
        support |= np.any(reference != 0, axis=2)
    return float(np.count_nonzero(active & support) / denominator)


def _target95_policy_before_score(scenario: pb.Scenario) -> tuple[bool, str, float]:
    """Fix target95 applicability from supplied evidence before restoration runs."""
    if not scenario.recoverable:
        return False, "scenario_declared_nonrecoverable", _reference_union_coverage(scenario)
    if scenario.opaque_without_evidence:
        return False, "opaque_damage_without_external_evidence", 0.0
    union = _reference_union_coverage(scenario)
    if not scenario.references:
        return False, "single_image_degradation_not_predeclared_95_evidence_recoverable", 0.0
    if union < 0.95:
        return False, "reference_union_below_95_percent_damage_coverage", union
    return True, "reference_union_covers_at_least_95_percent_of_known_damage", union


def _residual_obstruction_fraction(clean: np.ndarray, final: np.ndarray, damage_mask: np.ndarray) -> float:
    active = damage_mask > 0
    denominator = int(np.count_nonzero(active))
    if denominator == 0:
        return 0.0
    per_pixel_error = np.mean(np.abs(final.astype(np.float32) - clean.astype(np.float32)), axis=2)
    residual = active & (per_pixel_error > 24.0)
    return float(np.count_nonzero(residual) / denominator)


def _evaluate_scenario_with_visual_evidence(clean: np.ndarray, scenario: pb.Scenario, output_dir, *, core_paths=None):
    _export_canonical_visual_evidence(clean, scenario, output_dir)

    # IMPORTANT: this decision is made before the model output or score exists.
    target95_applicable, target95_reason, union_before = _target95_policy_before_score(scenario)
    record = _ORIGINAL_EVALUATE_SCENARIO(clean, scenario, output_dir, core_paths=core_paths)
    score = record.get("conservative_recovery_score")
    record["target95_applicable"] = bool(target95_applicable)
    record["target95_applicability_decided_before_score"] = True
    record["target95_policy_reason"] = target95_reason
    record["target95_reference_union_coverage_pre_score"] = float(union_before)
    record["target95_passed"] = bool(float(score) >= 95.0) if target95_applicable and score is not None else None

    case_dir = output_dir / scenario.name
    aliases = (
        ("final.png", "03_final_output.png"),
        ("diff-heatmap.png", "04_diff_heatmap.png"),
        ("final.source-map.png", "06_provenance_map.png"),
        ("final.reference-confidence.png", "07_confidence_map.png"),
    )
    for source_name, alias_name in aliases:
        source = case_dir / source_name
        if source.is_file():
            shutil.copy2(source, case_dir / alias_name)

    final = cv2.imread(str(case_dir / "final.png"), cv2.IMREAD_COLOR)
    if final is not None and final.shape == clean.shape:
        residual = _residual_obstruction_fraction(clean, final, scenario.damage_mask)
        reference_complete = bool(scenario.references and union_before >= 0.995)
        clean_face_pass = bool(residual <= 0.01) if reference_complete and scenario.recoverable else None
        record["reference_union_coverage"] = float(union_before)
        record["residual_obstruction_fraction"] = residual
        record["clean_face_pass"] = clean_face_pass
        # Tighten, never relax, the pre-declared >=95 gate: complete real evidence must
        # also remove the visible synthetic obstruction from the final facial region.
        if target95_applicable and clean_face_pass is False:
            record["target95_passed"] = False
    return record


def main() -> int:
    pb.make_scenarios = make_face_anchored_scenarios
    pb.evaluate_scenario = _evaluate_scenario_with_visual_evidence
    return int(pb.main())


if __name__ == "__main__":
    raise SystemExit(main())
