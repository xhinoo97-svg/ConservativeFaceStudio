from __future__ import annotations

"""Run the existing practical benchmark with degradations anchored to the detected face.

The original benchmark placed several masks in whole-image normalized coordinates. On
portrait photographs with different framing this could put a nominal face occlusion on
neck, clothing, or background. This wrapper keeps the existing report/CLI contract but
replaces only scenario construction, so the quality gate measures face restoration.
"""

import shutil

import cv2
import numpy as np

import app.practical_benchmark as pb


_ORIGINAL_EVALUATE_SCENARIO = pb.evaluate_scenario


def _largest_face_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image.shape[:2]
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(max(24, w // 12), max(24, h // 12)))
        if len(faces):
            x, y, fw, fh = max(faces, key=lambda item: int(item[2]) * int(item[3]))
            pad_x = int(round(fw * 0.12))
            pad_y_top = int(round(fh * 0.18))
            pad_y_bottom = int(round(fh * 0.12))
            x0 = max(0, int(x) - pad_x)
            y0 = max(0, int(y) - pad_y_top)
            x1 = min(w, int(x + fw) + pad_x)
            y1 = min(h, int(y + fh) + pad_y_bottom)
            if x1 > x0 + 8 and y1 > y0 + 8:
                return x0, y0, x1, y1
    except Exception:
        pass

    # Deterministic fallback for synthetic/unit-test portraits or detector misses.
    return int(0.24 * w), int(0.16 * h), int(0.76 * w), int(0.82 * h)


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


def make_face_anchored_scenarios(clean: np.ndarray, *, seed: int = 20260808, profile: str = "full") -> tuple[pb.Scenario, ...]:
    rng = np.random.default_rng(seed)
    h, w = clean.shape[:2]
    bbox = _largest_face_bbox(clean)
    x0, y0, x1, y1 = bbox
    fw = max(1, x1 - x0)
    fh = max(1, y1 - y0)

    face = _ellipse_in_bbox((h, w), bbox, (0.50, 0.52), (0.49, 0.51))
    sticker = _ellipse_in_bbox((h, w), bbox, (0.50, 0.58), (0.39, 0.28))
    eye_band = _rect_in_bbox((h, w), bbox, (0.10, 0.28), (0.90, 0.49))
    nose = _ellipse_in_bbox((h, w), bbox, (0.50, 0.57), (0.19, 0.24))
    mouth = _ellipse_in_bbox((h, w), bbox, (0.50, 0.78), (0.34, 0.14))

    left = np.zeros((h, w), dtype=np.uint8)
    right = np.zeros((h, w), dtype=np.uint8)
    mid = x0 + fw // 2
    overlap = max(2, fw // 14)
    left[max(0, y0):min(h, y1), max(0, x0):min(w, mid + overlap)] = 255
    right[max(0, y0):min(h, y1), max(0, mid - overlap):min(w, x1)] = 255

    opaque = clean.copy()
    opaque[sticker > 0] = (18, 18, 18)

    component_support = cv2.bitwise_or(cv2.bitwise_or(eye_band, nose), mouth)
    component_damage = cv2.bitwise_and(sticker, component_support)
    component_opaque = clean.copy()
    component_opaque[component_damage > 0] = (18, 18, 18)

    scribble = clean.copy()
    line_x0 = int(round(x0 + 0.18 * fw))
    line_x1 = int(round(x0 + 0.82 * fw))
    base_y0 = int(round(y0 + 0.64 * fh))
    base_y1 = int(round(y0 + 0.49 * fh))
    thickness = max(3, int(round(fw / 28)))
    for offset in (-18, -6, 6, 18):
        scaled = int(round(offset * max(0.45, fh / 220.0)))
        cv2.line(scribble, (line_x0, base_y0 + scaled), (line_x1, base_y1 + scaled), (10, 10, 10), thickness)
    scribble_mask = np.any(scribble != clean, axis=2).astype(np.uint8) * 255
    scribble_mask = cv2.bitwise_and(scribble_mask, face)
    scribble = _masked_variant(clean, scribble, face)

    translucent = clean.copy()
    overlay = np.empty_like(clean)
    overlay[:] = (210, 60, 180)
    alpha_mask = _ellipse_in_bbox((h, w), bbox, (0.50, 0.56), (0.43, 0.34))
    blended = cv2.addWeighted(clean, 0.58, overlay, 0.42, 0)
    translucent[alpha_mask > 0] = blended[alpha_mask > 0]

    mild = _masked_variant(clean, cv2.GaussianBlur(clean, (7, 7), 1.4), face)
    heavy = _masked_variant(clean, cv2.GaussianBlur(clean, (17, 17), 4.2), face)
    motion = _masked_variant(clean, pb._motion_blur(clean, 15), face)
    noisy = _masked_variant(clean, pb._jpeg_noise(clean, rng), face)
    mosaic = _masked_variant(clean, pb._mosaic(clean, 14), face)

    all_cases = (
        pb.Scenario("gaussian_mild_single", mild, (), face, True),
        pb.Scenario("gaussian_heavy_single", heavy, (), face, True),
        pb.Scenario("motion_blur_single", motion, (), face, True),
        pb.Scenario("noise_jpeg_single", noisy, (), face, True),
        pb.Scenario("mosaic_single", mosaic, (), face, True),
        pb.Scenario("translucent_single", translucent, (), alpha_mask, True),
        pb.Scenario("opaque_sticker_single", opaque, (), sticker, False, True),
        pb.Scenario("opaque_sticker_full_reference", opaque, (clean.copy(),), sticker, True),
        pb.Scenario("scribble_two_partial", scribble, (pb._partial_reference(clean, left), pb._partial_reference(clean, right)), scribble_mask, True),
        pb.Scenario(
            "component_only_references",
            component_opaque,
            (pb._partial_reference(clean, eye_band), pb._partial_reference(clean, nose), pb._partial_reference(clean, mouth)),
            component_damage,
            True,
        ),
    )
    if profile == "quick":
        chosen = {"gaussian_heavy_single", "opaque_sticker_single", "opaque_sticker_full_reference", "scribble_two_partial", "component_only_references"}
        return tuple(item for item in all_cases if item.name in chosen)
    return all_cases


def _export_canonical_visual_evidence(clean: np.ndarray, scenario: pb.Scenario, output_dir) -> None:
    """Persist the exact visual inputs needed to audit one benchmark case.

    This is intentionally independent from the runner export: even when the pipeline
    aborts, CI retains the clean target, damaged primary, every supplied donor and the
    benchmark damage mask. At most nine references are emitted by the product contract.
    """
    case_dir = output_dir / scenario.name
    case_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(case_dir / "00_ground_truth_original.png"), clean)
    cv2.imwrite(str(case_dir / "01_primary_degraded.png"), scenario.primary)
    for index, reference in enumerate(scenario.references[:9], start=1):
        cv2.imwrite(str(case_dir / f"02_reference_{index:02d}.png"), reference)
    cv2.imwrite(str(case_dir / "05_damage_mask.png"), scenario.damage_mask)


def _evaluate_scenario_with_visual_evidence(clean: np.ndarray, scenario: pb.Scenario, output_dir, *, core_paths=None):
    _export_canonical_visual_evidence(clean, scenario, output_dir)
    record = _ORIGINAL_EVALUATE_SCENARIO(clean, scenario, output_dir, core_paths=core_paths)
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
    return record


def main() -> int:
    pb.make_scenarios = make_face_anchored_scenarios
    pb.evaluate_scenario = _evaluate_scenario_with_visual_evidence
    return int(pb.main())


if __name__ == "__main__":
    raise SystemExit(main())
