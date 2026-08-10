from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.frontalization import pose_frontalness
from app.opencv_nafnet import NafNetDeblurEngine
from app.opencv_semantic_models import HeadPoseEngine
from app.opencv_zoo_face import OpenCVZooFaceEngine
from app.pretrained_values import FACE_MODEL_DEFAULTS, RESTORATION_SAFETY_DEFAULTS
from app.restoration import detect_occlusion_candidates, detail_reliability_map


@dataclass(frozen=True)
class PreflightCandidate:
    source_index: int
    identity_component: int
    detector_score: float
    pose: tuple[float, float, float] | None
    frontalness: float
    quality: float
    accepted_identity: bool


@dataclass(frozen=True)
class PreflightResult:
    selected_source_index: int
    candidates: tuple[PreflightCandidate, ...]
    deblurred_count: int
    identity_cluster_size: int
    reason: str


def _face_crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = (int(v) for v in bbox)
    height, width = image.shape[:2]
    mx = int(round(w * 0.12))
    my = int(round(h * 0.12))
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(width, x + w + mx), min(height, y + h + my)
    return image[y1:y2, x1:x2]


def _quality_score(image: np.ndarray, bbox: tuple[int, int, int, int], detector_score: float) -> float:
    """Score only genuinely useful face pixels, not sharp sticker/marker edges."""
    crop = _face_crop(image, bbox)
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    proposal = detect_occlusion_candidates(crop)
    visible = (proposal == 0).astype(np.uint8) * 255
    visible = cv2.erode(visible, np.ones((5, 5), np.uint8), iterations=1)
    active = visible > 0
    active_fraction = float(np.mean(active))

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    if int(np.count_nonzero(active)) >= max(64, int(gray.size * 0.08)):
        values = lap[active]
        sharp = float(np.var(values))
        mean = float(np.mean(gray[active]))
    else:
        sharp = 0.15 * float(lap.var())
        mean = float(np.mean(gray))
        active_fraction *= 0.25

    sharp_score = float(np.clip(np.log1p(max(0.0, sharp)) / np.log(501.0), 0.0, 1.0))
    exposure = float(np.clip(1.0 - abs(mean - 128.0) / 128.0, 0.0, 1.0))
    face_fraction = float(
        np.clip((bbox[2] * bbox[3]) / max(1.0, image.shape[0] * image.shape[1]) * 8.0, 0.0, 1.0)
    )
    return float(
        0.32 * sharp_score
        + 0.18 * exposure
        + 0.18 * float(detector_score)
        + 0.08 * face_fraction
        + 0.24 * active_fraction
    )


def _components(similarity: np.ndarray, threshold: float) -> list[list[int]]:
    n = similarity.shape[0]
    unseen = set(range(n))
    groups: list[list[int]] = []
    while unseen:
        root = unseen.pop()
        stack = [root]
        group = [root]
        while stack:
            current = stack.pop()
            linked = [j for j in list(unseen) if similarity[current, j] >= threshold]
            for j in linked:
                unseen.remove(j)
                stack.append(j)
                group.append(j)
        groups.append(sorted(group))
    return groups


def _pick_identity_component(similarity: np.ndarray) -> list[int]:
    groups = _components(similarity, FACE_MODEL_DEFAULTS.sface_same_identity_cosine)
    groups.sort(key=lambda g: (len(g), 0 in g, -min(g)), reverse=True)
    return groups[0] if groups else [0]


def _deblur_all(images: list[np.ndarray], model_path: Path | None, hardware_policy: dict[str, Any]) -> tuple[list[np.ndarray], int]:
    if model_path is None or not model_path.is_file():
        return [item.copy() for item in images], 0
    target = str(hardware_policy.get("dnn_target", "cpu")).lower()
    target = "opencl" if target == "opencl" else "cpu"
    tile = max(128, int(hardware_policy.get("heavy_tile_size", 384)))
    engines: dict[str, NafNetDeblurEngine] = {}

    def engine(requested: str) -> NafNetDeblurEngine:
        if requested not in engines:
            engines[requested] = NafNetDeblurEngine(
                model_path,
                target=requested,
                tile_size=tile,
                overlap=RESTORATION_SAFETY_DEFAULTS.tile_overlap,
            )
        return engines[requested]

    output: list[np.ndarray] = []
    applied = 0
    for image in images:
        try:
            try:
                learned = engine(target).infer(image)
            except Exception:
                if target != "opencl":
                    raise
                learned = engine("cpu").infer(image)
            if learned.shape != image.shape:
                learned = cv2.resize(learned, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            strength = RESTORATION_SAFETY_DEFAULTS.nafnet_observed_blend
            output.append(cv2.addWeighted(image, 1.0 - strength, learned, strength, 0.0))
            applied += 1
        except Exception:
            output.append(image.copy())
    return output, applied


def preprocess_and_select_front_base(workspace, model_paths: dict[str, str | Path]) -> PreflightResult:
    """Analyze every photo while keeping imported image #1 as the target canvas.

    `selected_source_index` is a recommended analysis/donor anchor only. It must never
    replace the user-selected MAIN target, pose, frame or final canvas.
    """
    originals = [workspace.primary, *workspace.references]

    original_occlusion = [detect_occlusion_candidates(item) for item in originals]
    original_reliability = [
        detail_reliability_map(item, mask)
        for item, mask in zip(originals, original_occlusion)
    ]

    hardware = workspace.metadata.get("hardware_policy")
    hardware_policy = hardware if isinstance(hardware, dict) else {}
    nafnet_raw = model_paths.get("opencv_nafnet_deblur")
    nafnet = Path(nafnet_raw) if nafnet_raw is not None else None
    processed, deblurred_count = _deblur_all(originals, nafnet, hardware_policy)
    nafnet_indices_raw = globals().get("_last_preflight_nafnet_indices", [])
    nafnet_indices = [int(value) for value in nafnet_indices_raw] if isinstance(nafnet_indices_raw, list) else []

    yunet_raw = model_paths.get("opencv_yunet")
    sface_raw = model_paths.get("opencv_sface")
    pose_raw = model_paths.get("head_pose_mobilenetv2_onnx")
    if yunet_raw is None or sface_raw is None or not Path(yunet_raw).is_file() or not Path(sface_raw).is_file():
        workspace.primary = processed[0].copy()
        workspace.references = [item.copy() for item in processed[1:]]
        workspace.metadata["preflight_deblurred_all"] = deblurred_count == len(processed) and len(processed) > 0
        workspace.metadata["preflight_deblur_evaluated_all"] = deblurred_count == len(processed) and len(processed) > 0
        workspace.metadata["preflight_nafnet_indices"] = nafnet_indices
        workspace.metadata["preflight_nafnet_inference_count"] = len(nafnet_indices)
        workspace.metadata["runtime_source_order"] = list(range(len(processed)))
        workspace.metadata["selected_primary_original_source_index"] = 0
        workspace.metadata["preflight_recommended_front_source_index"] = 0
        workspace.metadata["preflight_original_occlusion_masks"] = original_occlusion
        workspace.metadata["preflight_detail_reliability_maps"] = original_reliability
        return PreflightResult(0, (), deblurred_count, 1, "YuNet/SFace non disponibili: MAIN #1 mantenuta come target")

    target = str(hardware_policy.get("dnn_target", "cpu")).lower()
    target = "opencl" if target == "opencl" else "cpu"
    face = OpenCVZooFaceEngine(yunet_raw, sface_raw, dnn_target=target)
    workspace.metadata["_identity_backend"] = face
    observations = []
    for image in processed:
        try:
            observations.append(face.analyze(image))
        except Exception:
            observations.append(None)

    valid_indices = [i for i, obs in enumerate(observations) if obs is not None and obs.embedding is not None]
    selected = 0
    accepted: set[int] = {0}
    cluster_size = 1
    component_by_source: dict[int, int] = {}

    if valid_indices:
        size = len(valid_indices)
        sim = np.eye(size, dtype=np.float32)
        for a in range(size):
            for b in range(a + 1, size):
                ea = observations[valid_indices[a]].embedding
                eb = observations[valid_indices[b]].embedding
                denom = float(np.linalg.norm(ea) * np.linalg.norm(eb))
                value = float(np.dot(ea, eb) / denom) if denom > 1e-12 else -1.0
                sim[a, b] = sim[b, a] = value
        local_component = _pick_identity_component(sim)
        accepted = {valid_indices[i] for i in local_component}
        cluster_size = len(accepted)
        for source_index in range(len(processed)):
            component_by_source[source_index] = 0 if source_index in accepted else 1

        pose_engine = HeadPoseEngine(pose_raw) if pose_raw is not None and Path(pose_raw).is_file() else None
        ranked: list[tuple[float, float, int]] = []
        for source_index in accepted:
            obs = observations[source_index]
            if obs is None:
                continue
            frontal = 20.0
            if pose_engine is not None:
                try:
                    pose = pose_engine.estimate(_face_crop(processed[source_index], obs.bbox))
                    frontal = pose_frontalness(*pose)
                except Exception:
                    pass
            quality = _quality_score(processed[source_index], obs.bbox, obs.score)
            pose_cost = float(np.clip(frontal / 20.0, 0.0, 1.0))
            base_cost = 0.62 * pose_cost + 0.38 * (1.0 - quality)
            ranked.append((base_cost, -quality, source_index))
        if ranked:
            ranked.sort()
            selected = int(ranked[0][2])

    # Runtime target order is immutable: source 0 remains MAIN, sources 1..9 donors.
    runtime_order = list(range(len(processed)))
    workspace.primary = processed[0].copy()
    workspace.references = [item.copy() for item in processed[1:]]
    workspace.metadata["runtime_source_order"] = runtime_order
    workspace.metadata["selected_primary_original_source_index"] = 0
    workspace.metadata["preflight_recommended_front_source_index"] = selected
    workspace.metadata["preflight_analysis_anchor_source_index"] = selected
    workspace.metadata["preflight_target_canvas_source_index"] = 0
    workspace.metadata["preflight_deblurred_all"] = deblurred_count == len(processed) and len(processed) > 0
    workspace.metadata["preflight_deblur_evaluated_all"] = deblurred_count == len(processed) and len(processed) > 0
    workspace.metadata["preflight_deblurred_count"] = deblurred_count
    workspace.metadata["preflight_nafnet_indices"] = nafnet_indices
    workspace.metadata["preflight_nafnet_inference_count"] = len(nafnet_indices)
    workspace.metadata["preflight_original_occlusion_masks"] = [item.copy() for item in original_occlusion]
    workspace.metadata["preflight_detail_reliability_maps"] = [item.copy() for item in original_reliability]

    pose_engine = HeadPoseEngine(pose_raw) if pose_raw is not None and Path(pose_raw).is_file() else None
    candidates: list[PreflightCandidate] = []
    for i, obs in enumerate(observations):
        if obs is None:
            candidates.append(PreflightCandidate(i, component_by_source.get(i, 1), 0.0, None, 1e6, 0.0, i in accepted))
            continue
        pose = None
        frontal = 1e6
        if pose_engine is not None:
            try:
                pose = pose_engine.estimate(_face_crop(processed[i], obs.bbox))
                frontal = pose_frontalness(*pose)
            except Exception:
                pass
        candidates.append(
            PreflightCandidate(
                i,
                component_by_source.get(i, 0 if i in accepted else 1),
                float(obs.score),
                pose,
                float(frontal),
                _quality_score(processed[i], obs.bbox, obs.score),
                i in accepted,
            )
        )

    workspace.metadata["preflight_candidates"] = [item.__dict__.copy() for item in candidates]
    return PreflightResult(
        selected,
        tuple(candidates),
        deblurred_count,
        cluster_size,
        "MAIN #1 mantenuta come target; migliore source conservata solo come analysis/donor anchor",
    )
