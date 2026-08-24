from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from app.component_bank import canonical_component_masks
from app.damage_taxonomy import DAMAGE_CLASSES, HEALTHY_INDEX, INDEX_TO_CLASS


class _SessionInput(Protocol):
    name: str


class DamageSession(Protocol):
    def get_inputs(self) -> Sequence[_SessionInput]: ...
    def run(self, output_names, input_feed): ...


@dataclass(frozen=True)
class AffectedComponent:
    component: str
    damage_class: str
    affected_fraction: float
    mean_confidence: float


@dataclass(frozen=True)
class DamageMaskResult:
    class_map: np.ndarray
    confidence_map: np.ndarray
    soft_damage_mask: np.ndarray
    binary_damage_mask: np.ndarray
    dominant_damage_class: str
    dominant_confidence: float
    affected_components: tuple[AffectedComponent, ...]


class DamageMaskRuntime:
    """Local CPU ONNX adapter for the frozen CFS facial-damage taxonomy.

    This class performs no downloads. A missing or malformed model fails closed. The
    runtime output is diagnostic/routing evidence only; it never creates observed
    provenance or facial content.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        session: DamageSession | None = None,
        input_size: int = 192,
        damage_confidence_threshold: float = 0.55,
        component_fraction_threshold: float = 0.05,
    ) -> None:
        size = int(input_size)
        if size < 64:
            raise ValueError("input_size must be at least 64")
        threshold = float(damage_confidence_threshold)
        component_threshold = float(component_fraction_threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("damage_confidence_threshold must be in (0,1)")
        if not 0.0 < component_threshold <= 1.0:
            raise ValueError("component_fraction_threshold must be in (0,1]")

        if session is None:
            if model_path is None:
                raise ValueError("model_path is required when no session is supplied")
            path = Path(model_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RuntimeError("onnxruntime is required for DamageMaskNet") from exc
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

        inputs = list(session.get_inputs())
        if len(inputs) != 1 or not getattr(inputs[0], "name", None):
            raise RuntimeError("DamageMaskNet ONNX must expose exactly one named input")
        self._session = session
        self._input_name = str(inputs[0].name)
        self.input_size = size
        self.damage_confidence_threshold = threshold
        self.component_fraction_threshold = component_threshold

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        value = np.asarray(image)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
            raise ValueError("DamageMaskNet expects uint8 BGR HxWx3")
        if value.shape[0] < 8 or value.shape[1] < 8:
            raise ValueError("DamageMaskNet image is too small")
        return value

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(
            image,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        maximum = np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits - maximum, dtype=np.float32)
        denominator = np.sum(exp, axis=1, keepdims=True, dtype=np.float32)
        if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
            raise RuntimeError("DamageMaskNet produced invalid softmax denominator")
        probabilities = exp / denominator
        if not np.isfinite(probabilities).all():
            raise RuntimeError("DamageMaskNet produced non-finite probabilities")
        return probabilities.astype(np.float32, copy=False)

    def _run_probabilities(self, image: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(image)
        outputs = self._session.run(None, {self._input_name: tensor})
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
            raise RuntimeError("DamageMaskNet ONNX must return exactly one logits tensor")
        logits = np.asarray(outputs[0], dtype=np.float32)
        expected = (1, len(DAMAGE_CLASSES), self.input_size, self.input_size)
        if logits.shape != expected:
            raise RuntimeError(
                f"DamageMaskNet logits shape mismatch: got={tuple(logits.shape)} expected={expected}"
            )
        if not np.isfinite(logits).all():
            raise RuntimeError("DamageMaskNet produced non-finite logits")
        return self._softmax(logits)[0]

    @staticmethod
    def _resize_float(map_: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        if map_.shape == (height, width):
            return map_.astype(np.float32, copy=True)
        return cv2.resize(map_.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _resize_labels(map_: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        if map_.shape == (height, width):
            return map_.copy()
        return cv2.resize(map_, (width, height), interpolation=cv2.INTER_NEAREST)

    def infer(
        self,
        image_bgr: np.ndarray,
        *,
        landmarks5: np.ndarray,
        bbox: tuple[int, int, int, int],
    ) -> DamageMaskResult:
        image = self._validate_image(image_bgr)
        height, width = image.shape[:2]
        probabilities = self._run_probabilities(image)

        class_small = np.argmax(probabilities, axis=0).astype(np.uint8)
        confidence_small = np.max(probabilities, axis=0).astype(np.float32)
        soft_damage_small = (1.0 - probabilities[HEALTHY_INDEX]).astype(np.float32)
        admitted_small = (
            (class_small != HEALTHY_INDEX)
            & (confidence_small >= self.damage_confidence_threshold)
        )

        class_map = self._resize_labels(class_small, (height, width)).astype(np.uint8, copy=False)
        confidence_map = np.clip(
            self._resize_float(confidence_small, (height, width)), 0.0, 1.0
        ).astype(np.float32, copy=False)
        soft_damage = np.clip(
            self._resize_float(soft_damage_small, (height, width)), 0.0, 1.0
        ).astype(np.float32, copy=False)
        binary = self._resize_labels(
            np.where(admitted_small, 255, 0).astype(np.uint8),
            (height, width),
        ).astype(np.uint8, copy=False)

        # The class map is retained for diagnostics even for low-confidence pixels; all
        # routing authority below is gated by the admitted binary mask.
        admitted = binary > 0
        if np.any(admitted):
            labels, counts = np.unique(class_map[admitted], return_counts=True)
            valid = [(int(label), int(count)) for label, count in zip(labels, counts) if int(label) != HEALTHY_INDEX]
            if not valid:
                dominant_index = HEALTHY_INDEX
                dominant_confidence = 0.0
            else:
                dominant_index = max(valid, key=lambda item: item[1])[0]
                selected = admitted & (class_map == dominant_index)
                dominant_confidence = float(np.mean(confidence_map[selected])) if np.any(selected) else 0.0
        else:
            dominant_index = HEALTHY_INDEX
            dominant_confidence = 0.0

        component_masks = canonical_component_masks((height, width), landmarks5, bbox)
        affected: list[AffectedComponent] = []
        for component, region_u8 in component_masks.items():
            region = region_u8 > 0
            area = int(np.count_nonzero(region))
            if area <= 0:
                continue
            damaged = region & admitted
            damaged_pixels = int(np.count_nonzero(damaged))
            fraction = float(damaged_pixels / area)
            if fraction < self.component_fraction_threshold or damaged_pixels <= 0:
                continue

            labels, counts = np.unique(class_map[damaged], return_counts=True)
            valid = [(int(label), int(count)) for label, count in zip(labels, counts) if int(label) != HEALTHY_INDEX]
            if not valid:
                continue
            damage_index = max(valid, key=lambda item: item[1])[0]
            selected = damaged & (class_map == damage_index)
            mean_confidence = float(np.mean(confidence_map[selected])) if np.any(selected) else 0.0
            affected.append(
                AffectedComponent(
                    component=component,
                    damage_class=INDEX_TO_CLASS[damage_index],
                    affected_fraction=fraction,
                    mean_confidence=mean_confidence,
                )
            )

        affected.sort(key=lambda item: (-item.affected_fraction, item.component))
        return DamageMaskResult(
            class_map=class_map,
            confidence_map=confidence_map,
            soft_damage_mask=soft_damage,
            binary_damage_mask=binary,
            dominant_damage_class=INDEX_TO_CLASS[dominant_index],
            dominant_confidence=float(dominant_confidence),
            affected_components=tuple(affected),
        )
