from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import numpy as np

from app.resource_budget import (
    ResourceBudget,
    apply_resource_budget,
    assert_memory_within_budget,
    detect_resource_budget,
    resource_snapshot,
)


GENERATED_MODEL_INFERRED = 'GENERATED_MODEL_INFERRED'


@dataclass(frozen=True)
class RestorationContext:
    damage_class: str
    severity: str
    component: str | None = None
    identity_threshold: float = 0.363
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RestorationCandidate:
    image: np.ndarray
    model_key: str
    model_version: str
    backend: str
    generated_mask: np.ndarray
    provenance_class: str = GENERATED_MODEL_INFERRED
    identity_score: float | None = None
    identity_pass: bool | None = None
    quality_metrics: dict[str, float | int | str | bool | None] = field(default_factory=dict)
    timing_seconds: dict[str, float] = field(default_factory=dict)
    resource: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    rejection_reason: str | None = None

    def to_report(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop('image', None)
        payload.pop('generated_mask', None)
        payload['image_shape'] = list(self.image.shape)
        payload['generated_mask_shape'] = list(self.generated_mask.shape)
        return payload


class FaceRestorerBackend(Protocol):
    key: str
    version: str
    backend_name: str
    estimated_load_bytes: int

    def load(self) -> None: ...

    def restore(self, face_bgr: np.ndarray, context: RestorationContext) -> RestorationCandidate: ...

    def unload(self) -> None: ...


class FaceRestorerAdapter:
    """Run one heavy face-restoration backend at a time under the local budget.

    The adapter deliberately owns lifecycle rather than letting model-specific code
    stay resident. A backend is loaded, measured, executed and unloaded in a single
    guarded operation. This is the common boundary for GPEN/GFPGAN/CodeFormer and
    later specialist challengers.
    """

    def __init__(self, budget: ResourceBudget | None = None) -> None:
        self.budget = budget or detect_resource_budget(0.80)
        if self.budget.max_parallel_heavy_models != 1:
            raise ValueError('Paper Quality requires max_parallel_heavy_models == 1')
        self._active_model: str | None = None

    @property
    def active_model(self) -> str | None:
        return self._active_model

    def restore(
        self,
        backend: FaceRestorerBackend,
        face_bgr: np.ndarray,
        context: RestorationContext,
    ) -> RestorationCandidate:
        if self._active_model is not None:
            raise RuntimeError(
                f'Heavy model {self._active_model} is still active; '
                f'cannot load {backend.key}'
            )
        if not isinstance(face_bgr, np.ndarray) or face_bgr.ndim != 3 or face_bgr.shape[2] != 3:
            raise ValueError('face_bgr must be an HxWx3 numpy array')
        if face_bgr.dtype != np.uint8:
            raise ValueError('face_bgr must use uint8 pixels')

        apply_resource_budget(self.budget)
        reserve = max(0, int(getattr(backend, 'estimated_load_bytes', 0)))
        assert_memory_within_budget(
            self.budget,
            stage=f'{backend.key}_preload',
            reserve_bytes=reserve,
        )

        before = resource_snapshot(self.budget)
        load_start = time.perf_counter()
        self._active_model = str(backend.key)
        candidate: RestorationCandidate | None = None
        try:
            backend.load()
            load_seconds = time.perf_counter() - load_start
            assert_memory_within_budget(self.budget, stage=f'{backend.key}_postload')
            after_load = resource_snapshot(self.budget)

            infer_start = time.perf_counter()
            candidate = backend.restore(face_bgr, context)
            inference_seconds = time.perf_counter() - infer_start
            assert_memory_within_budget(self.budget, stage=f'{backend.key}_post_inference')
            after_inference = resource_snapshot(self.budget)

            if candidate.model_key != backend.key:
                raise RuntimeError('Backend returned a candidate with the wrong model key')
            if candidate.provenance_class != GENERATED_MODEL_INFERRED:
                raise RuntimeError('Paper Quality model output must be GENERATED_MODEL_INFERRED')
            if candidate.image.shape != face_bgr.shape:
                raise RuntimeError(
                    f'Candidate shape mismatch: {candidate.image.shape} != {face_bgr.shape}'
                )
            if candidate.generated_mask.shape != face_bgr.shape[:2]:
                raise RuntimeError('Generated mask must match face spatial dimensions')
            if candidate.generated_mask.dtype != np.uint8:
                raise RuntimeError('Generated mask must be uint8')

            candidate.timing_seconds.setdefault('model_load', float(load_seconds))
            candidate.timing_seconds.setdefault('inference', float(inference_seconds))
            candidate.resource.update({
                'budget': self.budget.to_dict(),
                'before_load': before,
                'after_load': after_load,
                'after_inference': after_inference,
            })
            return candidate
        finally:
            try:
                backend.unload()
            finally:
                self._active_model = None
                gc.collect()
                assert_memory_within_budget(self.budget, stage=f'{backend.key}_post_unload')
                if candidate is not None:
                    candidate.resource['post_unload'] = resource_snapshot(self.budget)
