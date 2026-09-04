from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from app.resource_budget import (
    ResourceBudget,
    apply_resource_budget,
    assert_memory_within_budget,
    detect_resource_budget,
    resource_snapshot,
)

if TYPE_CHECKING:
    from app.damage_router import DamageRoutePlan
    from app.model_qualification import ModelQualification


GENERATED_MODEL_INFERRED = 'GENERATED_MODEL_INFERRED'


class RouteExecutionBlocked(RuntimeError):
    """Raised before model load when a damage route cannot authorize execution."""


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
    upstream_repository: str | None = None
    upstream_revision: str | None = None
    checkpoint_sha256: str | None = None
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

    def restore_for_route(
        self,
        plan: DamageRoutePlan,
        backend: FaceRestorerBackend,
        face_bgr: np.ndarray,
        context: RestorationContext,
    ) -> RestorationCandidate:
        """Execute a backend only when the audited damage route authorizes that exact model.

        The route planner remains responsible for selecting a model from production
        qualification evidence. This boundary prevents a caller from bypassing that
        decision and loading FBCNN (or another heavy specialist) for an unqualified,
        healthy, unrelated, or differently-selected damage route.
        """
        selected = plan.selected_model_key
        attestation = plan.selected_model_attestation_sha256
        if not plan.qualified_for_execution or selected is None or not attestation:
            raise RouteExecutionBlocked(
                f'route_not_qualified:{plan.damage_kind}:{plan.reason}'
            )
        if str(selected) != str(backend.key):
            raise RouteExecutionBlocked(
                f'route_model_mismatch:selected={selected}:backend={backend.key}'
            )
        if int(plan.mask_pixels) <= 0:
            raise RouteExecutionBlocked('route_has_no_admitted_damage_pixels')
        if str(context.damage_class).upper() != str(plan.damage_kind).upper():
            raise RouteExecutionBlocked(
                f'route_context_mismatch:route={plan.damage_kind}:context={context.damage_class}'
            )
        return self.restore(backend, face_bgr, context)

    def restore_for_validation_route(
        self,
        plan: DamageRoutePlan,
        qualification: ModelQualification,
        backend: FaceRestorerBackend,
        face_bgr: np.ndarray,
        context: RestorationContext,
    ) -> RestorationCandidate:
        """Execute an explicitly non-production candidate without granting pixel authority.

        The caller may measure and report the returned candidate, but must not feed it
        into final fusion. Production-qualified routes use ``restore_for_route``. This
        separate boundary lets the installed application exercise a real backend while
        keeping missing installer/target-hardware gates fail-closed.
        """
        if context.metadata.get("validation_only") is not True:
            raise RouteExecutionBlocked("validation_route_requires_explicit_scope")
        if qualification.production_qualified or qualification.evidence_tier != "VALIDATION":
            raise RouteExecutionBlocked("validation_route_requires_validation_qualification")
        if str(qualification.model_key) != str(backend.key):
            raise RouteExecutionBlocked(
                f"validation_qualification_model_mismatch:qualification="
                f"{qualification.model_key}:backend={backend.key}"
            )
        if str(backend.key) not in tuple(str(item) for item in plan.candidate_model_keys):
            raise RouteExecutionBlocked(
                f"validation_route_model_not_allowed:route={plan.damage_kind}:backend={backend.key}"
            )
        if plan.qualified_for_execution:
            raise RouteExecutionBlocked("production_route_must_use_production_boundary")
        if int(plan.mask_pixels) <= 0:
            raise RouteExecutionBlocked("route_has_no_admitted_damage_pixels")
        if str(context.damage_class).upper() != str(plan.damage_kind).upper():
            raise RouteExecutionBlocked(
                f"route_context_mismatch:route={plan.damage_kind}:context={context.damage_class}"
            )
        return self.restore(backend, face_bgr, context)

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
