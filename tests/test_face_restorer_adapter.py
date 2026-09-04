from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / 'app'

# Load the two research modules without executing the production app/__init__.py policy
# chain. These tests are specifically for the isolated Paper Quality lifecycle boundary.
package = types.ModuleType('app')
package.__path__ = [str(APP)]
sys.modules.setdefault('app', package)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


resource = _load('app.resource_budget', APP / 'resource_budget.py')
adapter_mod = _load('app.face_restorer_adapter', APP / 'face_restorer_adapter.py')

FaceRestorerAdapter = adapter_mod.FaceRestorerAdapter
RestorationCandidate = adapter_mod.RestorationCandidate
RestorationContext = adapter_mod.RestorationContext
RouteExecutionBlocked = adapter_mod.RouteExecutionBlocked
GENERATED_MODEL_INFERRED = adapter_mod.GENERATED_MODEL_INFERRED


class FakeBackend:
    key = 'fake'
    version = '1'
    backend_name = 'cpu'
    estimated_load_bytes = 0

    def __init__(self, *, fail: bool = False, bad_provenance: bool = False) -> None:
        self.loaded = False
        self.unloaded = False
        self.fail = fail
        self.bad_provenance = bad_provenance
        self.load_calls = 0
        self.restore_calls = 0

    def load(self) -> None:
        self.load_calls += 1
        self.loaded = True

    def restore(self, face_bgr: np.ndarray, context: RestorationContext) -> RestorationCandidate:
        self.restore_calls += 1
        assert self.loaded
        if self.fail:
            raise RuntimeError('synthetic backend failure')
        return RestorationCandidate(
            image=face_bgr.copy(),
            model_key=self.key,
            model_version=self.version,
            backend=self.backend_name,
            generated_mask=np.full(face_bgr.shape[:2], 255, dtype=np.uint8),
            provenance_class=(
                'OBSERVED_MAIN' if self.bad_provenance else GENERATED_MODEL_INFERRED
            ),
        )

    def unload(self) -> None:
        self.unloaded = True
        self.loaded = False


def _budget():
    detected = resource.detect_resource_budget(0.80)
    # Unit tests exercise lifecycle semantics. Memory enforcement itself has its own
    # dedicated regression workflow, so make this test independent of runner RSS.
    return resource.ResourceBudget(
        max_fraction=0.80,
        logical_processors=detected.logical_processors,
        allowed_processors=detected.allowed_processors,
        total_ram_bytes=None,
        process_ram_limit_bytes=None,
        max_parallel_heavy_models=1,
    )


def _route(
    *,
    damage_kind: str = 'JPEG_ARTIFACT',
    selected_model_key: str | None = 'fake',
    qualified: bool = True,
    mask_pixels: int = 100,
    candidate_model_keys: tuple[str, ...] = ("fake",),
):
    # The adapter consumes only the immutable route authorization fields. The route
    # planner itself is covered independently by tests/test_damage_router.py.
    return types.SimpleNamespace(
        damage_kind=damage_kind,
        selected_model_key=selected_model_key,
        selected_model_attestation_sha256=('a' * 64 if qualified else None),
        qualified_for_execution=qualified,
        mask_pixels=mask_pixels,
        candidate_model_keys=candidate_model_keys,
        reason=(
            'qualified_route_planned_but_execution_not_performed'
            if qualified
            else 'no_production_qualified_model_for_route'
        ),
    )


def _validation_qualification(
    *,
    model_key: str = "fake",
    tier: str = "VALIDATION",
    production: bool = False,
):
    return types.SimpleNamespace(
        model_key=model_key,
        evidence_tier=tier,
        production_qualified=production,
    )


def test_adapter_loads_runs_and_always_unloads_one_model() -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    candidate = adapter.restore(backend, face, RestorationContext('JPEG_ARTIFACT', 'MEDIUM'))
    assert backend.unloaded is True
    assert adapter.active_model is None
    assert candidate.provenance_class == GENERATED_MODEL_INFERRED
    assert candidate.resource['budget']['max_fraction'] == 0.80
    assert candidate.resource['budget']['max_parallel_heavy_models'] == 1


def test_route_authorization_executes_exact_selected_jpeg_backend_once() -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    candidate = adapter.restore_for_route(
        _route(),
        backend,
        face,
        RestorationContext('JPEG_ARTIFACT', 'MEDIUM'),
    )
    assert candidate.model_key == 'fake'
    assert backend.load_calls == 1
    assert backend.restore_calls == 1
    assert backend.unloaded is True


def test_validation_route_executes_candidate_then_unloads_without_production_authority() -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    candidate = adapter.restore_for_validation_route(
        _route(qualified=False, selected_model_key=None),
        _validation_qualification(),
        backend,
        face,
        RestorationContext(
            "JPEG_ARTIFACT",
            "MEDIUM",
            metadata={"validation_only": True},
        ),
    )
    assert candidate.model_key == "fake"
    assert backend.load_calls == 1
    assert backend.restore_calls == 1
    assert backend.unloaded is True
    assert adapter.active_model is None


@pytest.mark.parametrize(
    ("damage_kind", "candidate_keys"),
    [
        ("GAUSSIAN_BLUR", ("opencv_nafnet_deblur",)),
        ("OCCLUSION", ("ref_face_inpainting",)),
        ("HEALTHY", ()),
    ],
)
def test_validation_route_never_loads_backend_for_unrelated_damage(
    damage_kind: str,
    candidate_keys: tuple[str, ...],
) -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    with pytest.raises(RouteExecutionBlocked, match="validation_route_model_not_allowed"):
        adapter.restore_for_validation_route(
            _route(
                damage_kind=damage_kind,
                qualified=False,
                selected_model_key=None,
                candidate_model_keys=candidate_keys,
            ),
            _validation_qualification(),
            backend,
            np.zeros((64, 64, 3), dtype=np.uint8),
            RestorationContext(
                damage_kind,
                "MEDIUM",
                metadata={"validation_only": True},
            ),
        )
    assert backend.load_calls == 0
    assert backend.restore_calls == 0


def test_validation_route_requires_explicit_scope_and_validation_tier() -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    route = _route(qualified=False, selected_model_key=None)
    with pytest.raises(RouteExecutionBlocked, match="explicit_scope"):
        adapter.restore_for_validation_route(
            route,
            _validation_qualification(),
            backend,
            face,
            RestorationContext("JPEG_ARTIFACT", "MEDIUM"),
        )
    with pytest.raises(RouteExecutionBlocked, match="validation_qualification"):
        adapter.restore_for_validation_route(
            route,
            _validation_qualification(tier="DEVELOPMENT"),
            backend,
            face,
            RestorationContext(
                "JPEG_ARTIFACT",
                "MEDIUM",
                metadata={"validation_only": True},
            ),
        )
    assert backend.load_calls == 0


def test_unqualified_route_never_loads_or_runs_backend() -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RouteExecutionBlocked, match='route_not_qualified'):
        adapter.restore_for_route(
            _route(qualified=False, selected_model_key=None),
            backend,
            face,
            RestorationContext('JPEG_ARTIFACT', 'MEDIUM'),
        )
    assert backend.load_calls == 0
    assert backend.restore_calls == 0


def test_route_selected_model_mismatch_never_loads_backend() -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RouteExecutionBlocked, match='route_model_mismatch'):
        adapter.restore_for_route(
            _route(selected_model_key='fbcnn'),
            backend,
            face,
            RestorationContext('JPEG_ARTIFACT', 'MEDIUM'),
        )
    assert backend.load_calls == 0
    assert backend.restore_calls == 0


def test_route_context_mismatch_never_loads_backend() -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RouteExecutionBlocked, match='route_context_mismatch'):
        adapter.restore_for_route(
            _route(damage_kind='JPEG_ARTIFACT'),
            backend,
            face,
            RestorationContext('GAUSSIAN_BLUR', 'MEDIUM'),
        )
    assert backend.load_calls == 0
    assert backend.restore_calls == 0


def test_route_without_admitted_pixels_never_loads_backend() -> None:
    backend = FakeBackend()
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RouteExecutionBlocked, match='route_has_no_admitted_damage_pixels'):
        adapter.restore_for_route(
            _route(mask_pixels=0),
            backend,
            face,
            RestorationContext('JPEG_ARTIFACT', 'MEDIUM'),
        )
    assert backend.load_calls == 0
    assert backend.restore_calls == 0


def test_adapter_unloads_backend_after_inference_exception() -> None:
    backend = FakeBackend(fail=True)
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match='synthetic backend failure'):
        adapter.restore(backend, face, RestorationContext('MIXED', 'SEVERE'))
    assert backend.unloaded is True
    assert adapter.active_model is None


def test_adapter_rejects_generated_pixels_mislabeled_as_observed() -> None:
    backend = FakeBackend(bad_provenance=True)
    adapter = FaceRestorerAdapter(_budget())
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match='GENERATED_MODEL_INFERRED'):
        adapter.restore(backend, face, RestorationContext('STICKER', 'SEVERE'))
    assert backend.unloaded is True
    assert adapter.active_model is None


def test_adapter_refuses_second_heavy_model_while_one_is_active() -> None:
    adapter = FaceRestorerAdapter(_budget())
    adapter._active_model = 'already_loaded'
    face = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match='still active'):
        adapter.restore(FakeBackend(), face, RestorationContext('BLUR', 'MEDIUM'))


def test_adapter_rejects_non_uint8_or_bad_shape_input() -> None:
    adapter = FaceRestorerAdapter(_budget())
    with pytest.raises(ValueError):
        adapter.restore(
            FakeBackend(),
            np.zeros((64, 64, 3), dtype=np.float32),
            RestorationContext('BLUR', 'MEDIUM'),
        )
    with pytest.raises(ValueError):
        adapter.restore(
            FakeBackend(),
            np.zeros((64, 64), dtype=np.uint8),
            RestorationContext('BLUR', 'MEDIUM'),
        )
