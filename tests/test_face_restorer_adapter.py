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

    def load(self) -> None:
        self.loaded = True

    def restore(self, face_bgr: np.ndarray, context: RestorationContext) -> RestorationCandidate:
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
