from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest

import app.automatic as automatic_module
import app.__main__ as desktop_entrypoint
import app.worker as worker_module
from app.candidate_selector_v2 import (
    CalibratedRankingWeights,
    CandidateSelectionPolicy,
    RANKING_METRICS,
)
from app.damage_mask_runtime import DamageMaskRuntime
from app.damage_taxonomy import CLASS_TO_INDEX, DAMAGE_CLASSES
from app.execution import ExecutionResult, Workspace
from app.face_restorer_adapter import RestorationCandidate
from app.fbcnn_upstream_backend import APPROVED_CHECKPOINT_SHA256
from app.installed_paper_quality_runtime import InstalledPaperQualityRuntime
from app.model_qualification import nonproduction_model_qualification
from app.pipeline import BlockKind
from app.settings import RuntimeSettings
from app.worker import PipelineWorker


class _Input:
    name = "image"


class _OpaqueBlockSession:
    def get_inputs(self):
        return [_Input()]

    def run(self, output_names, input_feed):
        del output_names, input_feed
        logits = np.full((1, len(DAMAGE_CLASSES), 64, 64), -8.0, dtype=np.float32)
        logits[:, CLASS_TO_INDEX["HEALTHY"], :, :] = 8.0
        logits[:, CLASS_TO_INDEX["OPAQUE_BLOCK"], 22:31, 20:29] = 12.0
        return [logits]


class _DamageSession:
    def __init__(self, damage_class: str) -> None:
        self.damage_class = damage_class

    def get_inputs(self):
        return [_Input()]

    def run(self, output_names, input_feed):
        del output_names, input_feed
        logits = np.full((1, len(DAMAGE_CLASSES), 64, 64), -8.0, dtype=np.float32)
        logits[:, CLASS_TO_INDEX["HEALTHY"], :, :] = 8.0
        if self.damage_class != "HEALTHY":
            logits[:, CLASS_TO_INDEX[self.damage_class], 18:45, 17:47] = 12.0
        return [logits]


class _ValidationFBCNN:
    key = "fbcnn"
    version = "synthetic-validation-backend"
    backend_name = "synthetic-fbcnn-contract-backend"
    estimated_load_bytes = 0

    def __init__(self) -> None:
        self.load_calls = 0
        self.restore_calls = 0
        self.unload_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def restore(self, face_bgr, context):
        self.restore_calls += 1
        assert context.damage_class == "JPEG_ARTIFACT"
        output = np.clip(face_bgr.astype(np.int16) + 4, 0, 255).astype(np.uint8)
        return RestorationCandidate(
            image=output,
            model_key=self.key,
            model_version=self.version,
            backend=self.backend_name,
            generated_mask=np.full(face_bgr.shape[:2], 255, dtype=np.uint8),
            upstream_repository="jiaxi-jiang/FBCNN",
            upstream_revision="54d1831927506b3247e2d4d245abb4f4dab1a1cd",
            checkpoint_sha256=APPROVED_CHECKPOINT_SHA256,
        )

    def unload(self) -> None:
        self.unload_calls += 1


@dataclass
class _Bootstrap:
    paths: dict[str, Path]
    errors: dict[str, str]
    face_ready: bool = True
    standard_ready: bool = True
    inpaint_ready: bool = True
    deblur_ready: bool = True
    semantic_ready: bool = True
    pose_ready: bool = True


@dataclass
class _HardwareProfile:
    opencl_functional: bool = False
    acceleration_available: bool = False
    profile_class: str = "synthetic-installed-path"

    def to_dict(self) -> dict[str, object]:
        return {
            "opencl_functional": self.opencl_functional,
            "acceleration_available": self.acceleration_available,
            "profile_class": self.profile_class,
        }


def _images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height = width = 96
    y, x = np.mgrid[:height, :width]
    clean = np.empty((height, width, 3), dtype=np.uint8)
    clean[:, :, 0] = np.uint8(70 + (x % 90))
    clean[:, :, 1] = np.uint8(85 + (y % 90))
    clean[:, :, 2] = np.uint8(100 + ((x + y) % 80))
    cv2.circle(clean, (48, 48), 31, (132, 164, 194), -1)
    cv2.circle(clean, (36, 39), 4, (35, 45, 55), -1)
    cv2.circle(clean, (60, 39), 4, (35, 45, 55), -1)
    cv2.ellipse(clean, (48, 61), (10, 4), 0, 0, 180, (40, 50, 70), 2)
    main = clean.copy()
    target = np.zeros((height, width), dtype=np.uint8)
    target[33:47, 30:44] = 255
    main[target > 0] = (4, 4, 4)
    return main, clean, target


def _selection_policy() -> CandidateSelectionPolicy:
    return CandidateSelectionPolicy(
        CalibratedRankingWeights(
            calibration_id="synthetic-installed-path-dev-v1",
            split="DEVELOPMENT",
            weights={name: 1.0 / len(RANKING_METRICS) for name in RANKING_METRICS},
        ),
        max_landmark_geometry_drift_px=2.0,
    )


def _install_deterministic_geometry(runner, target: np.ndarray) -> None:
    points = np.asarray(
        [[36.0, 39.0], [60.0, 39.0], [48.0, 51.0], [40.0, 63.0], [56.0, 63.0]],
        dtype=np.float32,
    )

    def no_change(block, parameters):
        del parameters
        return ExecutionResult(block.key, runner.executor.workspace.copy_primary(), {"engine": "synthetic-no-op"})

    def landmarks(block, parameters):
        del parameters
        runner.executor.workspace.metadata.update(
            {
                "primary_landmarks5": points.copy(),
                "primary_bbox": (18, 14, 60, 68),
                "primary_landmark_confidence": 1.0,
                "reference_landmarks5": [points.copy()],
                "reference_landmark_confidence": [1.0],
                "face_backend": "synthetic-installed-path-geometry",
            }
        )
        return ExecutionResult(
            block.key,
            runner.executor.workspace.copy_primary(),
            {"backend": "synthetic-installed-path-geometry", "landmark_count": 5},
        )

    def align(block, parameters):
        del parameters
        workspace = runner.executor.workspace
        workspace.aligned_references = [workspace.references[0].copy()]
        workspace.metadata.update(
            {
                "aligned_reference_source_indices": [0],
                "aligned_reference_original_source_indices": [1],
                "aligned_reference_support_masks": [np.full(target.shape, 255, dtype=np.uint8)],
                "aligned_reference_identity_scores": [0.99],
                "aligned_reference_identity_verified": [True],
                "aligned_reference_partial_geometry_verified": [False],
            }
        )
        return ExecutionResult(block.key, workspace.copy_primary(), {"aligned": 1, "identity_filter_applied": True})

    def occlusion(block, parameters):
        del parameters
        workspace = runner.executor.workspace
        workspace.occlusion_masks = [target.copy(), np.zeros_like(target)]
        workspace.metadata["reference_consensus_occlusion"] = target.copy()
        return ExecutionResult(block.key, workspace.copy_primary(), {"consensus_pixels": int(np.count_nonzero(target))})

    runner.executor._handlers[BlockKind.DEBLUR] = no_change
    runner.executor._handlers[BlockKind.LANDMARKS] = landmarks
    runner.executor._handlers[BlockKind.ALIGN] = align
    runner.executor._handlers[BlockKind.OCCLUSION_MASK] = occlusion
    runner.executor._handlers[BlockKind.REGION_SELECT] = no_change


def test_real_worker_path_invokes_paper_quality_modules_and_preserves_authority(
    monkeypatch,
    tmp_path: Path,
) -> None:
    main, reference, target = _images()
    runtime = InstalledPaperQualityRuntime(
        damage_runtime=DamageMaskRuntime(session=_OpaqueBlockSession(), input_size=64),
        model_qualifications={
            "ref_face_inpainting": nonproduction_model_qualification(
                "ref_face_inpainting",
                "VALIDATION",
                ("synthetic-installed-path:no-production-authority",),
            )
        },
        selection_policy=_selection_policy(),
    )
    workspace = Workspace(primary=main.copy(), references=[reference.copy()])
    output = tmp_path / "installed-path.png"

    monkeypatch.setattr(
        worker_module,
        "load_runtime_settings",
        lambda: RuntimeSettings("balanced", "https://example.invalid/manifest.json", True),
    )
    monkeypatch.setattr(
        worker_module,
        "resolve_local_production_models",
        lambda: _Bootstrap(paths={}, errors={}),
    )
    monkeypatch.setattr(
        worker_module,
        "detect_hardware_profile",
        lambda **kwargs: _HardwareProfile(),
    )

    original_init = automatic_module.AutomaticPipelineRunner.__init__

    def configured_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _install_deterministic_geometry(self, target)

    monkeypatch.setattr(automatic_module.AutomaticPipelineRunner, "__init__", configured_init)

    worker = PipelineWorker(
        workspace,
        output,
        upscale=1,
        paper_quality_runtime=runtime,
    )
    completed: list[object] = []
    failures: list[str] = []
    progress_details: list[dict[str, object]] = []
    worker.completed.connect(completed.append)
    worker.failed.connect(failures.append)
    worker.progress_detail.connect(progress_details.append)
    worker.run()

    assert failures == []
    assert len(completed) == 1
    result = completed[0]
    block = next(item for item in result.results if item.block == "inpaint")
    assert block.details["engine"] == "installed-paper-quality-runtime-v1"
    assert block.details["paper_quality_runtime_wired"] is True
    assert block.details["decision"] == "PASS"
    assert int(block.details["observed_reference_pixels"]) > 0
    assert block.details["generated_pixels"] == 0
    assert block.details["wrong_person_final_pixels"] == 0
    assert block.details["provenance_violations"] == 0
    assert block.details["outside_authority_changed_pixels"] == 0
    assert block.details["identity_guardrail"]["accepted"] is True
    assert block.details["identity_guardrail"]["trusted_observed_reference_transfer"] is True
    assert np.array_equal(block.image[target == 0], main[target == 0])
    assert np.count_nonzero(workspace.provenance_map == 1) > 0
    assert np.count_nonzero(workspace.provenance_map == 65535) == 0

    trace = {item["stage"]: item["status"] for item in block.details["paper_quality_trace"]}
    assert trace == {
            "DamageMaskRuntime": "EXECUTED",
            "damage_router": "EXECUTED",
            "model_qualification": "EXECUTED",
            "model_execution": "NOT_APPLICABLE",
            "PersonalizedReferenceBank": "EXECUTED",
        "component_selector": "EXECUTED",
        "reference_first_repair": "EXECUTED",
        "candidate_selector": "EXECUTED",
        "component_aware_fusion": "EXECUTED",
        "PaperQualityRuntime": "EXECUTED",
        "provenance": "VERIFIED",
    }
    block_evidence = [
        item for item in progress_details
        if item.get("block_index") == 8 and item.get("status") == "PASS"
    ]
    assert block_evidence
    assert block_evidence[-1]["engine"] == "installed-paper-quality-runtime-v1"
    assert output.is_file()


def test_feature_flag_disabled_does_not_replace_legacy_block_eight() -> None:
    main, _, _ = _images()

    class _MustNotRun:
        def run(self, *args, **kwargs):
            raise AssertionError("Paper Quality ran while feature flag was disabled")

    runner = automatic_module.AutomaticPipelineRunner(
        Workspace(primary=main, metadata={"paper_quality_enabled": False}),
        paper_quality_runtime=_MustNotRun(),
    )
    assert runner.executor.workspace.metadata["paper_quality_runtime_wired"] is False


def test_desktop_entrypoint_is_structurally_connected_to_the_dynamically_tested_path() -> None:
    entrypoint_source = inspect.getsource(desktop_entrypoint.main)
    window_source = Path(worker_module.__file__).with_name("main_window.py").read_text(
        encoding="utf-8"
    )
    worker_source = inspect.getsource(worker_module.PipelineWorker.run)

    assert "MainWindow()" in entrypoint_source
    assert "PipelineWorker(workspace, output" in window_source
    assert "AutomaticPipelineRunner(" in worker_source
    assert "paper_quality_runtime=self._paper_quality_runtime" in worker_source


def test_malformed_geometry_abstains_without_invoking_damage_inference() -> None:
    main, _, _ = _images()
    runtime = InstalledPaperQualityRuntime(
        damage_runtime=DamageMaskRuntime(session=_OpaqueBlockSession(), input_size=64)
    )
    workspace = Workspace(
        primary=main.copy(),
        metadata={
            "primary_landmarks5": np.zeros((5, 2), dtype=np.float32),
            "primary_bbox": 42,
        },
    )

    result = runtime.run(workspace, immutable_main=main)

    assert result.runtime_result.decision == "ABSTAIN"
    assert result.details["damage_runtime_error"] == "paper_quality_geometry_unavailable"
    trace = {item["stage"]: item["status"] for item in result.details["paper_quality_trace"]}
    assert trace["candidate_selector"] == "NOT_CONFIGURED"
    assert trace["component_aware_fusion"] == "NOT_EXECUTED"
    assert np.array_equal(result.image, main)
    assert np.count_nonzero(result.provenance_map) == 0


def test_jpeg_route_executes_validation_fbcnn_but_never_fuses_nonproduction_pixels() -> None:
    main, _, _ = _images()
    backend = _ValidationFBCNN()
    qualifications = {
        "fbcnn": nonproduction_model_qualification(
            "fbcnn",
            "VALIDATION",
            ("synthetic-installed-route-contract",),
        )
    }
    runtime = InstalledPaperQualityRuntime(
        damage_runtime=DamageMaskRuntime(
            session=_DamageSession("JPEG_ARTIFACT"),
            input_size=64,
        ),
        model_qualifications=qualifications,
        validation_backend_factories={"fbcnn": lambda: backend},
    )
    landmarks = np.asarray(
        [[36.0, 39.0], [60.0, 39.0], [48.0, 51.0], [40.0, 63.0], [56.0, 63.0]],
        dtype=np.float32,
    )
    workspace = Workspace(
        primary=main.copy(),
        metadata={"primary_landmarks5": landmarks, "primary_bbox": (18, 14, 60, 68)},
    )

    result = runtime.run(workspace, immutable_main=main)

    assert backend.load_calls == 1
    assert backend.restore_calls == 1
    assert backend.unload_calls == 1
    assert result.details["damage_route"]["damage_kind"] == "JPEG_ARTIFACT"
    assert result.details["validation_candidates_fused_to_final"] is False
    assert result.details["generated_pixels"] == 0
    assert np.array_equal(result.image, main)
    executed = result.details["models_actually_executed"]
    assert [item["model_key"] for item in executed] == ["fbcnn"]
    assert executed[0]["checkpoint_sha256"] == APPROVED_CHECKPOINT_SHA256
    model_trace = next(
        item for item in result.details["paper_quality_trace"] if item["stage"] == "model_execution"
    )
    assert model_trace["status"] == "VALIDATION_EXECUTED_NOT_FUSED"


@pytest.mark.parametrize("damage_class", ["BLUR", "STICKER", "HEALTHY"])
def test_non_jpeg_routes_never_construct_or_execute_fbcnn(damage_class: str) -> None:
    main, _, _ = _images()
    constructed: list[_ValidationFBCNN] = []

    def factory():
        backend = _ValidationFBCNN()
        constructed.append(backend)
        return backend

    runtime = InstalledPaperQualityRuntime(
        damage_runtime=DamageMaskRuntime(
            session=_DamageSession(damage_class),
            input_size=64,
        ),
        model_qualifications={
            "fbcnn": nonproduction_model_qualification(
                "fbcnn",
                "VALIDATION",
                ("synthetic-installed-route-contract",),
            )
        },
        validation_backend_factories={"fbcnn": factory},
    )
    landmarks = np.asarray(
        [[36.0, 39.0], [60.0, 39.0], [48.0, 51.0], [40.0, 63.0], [56.0, 63.0]],
        dtype=np.float32,
    )
    workspace = Workspace(
        primary=main.copy(),
        metadata={"primary_landmarks5": landmarks, "primary_bbox": (18, 14, 60, 68)},
    )

    result = runtime.run(workspace, immutable_main=main)

    assert constructed == []
    assert result.details["models_actually_executed"] == []
    assert np.array_equal(result.image, main)
