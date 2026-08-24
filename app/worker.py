from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.automatic import AutomaticPipelineRunner
from app.activity import RestorationActivityLock
from app.execution import Workspace
from app.hardware import apply_hardware_policy, detect_hardware_policy, detect_hardware_profile
from app.model_registry import sha256_path
from app.paths import user_data_root
from app.production_models import resolve_local_production_models
from app.progress_timeline import (
    BlockTimingHistory,
    ProcessResourceSampler,
    ProgressTimelineTracker,
)
from app.settings import load_runtime_settings


class PipelineWorker(QObject):
    """Esegue verifica dei modelli locali e pipeline fuori dal thread UI."""

    progress = Signal(int, str)
    progress_detail = Signal(object)
    block_completed = Signal(int, str, str, object, object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, workspace: Workspace, output: Path, upscale: int = 1) -> None:
        super().__init__()
        self.workspace = workspace
        self.output = Path(output)
        self.upscale = int(upscale)
        self._history = BlockTimingHistory(
            user_data_root() / "telemetry" / "block-timings.json"
        )
        self._timeline = ProgressTimelineTracker(self._history)
        self._resources = ProcessResourceSampler()
        self._verified_models: dict[str, dict[str, str]] = {}
        self._active_block_index: int | None = None

    def _emit_detail(self, payload: dict[str, object]) -> None:
        enriched = dict(payload)
        enriched.update(self._resources.sample())
        self.progress_detail.emit(enriched)

    def _register_verified_models(self, paths: dict[str, Path]) -> None:
        verified: dict[str, dict[str, str]] = {}
        for key, path in paths.items():
            candidate = Path(path)
            if not candidate.is_file():
                continue
            verified[str(key)] = {
                "model_key": str(key),
                "checkpoint_path": str(candidate),
                "checkpoint_sha256": sha256_path(candidate),
            }
        self._verified_models = verified
        self.workspace.metadata["verified_runtime_models"] = dict(verified)

    @staticmethod
    def _actual_model_keys(block_index: int, details: dict[str, object]) -> tuple[str, ...]:
        index = int(block_index)
        keys: list[str] = []
        if index == 2 and details.get("pretrained") is True:
            keys.append("opencv_nafnet_deblur")
        elif index == 4 and details.get("pretrained") is True:
            keys.append("opencv_yunet")
            backend = str(details.get("backend", "")).lower()
            if "sface" in backend:
                keys.append("opencv_sface")
        elif index == 6 and details.get("face_parsing_pretrained") is True:
            keys.append("face_parsing_resnet18_onnx")
        elif index == 8 and int(details.get("generated_pixels", 0) or 0) > 0:
            keys.append("opencv_lama_inpaint")
        elif index == 10 and details.get("head_pose_pretrained") is True:
            keys.append("head_pose_mobilenetv2_onnx")
        elif index == 11 and details.get("pretrained") is True:
            keys.append("opencv_sface")
        return tuple(keys)

    def _runtime_evidence(
        self,
        block_index: int,
        details: dict[str, object],
    ) -> dict[str, object]:
        keys = self._actual_model_keys(block_index, details)
        models = [self._verified_models[key] for key in keys if key in self._verified_models]
        engine = details.get("engine") or details.get("backend")
        if engine is None and details.get("face_parsing_pretrained") is True:
            engine = "resnet18-celebamaskhq-onnx"
        return {
            "engine": None if engine is None else str(engine),
            "model_keys": [item["model_key"] for item in models],
            "checkpoint_paths": [item["checkpoint_path"] for item in models],
            "checkpoint_sha256": [item["checkpoint_sha256"] for item in models],
        }

    def _runner_progress(self, index: int, name: str) -> None:
        progress_index = int(index)
        text = str(name)
        self.progress.emit(progress_index, text)
        if not text.startswith("Avvio:"):
            return
        block_index = max(1, min(13, progress_index + 1))
        self._active_block_index = block_index
        payload = self._timeline.start(block_index).to_dict()
        payload["message"] = text
        payload.update(
            {
                "engine": None,
                "model_keys": [],
                "checkpoint_paths": [],
                "checkpoint_sha256": [],
            }
        )
        self._emit_detail(payload)

    def _runner_block_completed(
        self,
        index: int,
        title: str,
        status: str,
        image: object,
        details: object,
    ) -> None:
        detail_map = dict(details) if isinstance(details, dict) else {}
        self.block_completed.emit(index, title, status, image, detail_map)
        payload = self._timeline.complete(int(index), status=str(status)).to_dict()
        payload["message"] = str(title)
        payload.update(self._runtime_evidence(int(index), detail_map))
        self._active_block_index = None
        self._emit_detail(payload)

    @Slot()
    def run(self) -> None:
        try:
            with RestorationActivityLock():
                settings = load_runtime_settings()
                policy = detect_hardware_policy(settings.hardware_mode)
                apply_hardware_policy(policy)
                self.workspace.metadata["hardware_policy"] = policy.to_dict()
                self.progress.emit(
                    0,
                    f"Hardware bilanciato: {policy.cv_threads} thread CPU, DNN {policy.dnn_target}, un modello alla volta",
                )

                self.progress.emit(0, "Verifica model pack production locale")
                bootstrap = resolve_local_production_models()
                if not (bootstrap.face_ready and bootstrap.standard_ready and bootstrap.inpaint_ready):
                    missing = ", ".join(sorted(bootstrap.errors))
                    raise RuntimeError(
                        f"Model pack offline incompleto o corrotto: {missing}. Usa Aggiornamenti per ripararlo."
                    )
                self.workspace.metadata["core_model_paths"] = {
                    key: str(path) for key, path in bootstrap.paths.items()
                }
                self._register_verified_models(dict(bootstrap.paths))
                self.workspace.metadata["core_model_errors"] = dict(bootstrap.errors)
                self.workspace.metadata["core_models_ready"] = bootstrap.face_ready
                self.workspace.metadata["pretrained_deblur_ready"] = bootstrap.deblur_ready
                self.workspace.metadata["pretrained_semantic_ready"] = bootstrap.semantic_ready
                self.workspace.metadata["pretrained_pose_ready"] = bootstrap.pose_ready
                self.workspace.metadata["pretrained_lama_ready"] = bootstrap.inpaint_ready
                self.workspace.metadata["pretrained_standard_ready"] = bootstrap.standard_ready

                profile = detect_hardware_profile(
                    dnn_model_path=bootstrap.paths.get("opencv_yunet"),
                    disk_path=self.output.parent,
                )
                if policy.opencl_enabled and not profile.opencl_functional:
                    policy = replace(policy, opencl_enabled=False, dnn_target="cpu")
                    apply_hardware_policy(policy)
                    self.workspace.metadata["hardware_policy"] = policy.to_dict()
                self.workspace.metadata["hardware_profile"] = profile.to_dict()
                acceleration = "Accelerazione disponibile" if profile.acceleration_available else "Modalità CPU sicura"
                self.progress.emit(0, f"{acceleration}: {profile.profile_class}")

                runner = AutomaticPipelineRunner(self.workspace)
                runner.on_progress = self._runner_progress
                runner.on_block_completed = self._runner_block_completed
                result = runner.run(self.output, upscale=self.upscale)
                self._history.save()
                self.completed.emit(result)
        except Exception as exc:
            if self._active_block_index is not None:
                event = self._timeline.heartbeat()
                if event is not None:
                    payload = event.to_dict()
                    payload.update({"phase": "error", "status": "ERROR", "message": str(exc)})
                    self._emit_detail(payload)
            try:
                self._history.save()
            except OSError:
                pass
            self.failed.emit(str(exc))
