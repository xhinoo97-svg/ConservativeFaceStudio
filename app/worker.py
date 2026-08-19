from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time

from PySide6.QtCore import QObject, Signal, Slot

from app.automatic import AutomaticPipelineRunner
from app.activity import RestorationActivityLock
from app.execution import Workspace
from app.hardware import apply_hardware_policy, detect_hardware_policy, detect_hardware_profile
from app.paths import user_data_root
from app.production_models import resolve_local_production_models
from app.progress_timeline import BlockTimingHistory, ProgressTimelineTracker
from app.settings import load_runtime_settings


class PipelineWorker(QObject):
    """Esegue verifica dei modelli locali e pipeline fuori dal thread UI."""

    progress = Signal(int, str)
    progress_detail = Signal(object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, workspace: Workspace, output: Path, upscale: int = 1) -> None:
        super().__init__()
        self.workspace = workspace
        self.output = Path(output)
        self.upscale = int(upscale)
        self._history = BlockTimingHistory(user_data_root() / "telemetry" / "block-timings.json")
        self._timeline = ProgressTimelineTracker(self._history)

    def _emit_detail(self, payload: dict[str, object]) -> None:
        self.progress_detail.emit(dict(payload))

    def _runtime_model_role(self, block_index: int) -> str | None:
        """Expose a concrete selected model when runtime metadata has one.

        Missing metadata deliberately falls back to the generic block role in
        ProgressTimelineTracker; the UI must not pretend that an optional research
        model is active before the selector actually chose it.
        """
        keys = {
            2: ("selected_deblur_model", "paper_quality_deblur_model"),
            3: ("selected_enhance_model", "paper_quality_jpeg_model"),
            6: ("selected_damage_model",),
            8: ("selected_inpaint_model", "paper_quality_repair_model"),
            12: ("selected_upscale_model",),
        }
        for key in keys.get(int(block_index), ()):
            value = self.workspace.metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        if block_index in {4, 11}:
            return "OpenCV Zoo YuNet + SFace"
        return None

    def _runner_progress(self, index: int, name: str) -> None:
        progress_index = int(index)
        text = str(name)
        self.progress.emit(progress_index, text)
        if text.startswith("Avvio:"):
            block_index = max(1, min(13, progress_index + 1))
            event = self._timeline.start(
                block_index,
                model_role=self._runtime_model_role(block_index),
            )
            payload = event.to_dict()
            payload["message"] = text
            self._emit_detail(payload)
            return
        if progress_index <= 0:
            return
        block_index = max(1, min(13, progress_index))
        status = "PASS"
        lowered = text.lower()
        if "saltato" in lowered:
            status = "SKIPPED"
        elif "rollback" in lowered:
            status = "ROLLBACK"
        event = self._timeline.complete(
            block_index,
            status=status,
            model_role=self._runtime_model_role(block_index),
        )
        payload = event.to_dict()
        payload["message"] = text
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
                self._emit_detail(
                    {
                        "phase": "prepare",
                        "block_index": 0,
                        "block_total": 13,
                        "block_title": "Preparazione",
                        "model_role": "Hardware policy",
                        "elapsed_block_seconds": 0.0,
                        "estimated_block_seconds": None,
                        "estimated_remaining_seconds": None,
                        "overall_percent": 0.0,
                        "status": "RUNNING",
                        "message": f"{policy.cv_threads} thread CPU; DNN {policy.dnn_target}; un modello pesante alla volta",
                        "cv_threads": int(policy.cv_threads),
                    }
                )

                self.progress.emit(0, "Verifica model pack production locale")
                model_pack_started = time.monotonic()
                self._emit_detail(
                    {
                        "phase": "prepare",
                        "block_index": 0,
                        "block_total": 13,
                        "block_title": "Preparazione",
                        "model_role": "Verified local model pack",
                        "elapsed_block_seconds": 0.0,
                        "estimated_block_seconds": None,
                        "estimated_remaining_seconds": None,
                        "overall_percent": 0.0,
                        "status": "RUNNING",
                        "message": "Verifica modelli e SHA-256 locali",
                    }
                )
                bootstrap = resolve_local_production_models()
                if not (bootstrap.face_ready and bootstrap.standard_ready and bootstrap.inpaint_ready):
                    missing = ", ".join(sorted(bootstrap.errors))
                    raise RuntimeError(
                        f"Model pack offline incompleto o corrotto: {missing}. Usa Aggiornamenti per ripararlo."
                    )
                model_pack_seconds = max(0.0, time.monotonic() - model_pack_started)
                self._emit_detail(
                    {
                        "phase": "prepare_complete",
                        "block_index": 0,
                        "block_total": 13,
                        "block_title": "Preparazione",
                        "model_role": "Verified local model pack",
                        "elapsed_block_seconds": model_pack_seconds,
                        "estimated_block_seconds": model_pack_seconds,
                        "estimated_remaining_seconds": None,
                        "overall_percent": 0.0,
                        "status": "PASS",
                        "message": "Model pack verificato",
                    }
                )
                self.workspace.metadata["core_model_paths"] = {
                    key: str(path) for key, path in bootstrap.paths.items()
                }
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
                self._emit_detail(
                    {
                        "phase": "prepare_complete",
                        "block_index": 0,
                        "block_total": 13,
                        "block_title": "Preparazione",
                        "model_role": "Hardware profile",
                        "elapsed_block_seconds": 0.0,
                        "estimated_block_seconds": None,
                        "estimated_remaining_seconds": None,
                        "overall_percent": 0.0,
                        "status": "PASS",
                        "message": f"{acceleration}: {profile.profile_class}",
                        "logical_processors": int(profile.logical_processors),
                        "total_ram_bytes": profile.total_ram_bytes,
                    }
                )

                runner = AutomaticPipelineRunner(self.workspace)
                runner.on_progress = self._runner_progress
                result = runner.run(self.output, upscale=self.upscale)
                self._history.save()
                self.completed.emit(result)
        except Exception as exc:
            try:
                self._history.save()
            except OSError:
                pass
            self.failed.emit(str(exc))
