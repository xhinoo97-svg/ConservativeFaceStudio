from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.automatic import AutomaticPipelineRunner
from app.activity import RestorationActivityLock
from app.execution import Workspace
from app.hardware import apply_hardware_policy, detect_hardware_policy, detect_hardware_profile
from app.production_models import resolve_local_production_models
from app.settings import load_runtime_settings


class PipelineWorker(QObject):
    """Esegue verifica dei modelli locali e pipeline fuori dal thread UI."""

    progress = Signal(int, str)
    block_completed = Signal(int, str, str, object, object)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, workspace: Workspace, output: Path, upscale: int = 1) -> None:
        super().__init__()
        self.workspace = workspace
        self.output = Path(output)
        self.upscale = int(upscale)

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
                runner.on_progress = lambda index, name: self.progress.emit(int(index), str(name))
                runner.on_block_completed = lambda index, title, status, image, details: self.block_completed.emit(index, title, status, image, details)
                result = runner.run(self.output, upscale=self.upscale)
                self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
