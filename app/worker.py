from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.automatic import AutomaticPipelineRunner
from app.core_models import ensure_core_pretrained_models
from app.execution import Workspace
from app.hardware import apply_hardware_policy, detect_hardware_policy


class PipelineWorker(QObject):
    """Esegue download verificati e pipeline fuori dal thread UI."""

    progress = Signal(int, str)
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
            policy = detect_hardware_policy("balanced")
            apply_hardware_policy(policy)
            self.workspace.metadata["hardware_policy"] = policy.to_dict()
            self.progress.emit(
                0,
                f"Hardware bilanciato: {policy.cv_threads} thread CPU, DNN {policy.dnn_target}, un modello alla volta",
            )

            bootstrap = ensure_core_pretrained_models(timeout_seconds=15)
            self.workspace.metadata["core_model_paths"] = {
                key: str(path) for key, path in bootstrap.paths.items()
            }
            self.workspace.metadata["core_model_errors"] = dict(bootstrap.errors)
            self.workspace.metadata["core_models_ready"] = bootstrap.ready

            runner = AutomaticPipelineRunner(self.workspace)
            runner.on_progress = lambda index, name: self.progress.emit(int(index), str(name))
            result = runner.run(self.output, upscale=self.upscale)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
