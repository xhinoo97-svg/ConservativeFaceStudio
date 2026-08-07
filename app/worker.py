from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from app.automatic import AutomaticPipelineRunner
from app.execution import Workspace


class PipelineWorker(QObject):
    """Esegue la pipeline fuori dal thread UI per evitare finestre bloccate."""

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
            runner = AutomaticPipelineRunner(self.workspace)
            runner.on_progress = lambda index, name: self.progress.emit(int(index), str(name))
            result = runner.run(self.output, upscale=self.upscale)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
