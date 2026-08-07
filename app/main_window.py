from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from .automatic import AutomaticPipelineRunner, AutomaticRunResult
from .execution import Workspace


class ImagePanel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 480)
        self.setStyleSheet("border: 1px solid #777; background: #202020; color: white;")
        self._image: np.ndarray | None = None

    def set_cv_image(self, image: np.ndarray | None) -> None:
        self._image = None if image is None else image.copy()
        self._refresh()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        rgb = cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)


class MainWindow(QMainWindow):
    """Interfaccia automatica: carica foto, avvia pipeline, scarica risultati."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Conservative Face Studio — Automatic Strict Mode")
        self.resize(1120, 760)

        self.primary: np.ndarray | None = None
        self.references: list[np.ndarray] = []
        self.source_paths: list[Path] = []
        self.run_result: AutomaticRunResult | None = None
        self.run_directory: Path | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        self.status = QLabel("1. Carica una o più foto. La prima sarà la foto principale.")
        self.status.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(self.status)

        panels = QHBoxLayout()
        self.before_panel = ImagePanel("Foto principale")
        self.after_panel = ImagePanel("Risultato finale")
        panels.addWidget(self.before_panel)
        panels.addWidget(self.after_panel)
        layout.addLayout(panels)

        self.progress = QProgressBar()
        self.progress.setRange(0, 13)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        controls = QHBoxLayout()
        self.load_button = QPushButton("Carica foto")
        self.start_button = QPushButton("Inizia")
        self.download_button = QPushButton("Scarica risultati ZIP")
        controls.addWidget(self.load_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.download_button)
        layout.addLayout(controls)

        self.load_button.clicked.connect(self.load_images)
        self.start_button.clicked.connect(self.start_pipeline)
        self.download_button.clicked.connect(self.download_results)
        self._update_controls()

    def load_images(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Seleziona foto principale e riferimenti",
            "",
            "Immagini (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not filenames:
            return
        images: list[np.ndarray] = []
        for filename in filenames:
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            if image is None:
                QMessageBox.critical(self, "Errore", f"Impossibile leggere:\n{filename}")
                return
            images.append(image)

        primary_shape = images[0].shape[:2]
        normalized = [images[0]]
        for image in images[1:]:
            if image.shape[:2] != primary_shape:
                image = cv2.resize(image, (primary_shape[1], primary_shape[0]), interpolation=cv2.INTER_AREA)
            normalized.append(image)

        self.primary = normalized[0]
        self.references = normalized[1:]
        self.source_paths = [Path(item) for item in filenames]
        self.run_result = None
        self.before_panel.set_cv_image(self.primary)
        self.after_panel.clear()
        self.after_panel.setText("Risultato finale")
        self.progress.setValue(0)
        self.status.setText(f"Caricate {len(normalized)} foto. Premi Inizia.")
        self._update_controls()

    def start_pipeline(self) -> None:
        if self.primary is None:
            return
        self.load_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.download_button.setEnabled(False)
        self.progress.setRange(0, 0)
        self.status.setText("Pipeline automatica in esecuzione…")

        try:
            self.run_directory = Path(tempfile.mkdtemp(prefix="ConservativeFaceStudio-"))
            stem = self.source_paths[0].stem if self.source_paths else "restauro"
            output = self.run_directory / f"{stem}_finale.png"
            runner = AutomaticPipelineRunner(
                Workspace(primary=self.primary.copy(), references=[item.copy() for item in self.references])
            )
            self.run_result = runner.run(output)
            final = cv2.imread(str(self.run_result.final_image), cv2.IMREAD_COLOR)
            if final is None:
                raise RuntimeError("Il risultato finale non è leggibile")
            self.after_panel.set_cv_image(final)
            self.progress.setRange(0, 13)
            self.progress.setValue(13)
            self.status.setText("Elaborazione completata. Premi Scarica risultati ZIP.")
        except Exception as exc:  # noqa: BLE001
            self.progress.setRange(0, 13)
            self.progress.setValue(0)
            self.status.setText("Elaborazione non completata")
            QMessageBox.critical(self, "Errore pipeline", str(exc))
        finally:
            self._update_controls()

    def download_results(self) -> None:
        if self.run_result is None:
            return
        suggested = self.run_result.blocks_zip.name
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Salva ZIP completo",
            suggested,
            "Archivio ZIP (*.zip)",
        )
        if not filename:
            return
        destination = Path(filename)
        if destination.suffix.lower() != ".zip":
            destination = destination.with_suffix(".zip")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.run_result.blocks_zip, destination)
        QMessageBox.information(self, "Download completato", f"Archivio salvato in:\n{destination}")

    def _update_controls(self) -> None:
        has_images = self.primary is not None
        self.load_button.setEnabled(True)
        self.start_button.setEnabled(has_images)
        self.download_button.setEnabled(self.run_result is not None)
