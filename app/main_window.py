from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt
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

from .automatic import AutomaticRunResult
from .execution import Workspace
from .imaging import fit_to_canvas
from .reference_limits import MAX_PROJECT_IMAGES, MAX_REFERENCE_IMAGES, validate_reference_count
from .worker import PipelineWorker


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
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(pixmap)


class MainWindow(QMainWindow):
    """Automatic UI with photo #1 fixed as MAIN IMAGE and up to nine references."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Conservative Face Studio — Automatic Strict Mode")
        self.resize(1120, 790)

        self.primary: np.ndarray | None = None
        self.references: list[np.ndarray] = []
        self.reference_normalization: list[dict[str, float | int]] = []
        self.primary_path: Path | None = None
        self.reference_paths: list[Path] = []
        self.run_result: AutomaticRunResult | None = None
        self.run_directory: Path | None = None
        self.worker_thread: QThread | None = None
        self.worker: PipelineWorker | None = None

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        self.status = QLabel(
            f"1. Carica la MAIN IMAGE. 2. Aggiungi fino a {MAX_REFERENCE_IMAGES} reference."
        )
        self.status.setStyleSheet("font-size: 16px; font-weight: 600;")
        layout.addWidget(self.status)

        self.confidence_label = QLabel("Original Information Confidence: —")
        self.confidence_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        layout.addWidget(self.confidence_label)

        panels = QHBoxLayout()
        self.before_panel = ImagePanel("MAIN IMAGE")
        self.after_panel = ImagePanel("Risultato finale")
        panels.addWidget(self.before_panel)
        panels.addWidget(self.after_panel)
        layout.addLayout(panels)

        self.progress = QProgressBar()
        self.progress.setRange(0, 13)
        layout.addWidget(self.progress)

        controls = QHBoxLayout()
        self.load_primary_button = QPushButton("Carica MAIN IMAGE")
        self.load_references_button = QPushButton(f"Aggiungi reference (max {MAX_REFERENCE_IMAGES})")
        self.clear_references_button = QPushButton("Svuota reference")
        self.start_button = QPushButton("Inizia")
        self.download_button = QPushButton("Scarica risultati ZIP")
        controls.addWidget(self.load_primary_button)
        controls.addWidget(self.load_references_button)
        controls.addWidget(self.clear_references_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.download_button)
        layout.addLayout(controls)

        self.load_primary_button.clicked.connect(self.load_primary)
        self.load_references_button.clicked.connect(self.load_references)
        self.clear_references_button.clicked.connect(self.clear_references)
        self.start_button.clicked.connect(self.start_pipeline)
        self.download_button.clicked.connect(self.download_results)
        self._update_controls()

    @staticmethod
    def _read_image(filename: str) -> np.ndarray | None:
        return cv2.imread(filename, cv2.IMREAD_COLOR)

    def load_primary(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Seleziona la MAIN IMAGE", "", "Immagini (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not filename:
            return
        image = self._read_image(filename)
        if image is None:
            QMessageBox.critical(self, "Errore", f"Impossibile leggere:\n{filename}")
            return

        self.primary = image
        self.primary_path = Path(filename)
        self.references = []
        self.reference_paths = []
        self.reference_normalization = []
        self.run_result = None
        self.before_panel.set_cv_image(self.primary)
        self.after_panel.clear()
        self.after_panel.setText("Risultato finale")
        self.progress.setValue(0)
        self.confidence_label.setText("Original Information Confidence: —")
        self.status.setText("MAIN IMAGE caricata. La foto #1 resterà il canvas finale. Aggiungi da 0 a 9 reference.")
        self._update_controls()

    def load_references(self) -> None:
        if self.primary is None:
            QMessageBox.information(self, "MAIN IMAGE mancante", "Carica prima la MAIN IMAGE.")
            return
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            f"Seleziona fino a {MAX_REFERENCE_IMAGES} fotografie di riferimento",
            "",
            "Immagini (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not filenames:
            return
        remaining = MAX_REFERENCE_IMAGES - len(self.references)
        if len(filenames) > remaining:
            QMessageBox.warning(
                self,
                "Troppe reference",
                f"Puoi aggiungere ancora {remaining} reference. Il progetto supporta {MAX_PROJECT_IMAGES} immagini totali: 1 MAIN IMAGE + {MAX_REFERENCE_IMAGES} reference.",
            )
            return

        primary_shape = self.primary.shape[:2]
        fitted_items: list[np.ndarray] = []
        metadata_items: list[dict[str, float | int]] = []
        paths: list[Path] = []
        for filename in filenames:
            image = self._read_image(filename)
            if image is None:
                QMessageBox.critical(self, "Errore", f"Impossibile leggere:\n{filename}")
                return
            fitted, metadata = fit_to_canvas(image, primary_shape)
            fitted_items.append(fitted)
            metadata_items.append(metadata)
            paths.append(Path(filename))

        self.references.extend(fitted_items)
        self.reference_normalization.extend(metadata_items)
        self.reference_paths.extend(paths)
        validate_reference_count(len(self.references))
        self.run_result = None
        self.status.setText(
            f"Caricate 1 MAIN IMAGE + {len(self.references)} reference. Puoi iniziare o aggiungerne altre fino a {MAX_REFERENCE_IMAGES}."
        )
        self._update_controls()

    def clear_references(self) -> None:
        if self.worker_thread is not None:
            return
        self.references = []
        self.reference_paths = []
        self.reference_normalization = []
        self.run_result = None
        self.confidence_label.setText("Original Information Confidence: —")
        self.status.setText("Reference rimosse. La MAIN IMAGE resta invariata.")
        self._update_controls()

    def start_pipeline(self) -> None:
        if self.primary is None or self.worker_thread is not None:
            return
        validate_reference_count(len(self.references))
        self.run_result = None
        self.run_directory = Path(tempfile.mkdtemp(prefix="ConservativeFaceStudio-"))
        output = self.run_directory / "final_restored_main.png"
        workspace = Workspace(
            primary=self.primary.copy(),
            references=[item.copy() for item in self.references],
            metadata={
                "reference_normalization": list(self.reference_normalization),
                "user_selected_primary": True,
                "primary_priority_policy": "fixed-photo-1-main-image",
                "primary_source_path": str(self.primary_path) if self.primary_path is not None else None,
                "reference_source_paths": [str(item) for item in self.reference_paths],
            },
        )

        thread = QThread(self)
        worker = PipelineWorker(workspace, output, upscale=2)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self.worker_thread = thread
        self.worker = worker
        self.progress.setRange(0, 13)
        self.progress.setValue(0)
        self.confidence_label.setText("Original Information Confidence: calcolo in corso")
        self.status.setText("Pipeline automatica in esecuzione")
        self._update_controls()
        thread.start()

    def _on_progress(self, index: int, name: str) -> None:
        self.progress.setValue(max(0, min(13, int(index))))
        self.status.setText(name)

    def _on_completed(self, result: object) -> None:
        if not isinstance(result, AutomaticRunResult):
            self._on_failed("Risultato pipeline non valido")
            return
        self.run_result = result
        final = cv2.imread(str(result.final_image), cv2.IMREAD_COLOR)
        if final is None:
            self._on_failed("Il risultato finale non è leggibile")
            return
        self.after_panel.set_cv_image(final)
        self.progress.setValue(13)

        report = None
        for item in reversed(result.results):
            candidate = item.details.get("evidence_confidence")
            if isinstance(candidate, dict):
                report = candidate
                break
        if report is None:
            self.confidence_label.setText("Original Information Confidence: non disponibile")
        else:
            evidence = float(report.get("evidence_confidence", 0.0))
            generated = 100.0 * float(report.get("generated_fraction", 0.0))
            symmetry = 100.0 * float(report.get("symmetry_fraction", 0.0))
            unresolved = 100.0 * float(report.get("unresolved_fraction", 0.0))
            self.confidence_label.setText(
                f"Original Information Confidence: {evidence:.1f}%   |   Generated: {generated:.1f}%   |   Symmetry: {symmetry:.1f}%   |   Unresolved: {unresolved:.1f}%"
            )
        self.status.setText("Elaborazione completata. Premi Scarica risultati ZIP.")
        self._update_controls()

    def _on_failed(self, message: str) -> None:
        self.progress.setValue(0)
        self.status.setText("Elaborazione non completata")
        self.confidence_label.setText("Original Information Confidence: non disponibile")
        QMessageBox.critical(self, "Errore pipeline", str(message))
        self._update_controls()

    def _on_thread_finished(self) -> None:
        self.worker_thread = None
        self.worker = None
        self._update_controls()

    def download_results(self) -> None:
        if self.run_result is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Salva ZIP completo", self.run_result.blocks_zip.name, "Archivio ZIP (*.zip)"
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
        busy = self.worker_thread is not None
        self.load_primary_button.setEnabled(not busy)
        self.load_references_button.setEnabled(self.primary is not None and not busy and len(self.references) < MAX_REFERENCE_IMAGES)
        self.clear_references_button.setEnabled(bool(self.references) and not busy)
        self.start_button.setEnabled(self.primary is not None and not busy)
        self.download_button.setEnabled(self.run_result is not None and not busy)
