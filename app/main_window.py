from __future__ import annotations

import json
import logging
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
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
from .activity import is_restoration_active
from .execution import Workspace
from .hardware import detect_hardware_profile
from .imaging import fit_to_canvas
from .paths import model_search_roots, user_data_root
from .project import ProjectDocument, load_project, save_project
from .reference_limits import MAX_PROJECT_IMAGES, MAX_REFERENCE_IMAGES, validate_reference_count
from .settings import load_runtime_settings
from .worker import PipelineWorker
from .update_manager import AppUpdater
from .update_worker import UpdateWorker


LOGGER = logging.getLogger("conservative_face_studio.ui")


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
        self.project_path: Path | None = None
        self.recovery_project_path: Path | None = None
        self.worker_thread: QThread | None = None
        self.worker: PipelineWorker | None = None
        self.update_thread: QThread | None = None
        self.update_worker: UpdateWorker | None = None
        self.update_applying = False
        self.pending_update_apply = False
        self.pending_installer_path: str | None = None

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

        self.hardware_label = QLabel("Hardware: rilevamento in corso")
        layout.addWidget(self.hardware_label)

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
        self.open_project_button = QPushButton("Apri progetto")
        self.save_project_button = QPushButton("Salva progetto")
        self.update_button = QPushButton("Aggiornamenti")
        self.start_button = QPushButton("Inizia")
        self.download_button = QPushButton("Scarica risultati ZIP")
        controls.addWidget(self.load_primary_button)
        controls.addWidget(self.load_references_button)
        controls.addWidget(self.clear_references_button)
        controls.addWidget(self.open_project_button)
        controls.addWidget(self.save_project_button)
        controls.addWidget(self.update_button)
        controls.addWidget(self.start_button)
        controls.addWidget(self.download_button)
        layout.addLayout(controls)

        self.load_primary_button.clicked.connect(self.load_primary)
        self.load_references_button.clicked.connect(self.load_references)
        self.clear_references_button.clicked.connect(self.clear_references)
        self.open_project_button.clicked.connect(self.open_project)
        self.save_project_button.clicked.connect(self.save_current_project)
        self.update_button.clicked.connect(self.check_updates)
        self.start_button.clicked.connect(self.start_pipeline)
        self.download_button.clicked.connect(self.download_results)
        self._refresh_hardware_status()
        self._update_controls()

    @staticmethod
    def _update_manifest_url() -> str:
        return load_runtime_settings().app_update_manifest_url

    def check_updates(self) -> None:
        if self.worker_thread is not None or self.update_thread is not None:
            return
        self._start_update_worker(apply_updates=False)

    def _start_update_worker(self, *, apply_updates: bool) -> None:
        thread = QThread(self)
        worker = UpdateWorker(self._update_manifest_url(), apply_updates=apply_updates)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._on_update_completed)
        worker.failed.connect(self._on_update_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_update_thread_finished)
        self.update_thread = thread
        self.update_worker = worker
        self.update_applying = apply_updates
        self.status.setText("Installazione aggiornamenti verificati" if apply_updates else "Controllo aggiornamenti")
        self._update_controls()
        thread.start()

    def _on_update_completed(self, result: object) -> None:
        if not isinstance(result, dict):
            self._on_update_failed("Risposta updater non valida")
            return
        if not self.update_applying:
            app_available = bool(result.get("app_update_available"))
            models_available = bool(result.get("model_update_available"))
            if not app_available and not models_available:
                QMessageBox.information(self, "Aggiornamenti", "Nessun aggiornamento production completo disponibile.")
                self.status.setText("Nessun aggiornamento disponibile")
                return
            answer = QMessageBox.question(
                self,
                "Aggiornamenti verificati",
                "È disponibile un aggiornamento app e/o model pack. Scaricare, verificare SHA-256, eseguire gli smoke test e attivarlo?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.pending_update_apply = True
                self.status.setText("Preparazione aggiornamento verificato")
            else:
                self.status.setText("Aggiornamento rimandato")
            return

        self.status.setText("Aggiornamenti verificati e attivati")
        app_result = result.get("app_result")
        staged = app_result.get("staged_path") if isinstance(app_result, dict) else None
        if isinstance(staged, str):
            answer = QMessageBox.question(
                self,
                "Installer verificato",
                "L'aggiornamento applicazione è pronto. Chiudere il programma e avviare l'installer verificato?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                self.pending_installer_path = staged
                self.status.setText("Chiusura sicura prima dell'avvio installer")
                return
        QMessageBox.information(self, "Aggiornamenti", "Aggiornamento completato. I modelli verificati saranno usati dalla prossima elaborazione.")

    def _on_update_failed(self, message: str) -> None:
        LOGGER.error("Update failed: %s", message)
        self.status.setText("Aggiornamento non completato; versione precedente conservata")
        QMessageBox.critical(self, "Errore aggiornamento", str(message))

    def _on_update_thread_finished(self) -> None:
        pending_apply = self.pending_update_apply
        pending_installer = self.pending_installer_path
        self.update_thread = None
        self.update_worker = None
        self.update_applying = False
        self.pending_update_apply = False
        self.pending_installer_path = None
        self._update_controls()
        if pending_apply:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, lambda: self._start_update_worker(apply_updates=True))
        elif pending_installer:
            try:
                AppUpdater(
                    user_data_root() / "updates" / "app",
                    restoration_active=is_restoration_active,
                ).launch_installer(pending_installer)
            except Exception as exc:
                QMessageBox.critical(self, "Errore installer", str(exc))
                return
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, QApplication.quit)

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.worker_thread is not None or self.update_thread is not None:
            event.ignore()
            QMessageBox.information(
                self,
                "Operazione in corso",
                "Attendi il completamento della restoration o dell'aggiornamento verificato prima di chiudere.",
            )
            return
        super().closeEvent(event)

    def _refresh_hardware_status(self) -> None:
        yunet = None
        relative = Path("models/detection/face_detection_yunet_2023mar.onnx")
        for root in model_search_roots():
            candidate = root / relative
            if candidate.is_file():
                yunet = candidate
                break
        profile = detect_hardware_profile(dnn_model_path=yunet)
        mode = "Accelerazione disponibile" if profile.acceleration_available else "Modalità CPU sicura"
        memory = "RAM sconosciuta"
        if profile.total_ram_bytes is not None:
            memory = f"RAM {profile.total_ram_bytes / (1024 ** 3):.1f} GB"
        self.hardware_label.setText(
            f"Hardware: {mode} — {profile.profile_class}; {profile.logical_processors} thread; {memory}"
        )

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
        self.project_path = None
        self.before_panel.set_cv_image(self.primary)
        self.after_panel.clear()
        self.after_panel.setText("Risultato finale")
        self.progress.setValue(0)
        self.confidence_label.setText("Original Information Confidence: —")
        self.status.setText("MAIN IMAGE caricata. La foto #1 resterà il canvas finale. Aggiungi da 0 a 9 reference.")
        self._update_controls()

    def _project_document(self, *, status: str, last_checkpoint: str | None = None) -> ProjectDocument:
        sources = []
        if self.primary_path is not None:
            sources.append(str(self.primary_path.resolve()))
        sources.extend(str(path.resolve()) for path in self.reference_paths)
        accepted: list[str] = []
        skipped: list[str] = []
        if self.run_result is not None:
            for result in self.run_result.results:
                (skipped if bool(result.details.get("skipped")) else accepted).append(str(result.block))
        metadata: dict[str, object] = {
            "status": status,
            "main_image_index": 0,
            "reference_count": len(self.reference_paths),
            "resume_policy": "restart_from_immutable_sources_with_verified_checkpoints",
        }
        if last_checkpoint:
            metadata["last_checkpoint"] = last_checkpoint
        if self.run_result is not None:
            metadata["final_image"] = str(self.run_result.final_image)
            metadata["blocks_zip"] = str(self.run_result.blocks_zip)
        return ProjectDocument(
            name=(self.project_path.stem if self.project_path is not None else "Conservative Face Studio project"),
            sources=sources,
            accepted_blocks=accepted,
            skipped_blocks=skipped,
            metadata=metadata,
        )

    def save_current_project(self) -> None:
        if self.primary is None or self.primary_path is None:
            QMessageBox.information(self, "MAIN IMAGE mancante", "Carica prima la MAIN IMAGE.")
            return
        default = str(self.project_path or self.primary_path.with_suffix(".cfs.json"))
        filename, _ = QFileDialog.getSaveFileName(
            self, "Salva progetto", default, "Progetto Conservative Face Studio (*.cfs.json)"
        )
        if not filename:
            return
        destination = Path(filename)
        if not destination.name.lower().endswith(".cfs.json"):
            destination = destination.with_suffix(".cfs.json")
        self.project_path = destination
        status = "completed" if self.run_result is not None else "ready"
        save_project(self._project_document(status=status), destination)
        self.status.setText(f"Progetto salvato: {destination.name}")

    def open_project(self) -> None:
        if self.worker_thread is not None or self.update_thread is not None:
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Apri progetto", "", "Progetto Conservative Face Studio (*.cfs.json *.json)"
        )
        if not filename:
            return
        project_path = Path(filename).resolve()
        try:
            document = load_project(project_path)
            if not document.sources or len(document.sources) > MAX_PROJECT_IMAGES:
                raise ValueError("Il progetto deve contenere 1 MAIN IMAGE e da 0 a 9 reference")
            paths = [Path(item) if Path(item).is_absolute() else project_path.parent / item for item in document.sources]
            images: list[np.ndarray] = []
            for path in paths:
                image = self._read_image(str(path))
                if image is None:
                    raise ValueError(f"Immagine del progetto non leggibile: {path}")
                images.append(image)
            main = images[0]
            fitted: list[np.ndarray] = []
            normalization: list[dict[str, float | int]] = []
            for image in images[1:]:
                item, details = fit_to_canvas(image, main.shape[:2])
                fitted.append(item)
                normalization.append(details)
            validate_reference_count(len(fitted))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "Progetto non valido", str(exc))
            return

        self.primary = main
        self.references = fitted
        self.reference_normalization = normalization
        self.primary_path = paths[0]
        self.reference_paths = paths[1:]
        self.project_path = project_path
        self.run_result = None
        self.before_panel.set_cv_image(main)
        self.after_panel.clear()
        checkpoint = document.metadata.get("last_checkpoint")
        checkpoint_image = self._read_image(str(checkpoint)) if isinstance(checkpoint, str) else None
        if checkpoint_image is not None:
            self.after_panel.set_cv_image(checkpoint_image)
        else:
            self.after_panel.setText("Risultato finale")
        self.progress.setValue(0)
        self.confidence_label.setText("Original Information Confidence: —")
        self.status.setText(
            f"Progetto caricato: 1 MAIN IMAGE + {len(fitted)} reference. Ripresa sicura dalle sorgenti immutabili."
        )
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
        if self.worker_thread is not None or self.update_thread is not None:
            return
        self.references = []
        self.reference_paths = []
        self.reference_normalization = []
        self.run_result = None
        self.confidence_label.setText("Original Information Confidence: —")
        self.status.setText("Reference rimosse. La MAIN IMAGE resta invariata.")
        self._update_controls()

    def start_pipeline(self) -> None:
        if self.primary is None or self.worker_thread is not None or self.update_thread is not None:
            return
        validate_reference_count(len(self.references))
        self.run_result = None
        self.run_directory = Path(tempfile.mkdtemp(prefix="ConservativeFaceStudio-"))
        self.recovery_project_path = self.run_directory / "recovery.cfs.json"
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
                "checkpoint_directory": str(self.run_directory / "checkpoints"),
                "recovery_project_path": str(self.recovery_project_path),
            },
        )
        save_project(self._project_document(status="running"), self.recovery_project_path)

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
        if self.recovery_project_path is not None:
            checkpoint = None
            if self.run_directory is not None:
                manifest = self.run_directory / "checkpoints" / "checkpoint-manifest.json"
                if manifest.is_file():
                    try:
                        payload = json.loads(manifest.read_text(encoding="utf-8"))
                        latest = payload.get("latest")
                        if isinstance(latest, str):
                            checkpoint = str(manifest.parent / latest)
                    except (OSError, ValueError):
                        checkpoint = None
            save_project(self._project_document(status="running", last_checkpoint=checkpoint), self.recovery_project_path)

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
        if self.recovery_project_path is not None:
            save_project(
                self._project_document(status="completed", last_checkpoint=str(result.final_image)),
                self.recovery_project_path,
            )
        self._update_controls()

    def _on_failed(self, message: str) -> None:
        LOGGER.error("Pipeline failed: %s", message)
        self.progress.setValue(0)
        self.status.setText("Elaborazione non completata")
        self.confidence_label.setText("Original Information Confidence: non disponibile")
        if self.recovery_project_path is not None:
            save_project(self._project_document(status="failed"), self.recovery_project_path)
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
        busy = self.worker_thread is not None or self.update_thread is not None
        self.load_primary_button.setEnabled(not busy)
        self.load_references_button.setEnabled(self.primary is not None and not busy and len(self.references) < MAX_REFERENCE_IMAGES)
        self.clear_references_button.setEnabled(bool(self.references) and not busy)
        self.open_project_button.setEnabled(not busy)
        self.save_project_button.setEnabled(self.primary is not None and not busy)
        self.update_button.setEnabled(not busy)
        self.start_button.setEnabled(self.primary is not None and not busy)
        self.download_button.setEnabled(self.run_result is not None and not busy)
