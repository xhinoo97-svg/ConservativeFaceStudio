from __future__ import annotations

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
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .restoration import DeblurSettings, conservative_deblur, quality_enhance


class ImagePanel(QLabel):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 420)
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
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Conservative Face Studio — Block Pipeline V1")
        self.resize(1180, 760)

        self.original: np.ndarray | None = None
        self.current: np.ndarray | None = None
        self.preview: np.ndarray | None = None
        self.source_path: Path | None = None
        self.block_index = 0

        root = QWidget()
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)

        self.status = QLabel("Blocco 0: importa una fotografia principale")
        self.status.setStyleSheet("font-size: 16px; font-weight: 600;")
        main_layout.addWidget(self.status)

        image_layout = QHBoxLayout()
        self.before_panel = ImagePanel("Originale / risultato accettato")
        self.after_panel = ImagePanel("Anteprima del blocco")
        image_layout.addWidget(self.before_panel)
        image_layout.addWidget(self.after_panel)
        main_layout.addLayout(image_layout)

        controls = QHBoxLayout()
        self.open_button = QPushButton("Importa foto")
        self.run_button = QPushButton("Esegui blocco")
        self.accept_button = QPushButton("Accetta")
        self.retry_button = QPushButton("Riprova")
        self.skip_button = QPushButton("Salta")
        self.next_button = QPushButton("Blocco successivo")
        self.export_button = QPushButton("Esporta foto finale")

        for button in (
            self.open_button,
            self.run_button,
            self.accept_button,
            self.retry_button,
            self.skip_button,
            self.next_button,
            self.export_button,
        ):
            controls.addWidget(button)
        main_layout.addLayout(controls)

        sliders = QHBoxLayout()
        sliders.addWidget(QLabel("Intensità"))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 20)
        self.strength_slider.setValue(10)
        sliders.addWidget(self.strength_slider)
        main_layout.addLayout(sliders)

        self.open_button.clicked.connect(self.open_image)
        self.run_button.clicked.connect(self.run_current_block)
        self.accept_button.clicked.connect(self.accept_preview)
        self.retry_button.clicked.connect(self.run_current_block)
        self.skip_button.clicked.connect(self.skip_block)
        self.next_button.clicked.connect(self.next_block)
        self.export_button.clicked.connect(self.export_image)

        self._update_controls()

    def open_image(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Seleziona foto principale",
            "",
            "Immagini (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not filename:
            return
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        if image is None:
            QMessageBox.critical(self, "Errore", "Impossibile leggere l'immagine selezionata.")
            return

        self.source_path = Path(filename)
        self.original = image
        self.current = image.copy()
        self.preview = None
        self.block_index = 1
        self.before_panel.set_cv_image(self.current)
        self.after_panel.clear()
        self.after_panel.setText("Anteprima del blocco")
        self._set_status()
        self._update_controls()

    def run_current_block(self) -> None:
        if self.current is None:
            return
        strength = self.strength_slider.value() / 10.0
        try:
            if self.block_index == 1:
                self.preview = conservative_deblur(
                    self.current,
                    DeblurSettings(
                        denoise=max(1, int(3 + strength * 4)),
                        sharpen=strength,
                        contrast=1.0,
                    ),
                )
            elif self.block_index == 2:
                self.preview = quality_enhance(self.current)
            else:
                self.preview = self.current.copy()
            self.after_panel.set_cv_image(self.preview)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Errore nel blocco", str(exc))
        self._update_controls()

    def accept_preview(self) -> None:
        if self.preview is None:
            return
        self.current = self.preview.copy()
        self.preview = None
        self.before_panel.set_cv_image(self.current)
        self.after_panel.clear()
        self.after_panel.setText("Risultato accettato. Passa al blocco successivo.")
        self._update_controls()

    def skip_block(self) -> None:
        self.preview = None
        self.next_block()

    def next_block(self) -> None:
        if self.current is None:
            return
        self.preview = None
        self.block_index = min(self.block_index + 1, 3)
        self.after_panel.clear()
        self.after_panel.setText("Anteprima del blocco")
        self._set_status()
        self._update_controls()

    def export_image(self) -> None:
        if self.current is None:
            return
        suggested = "restauro_finale.png"
        if self.source_path is not None:
            suggested = f"{self.source_path.stem}_restaurata.png"
        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Esporta foto finale",
            suggested,
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)",
        )
        if not filename:
            return
        suffix = Path(filename).suffix.lower()
        if not suffix:
            filename += ".png" if "PNG" in selected_filter else ".jpg"
        ok = cv2.imwrite(filename, self.current)
        if not ok:
            QMessageBox.critical(self, "Errore", "Non è stato possibile salvare il file.")
            return
        QMessageBox.information(self, "Esportazione completata", f"Foto salvata in:\n{filename}")

    def _set_status(self) -> None:
        labels = {
            0: "Blocco 0: importa una fotografia principale",
            1: "Blocco 1: deblur, denoise e nitidezza conservativa",
            2: "Blocco 2: contrasto locale e recupero qualità",
            3: "Pipeline iniziale completata: esporta la foto finale",
        }
        self.status.setText(labels[self.block_index])

    def _update_controls(self) -> None:
        has_image = self.current is not None
        has_preview = self.preview is not None
        processable = has_image and self.block_index in (1, 2)
        self.run_button.setEnabled(processable)
        self.accept_button.setEnabled(has_preview)
        self.retry_button.setEnabled(processable)
        self.skip_button.setEnabled(processable)
        self.next_button.setEnabled(has_image and not has_preview and self.block_index < 3)
        self.export_button.setEnabled(has_image)
