from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.pipeline import default_pipeline, validate_pipeline


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Conservative Face Studio")
    validate_pipeline(default_pipeline())
    if "--smoke-test" in sys.argv:
        window = MainWindow()
        window.close()
        return 0
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
