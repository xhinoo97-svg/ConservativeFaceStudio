from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow
from .pipeline import default_pipeline, validate_pipeline


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Conservative Face Studio")
    window = MainWindow()
    if "--smoke-test" in sys.argv:
        validate_pipeline(default_pipeline())
        window.close()
        return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
