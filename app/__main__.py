from __future__ import annotations

import sys


def main() -> int:
    from app.logging_setup import configure_logging

    configure_logging()
    if "--verify-installation" in sys.argv:
        from app.installation_verifier import report_json, verify_installation

        report = verify_installation()
        print(report_json(report), flush=True)
        return 0 if bool(report.get("ok")) else 2

    if "--offline-test" in sys.argv:
        from app.installation_verifier import offline_inference_test, report_json

        report = offline_inference_test()
        print(report_json(report), flush=True)
        return 0 if bool(report.get("ok")) and bool(report.get("inference_ok")) else 3

    from PySide6.QtWidgets import QApplication
    from app.main_window import MainWindow
    from app.pipeline import default_pipeline, validate_pipeline

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
