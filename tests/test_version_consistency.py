from __future__ import annotations

import re
from pathlib import Path

from app.version import APP_VERSION


def test_application_installer_and_update_version_are_identical() -> None:
    installer = Path("installer/ConservativeFaceStudio.iss").read_text(encoding="utf-8")
    match = re.search(r'^#define AppVersion "([^"]+)"$', installer, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == APP_VERSION
    workflow = Path(".github/workflows/windows-build.yml").read_text(encoding="utf-8")
    assert f"/releases/download/v{APP_VERSION}" in workflow
