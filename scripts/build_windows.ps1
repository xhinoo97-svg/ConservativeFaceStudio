$ErrorActionPreference = 'Stop'

python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name ConservativeFaceStudio `
  --collect-all PySide6 `
  --hidden-import cv2 `
  app/__main__.py

$package = 'dist/ConservativeFaceStudio'
New-Item -ItemType Directory -Force -Path "$package/models" | Out-Null
New-Item -ItemType Directory -Force -Path "$package/projects" | Out-Null
New-Item -ItemType Directory -Force -Path "$package/exports" | Out-Null
Copy-Item 'models/README.md' "$package/models/README.md" -Force
Copy-Item 'THIRD_PARTY_MODULES.md' "$package/THIRD_PARTY_MODULES.md" -Force
Copy-Item 'README.md' "$package/README.md" -Force
