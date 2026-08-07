$ErrorActionPreference = 'Stop'

python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name ConservativeFaceStudio `
  --hidden-import cv2 `
  --hidden-import app.model_registry `
  app/__main__.py

$package = 'dist/ConservativeFaceStudio'
New-Item -ItemType Directory -Force -Path "$package/models" | Out-Null
New-Item -ItemType Directory -Force -Path "$package/projects" | Out-Null
New-Item -ItemType Directory -Force -Path "$package/exports" | Out-Null

if (!(Test-Path 'models/model-registry.json')) {
  python -c "from app.model_registry import export_registry; export_registry('models/model-registry.json')"
}

Copy-Item 'models/README.md' "$package/models/README.md" -Force
Copy-Item 'models/model-registry.json' "$package/models/model-registry.json" -Force
Copy-Item 'THIRD_PARTY_MODULES.md' "$package/THIRD_PARTY_MODULES.md" -Force
Copy-Item 'README.md' "$package/README.md" -Force
