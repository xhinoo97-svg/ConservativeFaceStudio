$ErrorActionPreference = 'Stop'

python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name ConservativeFaceStudio `
  --hidden-import cv2 `
  --hidden-import app.model_registry `
  --hidden-import app.standard_pretrained `
  --hidden-import app.opencv_nafnet `
  --hidden-import app.opencv_semantic_models `
  --hidden-import app.pretrained_semantic_handlers `
  --hidden-import app.opencv_lama `
  --hidden-import app.reference_inpainting `
  --hidden-import app.pretrained_inpaint_handler `
  app/__main__.py

$package = 'dist/ConservativeFaceStudio'
New-Item -ItemType Directory -Force -Path "$package/models" | Out-Null
New-Item -ItemType Directory -Force -Path "$package/projects" | Out-Null
New-Item -ItemType Directory -Force -Path "$package/exports" | Out-Null

if (!(Test-Path 'models/model-registry.json')) {
  python -c "from app.model_catalog import all_model_manifests; from app.model_registry import export_registry; export_registry('models/model-registry.json', all_model_manifests())"
}

# Ship the verified production checkpoints with the portable package/installer.
# The application still keeps a writable per-user cache for future replacements,
# but a first run must not depend on network availability.
Copy-Item 'models/*' "$package/models" -Recurse -Force
Copy-Item 'THIRD_PARTY_MODULES.md' "$package/THIRD_PARTY_MODULES.md" -Force
Copy-Item 'README.md' "$package/README.md" -Force
