$ErrorActionPreference = 'Stop'

python -m PyInstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name ConservativeFaceStudio `
  --hidden-import cv2 `
  --hidden-import onnxruntime `
  --hidden-import app.model_registry `
  --hidden-import app.standard_pretrained `
  --hidden-import app.opencv_nafnet `
  --hidden-import app.opencv_semantic_models `
  --hidden-import app.pretrained_semantic_handlers `
  --hidden-import app.opencv_lama `
  --hidden-import app.reference_inpainting `
  --hidden-import app.pretrained_inpaint_handler `
  --hidden-import app.update_manager `
  --hidden-import app.update_worker `
  --hidden-import app.production_model_smoke `
  app/__main__.py

$package = 'dist/ConservativeFaceStudio'
foreach ($directory in @('models','config','licenses','runtime','logs','projects','exports','cache')) {
  New-Item -ItemType Directory -Force -Path "$package/$directory" | Out-Null
}
foreach ($modelDirectory in @('detection','identity','landmarks','parsing','pose','deblur','reference','inpainting','restoration','optional')) {
  New-Item -ItemType Directory -Force -Path "$package/models/$modelDirectory" | Out-Null
  "Managed model category: $modelDirectory. Only manifest-verified files may be activated." | Set-Content "$package/models/$modelDirectory/README.txt" -Encoding utf8
}
foreach ($directory in @('runtime','logs','projects','exports','cache')) {
  "Conservative Face Studio managed $directory directory." | Set-Content "$package/$directory/README.txt" -Encoding utf8
}

if (!(Test-Path 'models/model-registry.json')) {
  python -c "from app.model_catalog import all_model_manifests; from app.model_registry import export_registry; export_registry('models/model-registry.json', all_model_manifests())"
}

# Keep the PyInstaller step independent from large checkpoint downloads. Production
# models are checksum-verified and staged explicitly by stage_production_models.ps1
# after the executable exists. This avoids duplicating optional/research checkpoints
# and keeps peak disk usage low on GitHub-hosted Windows runners.
Copy-Item 'models/model-registry.json' "$package/models/model-registry.json" -Force
if (!(Test-Path 'models/model-manifests.json')) {
  throw 'Model manifest catalog missing before packaging'
}
Copy-Item 'models/model-manifests.json' "$package/models/model-manifests.json" -Force
Copy-Item 'config/*' "$package/config" -Recurse -Force
Copy-Item 'licenses/*' "$package/licenses" -Recurse -Force
Copy-Item 'THIRD_PARTY_MODULES.md' "$package/THIRD_PARTY_MODULES.md" -Force
Copy-Item 'THIRD_PARTY_MODULES.md' "$package/licenses/THIRD_PARTY_MODULES.md" -Force
Copy-Item 'README.md' "$package/README.md" -Force
