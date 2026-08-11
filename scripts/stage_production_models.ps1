$ErrorActionPreference = 'Stop'

$package = 'dist/ConservativeFaceStudio'
foreach ($directory in @('detection','identity','landmarks','parsing','pose','deblur','reference','inpainting','restoration','optional')) {
  New-Item -ItemType Directory -Force -Path "$package/models/$directory" | Out-Null
}
$models = @(
  'models/detection/face_detection_yunet_2023mar.onnx',
  'models/identity/face_recognition_sface_2021dec.onnx',
  'models/deblur/deblurring_nafnet_2025may.onnx',
  'models/parsing/resnet18.onnx',
  'models/pose/mobilenetv2.onnx',
  'models/inpainting/inpainting_lama_2025jan.onnx'
)

foreach ($model in $models) {
  if (!(Test-Path $model)) {
    throw "Verified production model missing before packaging: $model"
  }
  $relative = $model.Substring('models/'.Length)
  $destination = Join-Path "$package/models" $relative
  $parent = Split-Path $destination -Parent
  New-Item -ItemType Directory -Force -Path $parent | Out-Null
  Copy-Item $model $destination -Force
}

if (!(Test-Path 'models/model-registry.json')) {
  throw 'Model registry missing before packaging'
}
Copy-Item 'models/model-registry.json' "$package/models/model-registry.json" -Force

Write-Host "Staged $($models.Count) verified production checkpoints into Windows package."
