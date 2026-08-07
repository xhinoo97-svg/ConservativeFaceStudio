$ErrorActionPreference = 'Stop'

$package = 'dist/ConservativeFaceStudio'
$models = @(
  'models/opencv_zoo/face_detection_yunet_2023mar.onnx',
  'models/opencv_zoo/face_recognition_sface_2021dec.onnx',
  'models/nafnet/deblurring_nafnet_2025may.onnx',
  'models/face_parsing/resnet18.onnx',
  'models/head_pose/mobilenetv2.onnx',
  'models/lama/inpainting_lama_2025jan.onnx'
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
