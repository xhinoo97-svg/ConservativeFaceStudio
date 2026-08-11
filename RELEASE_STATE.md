# Conservative Face Studio — Release State

- Branch: `feature/block-pipeline-v1`
- Current remote HEAD: `feeb27675e9b3f63f5af10c0ea01e9ac181a4e11`
- Current local base HEAD: `feeb27675e9b3f63f5af10c0ea01e9ac181a4e11`
- Local product-completion batch: **uncommitted; final local regression complete**
- Last fully green release commit: **none yet**
- `PRODUCT_COMPLETE_PRE_TUNING`: **FALSE**
- TARGET95: **REPORT ONLY**

## First current release blocker

Windows build `#1175` (`31438960782`) reached the portable package and failed at:

`Smoke and offline test portable package`

All preceding steps passed, including compile/imports, pytest, validation, CPU benchmark,
practical runtime, extended matrix runtime, PyInstaller EXE, production-model bootstrap,
six real CPU model smokes, runtime registry export and model staging. ZIP and installer were
correctly skipped as cascades.

The complete job log proves that source-process YuNet/SFace smoke passed under OpenCV 5 auto
engine, but the packaged process forced classic engine `1` before importing OpenCV. Classic
YuNet then failed when `FaceDetectorYN` redefined its input shape. The prepared local policy
uses auto engine `3`, which passes the same real YuNet/SFace route and keeps OpenCV's supported
fallback behavior. A subprocess regression test pins this boot policy.

## Verified local functional gates

- `compileall`: **PASS**
- core imports: **PASS**
- local GUI import: **BLOCKED BY HOST `libEGL.so.1`; Windows packaged GUI remains the required gate**
- pytest: **393 passed / 0 failed**
- conservative validation suite: **PASS**
- CPU benchmark: **PASS**
- practical public-portrait runtime: **120/120 complete; 0 errors**
- practical TARGET95 report-only: **75/83 applicable pass**
- extended degradation matrix runtime: **82/82 complete; 0 errors**
- matrix TARGET95 report-only: **5/70 applicable pass**
- six production manifests/checksums: **PASS**
- real CPU inference: **YuNet, SFace, NAFNet, Face Parsing, Head Pose, LaMa PASS**
- production pipeline MAIN + `0..9` references: **10/10 PASS**
- incompatible final donor abstention: **PASS for counts 1..9**
- MAIN target/canvas contract: **PASS for counts 0..9**
- provenance shape/firewall: **PASS for counts 0..9**
- offline network block: **PASS; 0 network attempts**
- offline representative MAIN+2-reference pipeline/export: **PASS**
- model updater atomic staging/checksum/smoke/activation/rollback: **PASS**
- real updater staging with preserved ONNX suffix, YuNet/SFace pre/post-activation inference: **PASS**
- app updater verified installer staging and restoration lock: **PASS**
- cross-process restoration activity lock and stale-lock recovery: **PASS**
- recovery project and per-block atomic checkpoints: **PASS**

## Current product-completion batch

- OpenCV 5 stable DNN engine policy and regression test.
- Standalone script bootstrapping so every CI/product tool resolves `app` and applies
  the same OpenCV boot policy when invoked as `python scripts/...`.
- Real `HardwareProfile`: CPU/cores/RAM/GPU/vendor/OpenCL/CUDA/Vulkan/VRAM/disk/OS/architecture,
  acceleration self-tests and safe CPU fallback.
- Definitive model folders under `models/{detection,identity,landmarks,parsing,pose,deblur,reference,inpainting,restoration,optional}`.
- Six production weights routed to the definitive folders.
- Real production smoke callback used by the updater before activation.
- Separate atomic MODEL UPDATE and APP UPDATE flows; previous working model preserved and pack rollback verified.
- User-updated verified models take precedence over bundled weights on next run.
- GUI project open/save/recovery, hardware status and update controls.
- Bounded runtime settings with persistent per-user override for hardware mode/update manifest.
- Offline verifier now blocks network and executes a representative restoration pipeline.
- Normal GUI restoration resolves only checksum-verified local models; downloads are confined to explicit bootstrap/update flows.
- Portable/installer folder verification and installed `--smoke-test`, `--verify-installation`, `--offline-test` gates.
- Portable metadata/license contract: runtime registry, complete manifest catalog and packaged license notice.
- Release update manifest generation for app plus six individual model assets.
- Female-domain quality changed to report-only while runtime/corruption/provenance remain blocking.
- Complete source/model inventory in `product-audit.json`.
- Female-domain source acquisition now requests bounded 640 px Commons thumbnails while
  retaining original dimensions, page URL, license and original URL for auditability. Full
  originals are used only when Commons provides no thumbnail.

## Remote workflows

- Windows build `#1175`: **FAIL at portable smoke; prior functional/model/build steps PASS**.
- Female-domain `#443` and duplicate `#442`: **CANCELLED at the 140-minute job timeout**.
  The first real failure was Commons HTTP 429 while the duplicate jobs downloaded original,
  full-resolution files; the missing final JSON was a cascade. No pipeline crash or provenance
  violation was demonstrated. The local bounded-thumbnail fix and removal of duplicate feature
  push triggers are regression-tested without reducing the 80-portrait/60-source gate.
- Push policy: **one logical push is now allowed; do not push again while its workflows run**.

## Not yet verified

- Windows portable smoke with the local completion batch.
- Portable ZIP creation/test.
- Inno Setup build.
- Clean silent install.
- Installed GUI/offline verification.
- GitHub artifact upload for the new tested commit.
- `PRODUCT_COMPLETE_PRE_TUNING` remains false until all of those pass.

## Next exact action

Create one logical product-completion commit, push once, and let the new Windows and
Female-domain runs finish without interruption. Inspect the first real failure if either run is
red; otherwise record the portable ZIP, installer and installed-app verification artifacts.
