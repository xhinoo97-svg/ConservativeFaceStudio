# Conservative Face Studio — Release State

- Branch: `feature/block-pipeline-v1`
- Current remote HEAD: `d4a017705273a2c939ddd3e6e1e23558b805da96`
- Current local commit: `8214455` (tree `b55b65e45cb67bb4f919052831b9dc5a6391cbcd`, byte-identical to remote HEAD)
- Product-completion batch: **committed and pushed once**
- Functional follow-up: **Windows-safe PID probe plus explicit MAIN provenance fixed and verified locally**
- Last fully green release commit: **none yet**
- `PRODUCT_COMPLETE_PRE_TUNING`: **FALSE**
- TARGET95: **REPORT ONLY**

## First current release blocker

Windows build `#1176` (`31448331116`) passed checkout, dependencies and the complete import/model
manifest gate, then stopped at pytest after two tests with `KeyboardInterrupt`.

The root cause is the new cross-process restoration lock using Unix-style `os.kill(pid, 0)` to
probe a PID on Windows. Windows does not provide Unix signal-0 semantics and that call can signal
or terminate the process being checked. The local follow-up uses a non-signalling Win32
`OpenProcess + WaitForSingleObject` query, retains the Unix path on non-Windows hosts, and adds a
regression test proving the Windows path never calls `os.kill`.

Female-domain `#444` completed its full benchmark (76/80 sources, 346 completed cases) and then
reported 33 runtime errors. All 33 shared one root cause: a MAIN-only gaussian/no-op path could
finish with `provenance_map=None`. `Workspace` now establishes an explicit zero-valued `uint16`
MAIN-original provenance map at creation and rejects mismatched supplied maps. The same first-ten
portrait profile improved from 44 completed + 6 provenance errors to **50 completed + 0 errors**.

## Verified local functional gates

- `compileall`: **PASS**
- core imports: **PASS**
- local GUI import: **BLOCKED BY HOST `libEGL.so.1`; Windows packaged GUI remains the required gate**
- pytest: **396 passed / 0 failed**
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

- Windows build `#1176`: **FAIL at pytest due Windows PID-probe bug; local root fix PASS**.
- Female-domain `#444`: **FAIL after benchmark; 33 missing-provenance errors share one locally fixed root cause**.
- Previous Female-domain `#443` and duplicate `#442`: **CANCELLED at the 140-minute job timeout**.
  The first real failure was Commons HTTP 429 while the duplicate jobs downloaded original,
  full-resolution files; the missing final JSON was a cascade. No pipeline crash or provenance
  violation was demonstrated. The local bounded-thumbnail fix and removal of duplicate feature
  push triggers are regression-tested without reducing the 80-portrait/60-source gate.
- Push policy: **Female-domain #444 is finished; one logical follow-up push is now allowed**.

## Not yet verified

- Windows portable smoke with the local completion batch.
- Portable ZIP creation/test.
- Inno Setup build.
- Clean silent install.
- Installed GUI/offline verification.
- GitHub artifact upload for the new tested commit.
- `PRODUCT_COMPLETE_PRE_TUNING` remains false until all of those pass.

## Next exact action

Commit the two verified functional root fixes and lightweight Female report artifact together,
push once, and allow Windows/Female CI to finish without interruption. Inspect only the first real
failure or record the verified portable ZIP, installer and installed-app artifacts.
