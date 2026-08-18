# Conservative Face Studio — Paper Quality research status

Snapshot date: 2026-08-18
Branch: `research/paper-quality-local-v2`
Base: certified V1 `2767513f95dde2d417e7c6f1faf2357149a1a32f`

This document records evidence state, not intent. `IMPLEMENTED` is not equivalent to `TEST_PASS`, and `TEST_PASS` is not equivalent to `QUALIFIED`.

## Resource contract

- CPU cap: <=80% logical processors.
- Process RAM: <=80% physical RAM.
- Whole-system RAM: <=80% physical RAM.
- Heavy models: maximum one at a time.
- Status: **IMPLEMENTED; latest post-hardening push-run result NOT_VERIFIED in the current control interface**.

## Phase 3 — GPEN BFR-512

Status: **BENCHMARKING**.

Real Linux CPU DEVELOPMENT evidence:

- SFace clean/restored: `0.95397`.
- measured 512 inference: about `2.697 s`.
- peak RSS: about `1.828 GB`.
- PSNR: `28.07 dB`.
- SSIM: `0.7474`.
- production blockers include licensing/redistribution clarity, Windows and real EliteBook acceptance.

## Phase 4 — GFPGAN v1.4

Status: **BENCHMARKING**.

Real comparable Linux CPU DEVELOPMENT evidence:

- SFace: `0.91665`.
- measured inference: about `2.787 s`.
- peak RSS: about `1.666 GB`.
- PSNR: `30.65 dB`.
- SSIM: `0.8604`.

One case is insufficient to choose a production winner.

## Phase 5 — CodeFormer

Status: **BENCHMARKING**.

- real CPU aligned-face slice at official `w=0.5`: PASS in prior research evidence.
- 80% governor was active.
- exact final phase metrics must be read from the evidence artifact before any comparative report; do not infer or reconstruct them from memory.
- licensing remains a production blocker.

## Phase 7 — FBCNN

Status: **BENCHMARKING / current JPEG leader**.

Real CPU JPEG QF=20 DEVELOPMENT evidence:

- PSNR `34.62 -> 36.78 dB`.
- SSIM `0.9486 -> 0.9634`.
- SFace `0.9571 -> 0.9691`.
- peak RSS about `1.305 GB`.
- needs double-JPEG, social recompression, broader validation, Windows and EliteBook qualification.

## Phase 8 — DamageMaskNet

Status: **IMPLEMENTED VERTICAL-SLICE PIPELINE; TRAINING RESULT NOT_VERIFIED**.

Implemented:

- one frozen 12-class taxonomy: HEALTHY + 11 damage classes;
- exact deterministic synthetic masks;
- mixed open-source source-bank builder;
- FairFace real-face training source;
- ControlFace10K pinned identity/multi-view source;
- identity-disjoint ControlFace validation partition;
- small U-Net training/export script;
- ONNX parity contract;
- fail-closed ONNX runtime returning class map, confidence, soft/binary damage mask, dominant class and affected facial components;
- all 13 CFS facial component regions, including FACE_CONTOUR.

Attempt history:

1. Wikimedia acquisition stopped by HTTP 403 before training.
2. HTTP-compatible downloader verified four sources, then Wikimedia stopped by HTTP 429 before training.
3. acquisition switched to the mixed FairFace + ControlFace source bank without changing U-Net hyperparameters; current training/run result cannot be read from the available Actions interface and is therefore **NOT_VERIFIED**, not PASS or FAIL.

Do not launch a fourth U-Net training attempt merely to recover observability. If third-attempt evidence is eventually confirmed as a model/data failure, stop this U-Net hypothesis and reassess architecture as required by the three-attempt policy.

## Phase 9 — Personalized Reference Bank

Status: **IMPLEMENTED; latest workflow result NOT_VERIFIED**.

Implemented:

- MAIN + up to 9 reference contract;
- full-reference global identity anchors;
- partial-reference component-local authority only;
- wrong-person exclusion;
- robust coordinate-median consensus embedding over accepted FULL refs only;
- per-reference raw diagnostics: blur, noise, exposure, yaw, pitch, roll, resolution and occlusion;
- per-component visibility/sharpness/coverage;
- local `PersonIdentityProfile` with no upload/network behavior;
- per-component ranking;
- intersection with observed geometric component support from existing `component_bank`.

## Phase 10 — Reference-first information-loss route

Status: **IMPLEMENTED; latest workflow result NOT_VERIFIED**.

Reference-first classes:

- PIXELATION
- BLOCK_MOSAIC
- SCRIBBLE
- STICKER
- OPAQUE_BLOCK
- BLACK_BAR
- PARTIAL_OCCLUSION
- MISSING_COMPONENT

Properties:

- reuses the V1 observed-reference repair kernel;
- every component starts from immutable MAIN;
- local subset provenance is remapped to original user source index 1..9;
- high-priority semantic component authority is reserved so cheek/jaw cannot masquerade as eye/nose/mouth;
- BLUR/MOTION_BLUR remain deblur routes;
- unresolved pixels remain unresolved; no generator is called by this block.

## Phase 11 — Candidate selection

Status: **IMPLEMENTED FRAMEWORK; CALIBRATION NOT_RUN**.

Hard gates before ranking:

- SFace threshold exactly `0.363`;
- wrong-person observed pixels = 0;
- provenance violations = 0;
- healthy-region MAE <=8.0;
- explicit calibrated geometry-drift ceiling;
- generated candidate provenance must be `GENERATED_MODEL_INFERRED`.

Ranking weights cannot exist without a calibration ID and DEVELOPMENT/VALIDATION source. FINAL_HOLDOUT is rejected as a weight-calibration source. No production weights are currently claimed.

## Phase 12 — Component-aware deterministic fusion

Status: **IMPLEMENTED; latest workflow result NOT_VERIFIED**.

Absolute authority order:

1. healthy/unchanged MAIN;
2. observed same-person reference pixels with exact original source index;
3. accepted generated candidate only within remaining repair authority.

Component-specific generated candidates precede whole-face fallback. Generated content cannot overwrite observed-reference pixels or healthy MAIN.

## Phase 13 — first specialist: RefFaceInpainting

Status: **PREPARED / NOT_RUN**.

Why first:

- directly targets large facial occlusions with a same-person reference;
- more specialized to CFS sticker/black-bar/missing-region cases than another generic blind restorer.

Verified upstream facts:

- official repository: `WuyangLuo/RefFaceInpainting`;
- pinned source commit for the CFS experiment: `0f1ad75677cc8fae4ae14d878e4c6cfce9365f28`;
- repository license: MIT;
- official generator checkpoint link exists;
- official ArcFace checkpoint link exists;
- upstream test path is CUDA-hardcoded;
- the core `SegBranch` contains one hard-coded `torch.cuda.FloatTensor` allocation;
- official Trainer initializes four discriminators that are unnecessary for inference.

CFS prepared vertical slice:

- manual-only workflow: `.github/workflows/research-refface-cpu-vertical-slice.yml`;
- it does **not** auto-run on push;
- loads only `UnetG + ArcFace resnet101`;
- patches only the one hard-coded CUDA one-hot allocation to `x.new_zeros`;
- rejects residual CUDA calls in the core files;
- uses the CFS ACTIVE yakhyo ResNet18 face-parsing ONNX with expected SHA-256 from the CFS production model registry;
- uses CFS YuNet + SFace with frozen `0.363` threshold;
- uses two same-identity ControlFace views;
- creates a deterministic exact opaque mouth-region mask;
- exported candidate is forced/verified identical to MAIN outside the mask;
- generated pixels are `GENERATED_MODEL_INFERRED`;
- records observed checkpoint hashes, RAM, timing and identity evidence;
- enforces the 80% total-PC budget.

RefFace must remain NOT_RUN until the preceding DamageMaskNet gate state is recoverable/verified. Do not run RestoreFormer++, VQFR or GPEN-inpainting before RefFace is either measured or formally stopped/reassessed.

## Specialist feasibility notes

- **InstantRestore**: highest scientific relevance for MAIN + several same-person refs, but official implementation contains two UNets + two VAEs + CLIP and explicit CUDA/FP16 assumptions; repository licensing is unresolved in the current audit. Research only until real CPU feasibility.
- **OSDFace**: modern one-step severe blind challenger, but official inference is CUDA/stream hard-coded. Research only.
- **RefineFIR**: useful copy-or-not concept; public repository currently lacks an executable model/checkpoint path suitable for CFS benchmark.
- **PerFuSe / RefIPFR**: architecture teachers until official executable implementations exist.

## Release state

FORENSIC_MODE_READY: **TRUE only for certified V1/main**.

PAPER_QUALITY_MODE_READY: **FALSE**.

LOCAL_MODEL_PACK_READY: **FALSE for Paper Quality**.

WINDOWS_INSTALLER_READY: **FALSE for Paper Quality**.

TARGET_HARDWARE_READY: **FALSE**.

QUALITY_TARGET_ACHIEVED: **FALSE**.

PROJECT_FINISHED: **FALSE**.

## Exact next blocker

Recover/verify the already-triggered third DamageMaskNet mixed-source run **without rerunning/tuning it**. Then:

- if PASS: read IoU/F1 per damage class + ONNX parity + resource evidence, decide whether to retain the lightweight U-Net and scale the DEVELOPMENT/VALIDATION bank;
- if FAIL due acquisition/infrastructure only: report and correct infrastructure without changing data/model hypothesis;
- if FAIL due model/data quality: three attempts are consumed for the U-Net hypothesis; stop and benchmark the next lightweight segmentation architecture instead;
- only after that gate is resolved, execute RefFaceInpainting CPU vertical slice attempt 1/3.
