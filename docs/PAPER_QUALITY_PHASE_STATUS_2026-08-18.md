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

Recovered authoritative evidence:

- workflow: `Research FBCNN JPEG specialist vertical slice` run `32304605507`, success;
- exact CFS commit: `6ea5d113a1400da97864fcb43c69c53f789605ea`;
- artifact: `9384276352`, `fbcnn-color-q20-upstream-adapter-3`;
- artifact archive SHA-256: `da38fb6d925c1f7011bebc1929c17153ffc612f092c7cf048535fbb631a1514c`;
- official upstream commit: `54d1831927506b3247e2d4d245abb4f4dab1a1cd`;
- official checkpoint SHA-256: `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`;
- code/checkpoint license recorded from official upstream: Apache-2.0.

Real CPU JPEG QF=20 DEVELOPMENT evidence:

- PSNR `34.6184 -> 36.7801 dB`.
- SSIM `0.948646 -> 0.963414`.
- SFace `0.957095 -> 0.969138`.
- measured 512 inference after warm-up: `9.0507 s`.
- peak inference RSS: `1226.25 MiB`.
- needs double-JPEG, social recompression, broader validation, Windows and EliteBook qualification.

## Phase 8 — DamageMaskNet

Status: **U-NET HYPOTHESIS STOPPED — MODEL/DATA QUALITY FAIL**.

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
3. acquisition switched to the mixed FairFace + ControlFace source bank without changing U-Net hyperparameters. The evidence has now been recovered and verified from GitHub Actions run `32087249287` at exact CFS commit `08df3163f317fe6a16571337178168d9845a749c`.

Attempt 3 evidence:

- workflow conclusion: success; training, export and evidence-contract steps all completed;
- artifact: `9307331508`, `damage-mask-net-unet-mixed-dev-3`;
- artifact archive SHA-256: `e3b7aa05bcfdc9d48a803595218727f84bc255a4095caab84239c1850f7b52b8`;
- checkpoint SHA-256: `e3b05272782aded20f209ddd39a3ac847cf4f3a90e5e3f02b63cae90474e2b7d`;
- ONNX SHA-256: `64e032d8693edc55d69a0a77d8665034d4edbeff43a93b6a622c4639a0d018c7`;
- train/validation identity separation: verified; final holdout used: false;
- source bank: 22 train identities, 2 identity-disjoint validation identities, 1452 train samples and 66 validation samples;
- ONNX argmax parity: exact; maximum absolute logit difference `5.2452e-6`;
- first-call ONNX Runtime CPU inference: `0.01282 s` per aligned face;
- process RSS: `824745984` bytes under the 80% resource contract;
- validation damage macro-F1: `0.173198`;
- validation damage macro-IoU: `0.113028`;
- F1 was zero for `BLUR`, `MOTION_BLUR`, `PIXELATION`, `BLOCK_MOSAIC`, `JPEG_ARTIFACT` and `STICKER`.

Classification: **MODEL/DATA QUALITY FAIL**. Infrastructure, checkpoint creation, loader, ONNX export and inference are verified, but the produced mask is not accurate enough to authorize restoration. Under the three-attempt policy, the small U-Net hypothesis is stopped. Do not tune or rerun it.

Two later historical Actions runs (`32087329763` and `32088670503`) already existed when this audit began because subsequent path-triggering commits caused the workflow to execute again. This audit did not launch them and does not use them to reinterpret or tune attempt 3.

Replacement comparison: official torchvision LR-ASPP with MobileNetV3-Large backbone, source `pytorch/vision@c6f39778e636ec40a69bdbc74386818c57a65af3` (`v0.16.2`). Run `32675225785` on exact Track B HEAD `2b775b8186ac974f568b3644c59350cc1f12181a` passed every infrastructure step and the pre-run DEVELOPMENT adequacy gate: damage macro-F1 `0.711144 >= 0.70`, macro-IoU `0.569570 >= 0.55`, minimum class F1 `0.423585 >= 0.35`. Artifact `9502642834`, archive SHA-256 `0bef114cfeed95ebcceb81ce8f5dfc43c3fdb37bca82c69a346ed6219c137a11`; checkpoint SHA-256 `d510e6991cca582c3696b6b9132bf3fdb7948e240f4bf136440d8b75046910f4`; ONNX SHA-256 `708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9`. Result: **DEVELOPMENT MASK ADEQUACY PASS; NOT PRODUCTION QUALIFIED**. Only two validation identities were used, boundary/class errors remain visible, Windows is untested and upstream provides no explicit checkpoint redistribution license. RefFace remains unauthorized.

External validation run `32676602851` reused the exact frozen checkpoint/ONNX without retraining or tuning on 40 new ControlFace identities, excluding every prior ControlFace identity and balancing five identities in each race-by-sex stratum (20 female, 20 male). All 880 cases completed with zero errors and the unchanged overall gate passed: macro-F1 `0.716639`, macro-IoU `0.579849`, minimum class F1 `0.387499`. Artifact `9502870418`, archive SHA-256 `1357c983343b22f81942b130ae359a0051c0d6b79750417d17d832a89cf6b19c`. Result: **OVERALL DEVELOPMENT PASS; DOMAIN ROBUSTNESS NOT QUALIFIED; NOT PRODUCTION QUALIFIED**. Asian macro-F1 was `0.696471` and minimum class F1 `0.243086`; age-50 minimum class F1 was `0.322288`. Binary mask precision was `0.673451`, so healthy-pixel over-masking remains material. RefFace remains unauthorized.

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

RefFace remains **NOT_RUN**. DamageMaskNet attempt 3 is now verified, but its mask failed the model/data quality gate. RefFace cannot execute until a replacement damage-localization architecture produces an adequate DEVELOPMENT mask. Do not run RestoreFormer++, VQFR or GPEN-inpainting before that prerequisite is satisfied or the routing contract is formally redesigned and independently validated.

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

Resolve the checkpoint licensing path and pre-register domain-aware acceptance on a new identity-disjoint real-photo validation bank. Do not tune on the observed 40-identity set or the stopped small U-Net. RefFaceInpainting remains gated until domain robustness, compatible licensing and Windows/offline evidence are demonstrated.
