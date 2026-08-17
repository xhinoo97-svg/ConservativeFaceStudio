# BLOCK_MODEL_ARCHITECTURE_V2.md

Status date: 2026-08-17

Base: certified V1 `2767513f95dde2d417e7c6f1faf2357149a1a32f` on isolated branch `research/paper-quality-local-v2`.

## Global invariants

Two independent modes remain explicit.

### FORENSIC / CONSERVATIVE

- Observed evidence first.
- Exact provenance and original source indices.
- Wrong-person observed contribution = 0 pixels.
- Partial references have component-local authority only.
- SFace safety threshold remains frozen; it is never lowered to improve acceptance.
- Generated content is never represented as observed evidence.
- Healthy pixels are preserved unless a pre-existing conservative authority explicitly permits a change.

### PAPER QUALITY / PERSONALIZED

- Generative restoration is allowed only inside a measured damage/repair authority.
- Every generated pixel is labeled `GENERATED_MODEL_INFERRED` in generated-pixel mask and source map.
- `OBSERVED_MAIN` and `OBSERVED_SAME_PERSON_REFERENCE` outrank generated information as evidence.
- Heavy candidates branch from a common controlled checkpoint; they are never blindly chained.
- Identity is a hard gate before perceptual ranking.
- One heavy model is loaded at a time on the 16 GB target.

## Common data objects to add

### `PersonIdentityProfile`

Local-only profile:

- consensus full-reference identity embedding
- accepted full-reference source indices
- partial-reference component authority
- pose / yaw / pitch / roll
- face/reference quality
- blur / noise / exposure / JPEG severity
- component visibility and sharpness
- per-component crops/features
- photometric metadata

No photograph or embedding leaves the machine.

### `RestorationCandidate`

Common adapter output:

- image / aligned face
- model key and version
- backend
- generated mask
- per-component quality features
- SFace identity evidence
- optional secondary identity evidence
- landmark geometry
- healthy-region change
- artifact/boundary/colour metrics
- peak RAM
- model load time
- inference time
- rejection reason or acceptance state

Interface target:

`FaceRestorerAdapter.restore(face, context) -> RestorationCandidate`

## 13-block map

| Block | PRIMARY | SECONDARY | FALLBACK | REJECTED / gated | Why |
|---|---|---|---|---|---|
| 1 IMPORT | OpenCV/Pillow deterministic import | none | fail on invalid input | any learned generator | Import creates the immutable evidence snapshot; a model cannot add information here. |
| 2 DEBLUR | NAFNet for mild/general deblur; in Paper mode GPEN BFR-512 candidate from the common aligned checkpoint when damage is severe | GFPGAN v1.4 and CodeFormer candidates only after GPEN qualification and router trigger | existing conservative OpenCV path | blind sequential GPEN→GFPGAN→CodeFormer; global beauty processing | Mild restoration should stay low-risk; severe face restoration is candidate generation, not destructive chaining. |
| 3 ENHANCE | degradation-aware deterministic routing; FBCNN for severe JPEG; existing NAFNet denoise for noise | Zero-DCE++ only for genuinely detected low light; SwinIR only if validation proves a useful specialist | calibrated luminance/CLAHE V1 path | pre-clean every image; colour/beauty enhancement | Enhancement must solve a diagnosed degradation and avoid erasing identity evidence before BFR. |
| 4 LANDMARKS | YuNet | MediaPipe Face Landmarker after dependency/runtime qualification; optional 3DDFA for hard pose | verified reference/RANSAC geometry or abstain | invented landmarks; generator-derived geometry as truth | Geometry must be observed or confidently inferred by geometry models. |
| 5 ALIGN | deterministic similarity/partial affine/RANSAC to 512 aligned face checkpoint | verified local refinement | existing deterministic alignment | learned/generative alignment | All heavy face candidates must receive the same geometry checkpoint. |
| 6 OCCLUSION_MASK | existing face parsing + future lightweight DamageMaskNet | deterministic/reference consensus supporting evidence | conservative heuristic mask | a generative restorer deciding its own repair mask | DamageMaskNet adds class, soft/binary mask, confidence and affected component; it does not authorize identity on its own. |
| 7 REGION_SELECT | existing `component_bank` + `reference_memory` extended with per-component quality/reliability | DMDNet/ASFFNet/RefineFIR-inspired component scoring and copy-or-not decision | preserve MAIN / abstain | one global “best reference” for full face | Left/right eyes, brows, nose, philtrum, mouth, cheeks, chin, jaw, forehead and contour compete independently. |
| 8 INPAINT | observed same-person reference repair first; then component-bank reconstruction | Paper mode: CodeFormer inpainting; then GPEN inpainting if separately qualified; RefFaceInpainting only after hardware/license qualification | small non-identity-critical LaMa residual under existing conservative limits | LaMa-generated eye/nose/lip as evidence; wrong-person donor; unrestricted generator | Missing identity-critical information should use real personal evidence first; generative pixels remain explicit model inference. |
| 9 FUSION | deterministic confidence/source-map fusion | component-level candidate ensemble only if validation beats whole-face selection | MAIN preservation / rollback | learned whole-image fusion that silently rewrites healthy pixels | Fuse using component confidence, identity, geometry, reference agreement, boundary and photometric consistency. |
| 10 FRONTALIZE | pose/3D geometry only | optional qualified 3DDFA geometry | V1 roll-only / abstain | hidden-side synthesis in Conservative mode | Paper mode may synthesize hidden regions only with generated provenance. |
| 11 IDENTITY_CHECK | existing frozen SFace hard gate | optional ArcFace/InsightFace second backend after separate calibration | fail closed / rollback | lower SFace threshold; max raw-reference score; wrong-person anchors | Full same-person refs form robust consensus anchors; partial refs are local only. Use median/trimmed-mean/consensus embedding rather than max raw reference score. |
| 12 UPSCALE | Conservative: Lanczos | Paper: Real-ESRGAN x2 only after identity/quality/CPU qualification; SwinIR alternative if it wins | Lanczos | global background SR by default; SR before identity-critical restoration without benchmark | CPU target makes face-only or bounded SR preferable. |
| 13 EXPORT | deterministic encoder + reports | none | explicit failure | model-driven export | Persist final image, provenance, generated mask, per-component source map, model-selection report, identity/damage reports, timing, peak RAM, model versions/hashes. |

## DamageMaskNet contract

Output channels/classes must cover:

`HEALTHY`, `BLUR`, `MOTION_BLUR`, `PIXELATION`, `BLOCK_MOSAIC`, `JPEG_ARTIFACT`, `SCRIBBLE`, `STICKER`, `OPAQUE_BLOCK`, `BLACK_BAR`, `PARTIAL_OCCLUSION`, `MISSING_COMPONENT`.

For each detected region persist:

- class
- soft probability
- binary mask
- confidence
- affected facial component

Architecture candidates to benchmark rather than assume:

1. MobileNetV3 encoder + lightweight segmentation head.
2. Small U-Net.
3. Lightweight DeepLab variant.

Ground truth comes from exact synthetic corruption masks over real face crops. Architecture is chosen by validation IoU/F1, CPU time, RAM and ONNX parity, not paper reputation.

## Per-component Personalized Reference Bank

The existing V1 component bank already defines most required facial regions and the V1 reference memory supports up to nine sources. V2 adds independent quality vectors for each `(source, component)`:

- observed support/coverage
- sharpness
- blur/noise
- exposure
- occlusion
- pose compatibility
- geometric residual
- photometric consistency
- full-reference identity acceptance or partial-reference local authority

A valid final source map may therefore contain, for example, left eye from REF3, right eye from REF5, nose from REF2, mouth from REF7, while forehead remains MAIN.

## Candidate routing

All heavy generators start from the same aligned checkpoint.

### LIGHT

- NAFNet / FBCNN specialist only as indicated by degradation.
- Do not run a face generator if gates already pass.

### MEDIUM FACE DAMAGE

1. GPEN candidate.
2. Hard identity / healthy-region / geometry gates.
3. Stop if acceptable.
4. Only if rejected for repair quality rather than identity safety, allow next qualified candidate.

### SEVERE

1. Observed reference/component reconstruction where available.
2. GPEN.
3. GFPGAN v1.4 if router says GPEN insufficient.
4. CodeFormer if still needed or if inpainting/fidelity-control path is specifically indicated.

### STICKER / SCRIBBLE / MOSAIC

1. DamageMaskNet region/component.
2. Reference component bank.
3. Same-person observed reconstruction if supported.
4. Paper-only generated candidates if evidence is insufficient.
5. Compare generated component against all usable personal evidence.
6. Fuse only inside valid authority.
7. Identity gate.
8. Atomic rollback on failure.

## Candidate hard gates and ranking

Candidate selection is never “looks sharper”.

### Hard rejection gates

- SFace identity safety
- wrong-person observed pixels = 0
- provenance validity
- healthy-region preservation ceiling
- component/landmark geometry ceiling
- non-finite or malformed output

### Development-calibrated ranking after hard gates

Features:

- `IDENTITY`
- `COMPONENT_REFERENCE_AGREEMENT`
- `LANDMARK_GEOMETRY`
- `HEALTHY_REGION_PRESERVATION`
- `PERCEPTUAL_QUALITY`
- `ARTIFACT_SCORE`
- `BOUNDARY_QUALITY`
- `COLOUR_CONSISTENCY`

Weights may be calibrated only on DEVELOPMENT/VALIDATION data, never a final holdout. Every winner stores a machine-readable explanation.

## Provenance V2

Minimum semantic classes:

- `OBSERVED_MAIN`
- `OBSERVED_SAME_PERSON_REFERENCE(source_index)`
- `GENERATED_MODEL_INFERRED(model_key)`
- `UNRESOLVED`

Wrong-person references can never appear as observed output provenance. A generated output matching a reference visually remains `GENERATED_MODEL_INFERRED` unless the pixels were actually copied from the observed source through an auditable transform.

## CPU / RAM execution policy

For heavy models:

`LOAD -> RUN batch_size=1 -> STORE candidate -> UNLOAD -> GC/release -> next model`.

Record model-load RSS, peak inference RSS and post-unload RSS. A leak is a blocker. Backends are qualified independently: PyTorch CPU first, then ONNX Runtime CPU, then OpenVINO CPU/iGPU only when conversion parity and hardware support are measured.

## Immediate implementation sequence

1. Preserve this architecture as research documentation only.
2. Implement exactly one GPEN BFR-512 vertical slice: detect -> align 512 -> GPEN CPU -> SFace -> comparison -> actual RAM/time report.
3. Resolve GPEN license/weights status before any production qualification or redistribution.
4. Only after genuine GPEN output, add the common `FaceRestorerAdapter` boundary needed to compare GFPGAN v1.4 under the same input/output contract.
5. Do not touch V4 final holdout during these phases.
