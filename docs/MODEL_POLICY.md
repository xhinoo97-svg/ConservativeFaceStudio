# Model integration policy

ConservativeFaceStudio keeps model code, model weights, datasets, and generated artifacts as separate licensing concerns.

## Strict mode rule

Strict mode must prefer observed pixels and deterministic transforms. Generative restoration, inpainting, frontalization, and neural super-resolution are optional and remain disabled unless explicitly enabled. A model must not silently replace unsupported facial detail.

## InsightFace

- Upstream code: MIT.
- Upstream pretrained recognition models and annotated training data: upstream documentation states non-commercial research use unless separately licensed.
- Policy here: do not bundle or auto-download InsightFace recognition weights. The adapter may use user-supplied or separately licensed weights.

## MediaPipe Face Landmarker

- Google MediaPipe code samples/documentation are Apache-2.0/CC BY 4.0 as documented upstream.
- The task model bundle has its own model-card/distribution terms and is therefore not bundled until those exact terms and a stable checksum are recorded.
- Policy here: allow a user-provided `.task` file through the model registry after checksum verification.

## Real-ESRGAN

- Upstream code: BSD-3-Clause.
- Neural super-resolution is not conservative reconstruction because it can synthesize texture.
- Policy here: strict mode uses deterministic Lanczos. Real-ESRGAN stays optional and results must be labelled generative/neural in provenance.
- CPU inference can be slow; large-image tiling and memory limits must be used before enabling it by default on CPU-first machines.

## CodeFormer / GFPGAN

- These systems can improve perceptual quality but may alter identity or facial geometry.
- CodeFormer exposes an explicit quality/fidelity trade-off; lower fidelity settings may invent more detail.
- Policy here: never use them in strict mode. They may be offered later as a clearly separated optional generative branch with before/after identity checks and provenance.

## LaMa

- Upstream code: Apache-2.0.
- Inpainting is generative for hidden facial regions even when the mask is correct.
- Policy here: disabled in strict mode; optional branch only, with mask export and a provenance flag marking generated pixels.

## Datasets

Datasets are never redistributed by this repository unless their licenses explicitly permit redistribution. Evaluation adapters may point users to official dataset pages. Dataset licenses must be reviewed separately from model-code licenses.

## Required checks before adding any pretrained model

1. stable upstream source URL;
2. exact code license;
3. exact model-weight license;
4. SHA-256 checksum;
5. maximum expected download size;
6. CPU/GPU execution requirements;
7. known failure modes;
8. whether the model can synthesize unsupported facial details;
9. provenance label used by the application;
10. regression tests and a failure-safe fallback.
