# DamageMaskNet source bank V2

Status: DEVELOPMENT
Branch: `research/paper-quality-local-v2`

## Purpose

Remove DamageMaskNet training from any dependency on Wikimedia rate limits while preserving auditable source provenance, real-face exposure, explicit identity separation, and final-holdout isolation.

## Sources

### FairFace

- official project: `joojs/fairface`
- license reported by the official project: CC BY 4.0
- official aligned image archive: Google Drive file `1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86`
- role: **real-face training domain only**
- first vertical slice: 8 deterministic training images selected by a stable filename hash ordering
- each extracted member receives its own SHA-256 in the generated source-bank manifest
- the archive SHA-256 is recorded as an observed acquisition hash; no upstream expected archive digest is invented

FairFace does not expose stable subject identity labels in the image archive used here. Therefore FairFace records are intentionally excluded from the held-out identity validation partition.

### ControlFace10K

- official dataset: `HuMInGameLab/ControlFace10K`
- pinned revision: `a03589de1a9e028b2d16fa1eb0e019a6930e817c`
- license: CC BY 4.0
- role: explicit synthetic identity and multi-view/pose source
- accessed with HTTP Range through `remotezip==0.12.3`; the complete 3.14 GB archive is not downloaded for the vertical slice
- identity is read from the documented `identity-*` directory hierarchy; filename tokens are not treated as identity authority

First vertical slice selection:

- training: 10 female identities + 4 male identities
- validation: 2 independent female identities
- exactly one image is selected per identity for this segmentation vertical slice
- train and validation identity keys are disjoint before training starts

## Why this split is scientifically defensible

DamageMaskNet is a degradation segmentation model, not a face-recognition model. It needs realistic face appearance plus exact corruption masks.

The first vertical slice therefore uses:

1. real FairFace crops in training to reduce the synthetic-face domain gap;
2. ControlFace10K identities in training for controlled pose/appearance diversity;
3. held-out ControlFace10K identities in validation so identity separation is explicit rather than inferred;
4. exact CFS-generated corruption masks for every damage class.

The validation identities are synthetic by construction, so they cannot be the same real subjects as the FairFace training images. This gives a strict identity-disjoint validation partition without pretending FairFace provides identity labels that it does not provide.

## Smartphone-domain construction

The clean source does not need to have been captured by a phone for every training sample. The target phone domain is produced by the corruption pipeline after clean-face acquisition, including:

- Gaussian blur
- motion blur
- pixelation
- block mosaic
- JPEG artifact regions
- scribble
- sticker
- opaque block
- black bar
- partial occlusion
- missing facial component

The larger qualification bank will additionally include whole-image phone-like operations such as resize/re-encode cycles, screenshot scaling, low-light/noise, double JPEG, and mixed degradations.

## Integrity rules

- V2/V3/V4 final holdouts are not used.
- Source files are hashed before training.
- Synthetic damage masks are exact because CFS creates the corruption.
- Generated pixels are never labelled observed evidence.
- The source-bank manifest is uploaded with the training evidence.
- This 24-source bank is DEVELOPMENT only and cannot qualify DamageMaskNet for release.

## Scale-up gate

After the vertical slice proves data acquisition, training, ONNX export and parity, expand to approximately 300–400 source identities/images with an identity-disjoint DEVELOPMENT/VALIDATION split and the intended female-primary-domain weighting. The final holdout is frozen only after architecture and routing choices are complete.
