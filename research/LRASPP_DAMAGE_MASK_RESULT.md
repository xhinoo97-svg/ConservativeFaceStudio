# LR-ASPP damage-mask DEVELOPMENT result

Date: `2026-08-24`  
Track B exact HEAD: `2b775b8186ac974f568b3644c59350cc1f12181a`  
Workflow: `Research DamageMask LRASPP comparison`  
Run: `32675225785` (`SUCCESS`)  
Artifact: `damage-mask-lraspp-dev-1`, ID `9502642834`  
Artifact archive SHA-256: `0bef114cfeed95ebcceb81ce8f5dfc43c3fdb37bca82c69a346ed6219c137a11`

## Classification

`DEVELOPMENT_MASK_ADEQUACY_PASS_NOT_PRODUCTION_QUALIFIED`

The pre-run DEVELOPMENT gate passed without changing thresholds:

| Frozen check | Required | Observed | Result |
|---|---:|---:|---|
| Damage macro-F1 | `>=0.70` | `0.711144` | PASS |
| Damage macro-IoU | `>=0.55` | `0.569570` | PASS |
| Minimum per-damage-class F1 | `>=0.35` | `0.423585` | PASS |

This is not a release, Target95, RefFace, Windows or product qualification result. RefFace execution remains unauthorized.

## Exact upstream and checkpoint provenance

- official repository: `https://github.com/pytorch/vision.git`;
- exact source: tag `v0.16.2`, commit `c6f39778e636ec40a69bdbc74386818c57a65af3`;
- installed LR-ASPP and MobileNetV3 source files: byte-equal to that checkout;
- code license: BSD-3-Clause; LICENSE SHA-256 `6502f676851cfe25f8af75531dfb32375b7325b73c37e7b43741fa422893e71d`;
- official MobileNetV3 backbone: `22139423` bytes, SHA-256 `8738ca797c879b547d18bbd15da5736ff2557b2036a9af72225393ca61759a04`;
- upstream checkpoint redistribution license: `NOT_EXPLICIT_UPSTREAM_RESEARCH_ONLY`;
- final trained checkpoint SHA-256: `d510e6991cca582c3696b6b9132bf3fdb7948e240f4bf136440d8b75046910f4`;
- final ONNX SHA-256: `708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9`.

The checkpoint licensing ambiguity is a blocking production issue even though the DEVELOPMENT quality gate passed.

## Data and training

- unchanged mixed DEVELOPMENT source-bank contract;
- 22 training identities: 8 FairFace real-image sources plus 14 ControlFace identities;
- 2 identity-disjoint ControlFace validation identities;
- 1452 training samples and 66 validation samples;
- exact deterministic synthetic masks;
- final holdout used: false;
- architecture: official LR-ASPP/MobileNetV3-Large, 3,220,008 parameters;
- epochs: 8; batch size: 8; deterministic seed: `240823`;
- measured training time: `715.378 s` on the GitHub CPU runner.

## Per-class validation

| Class | F1 | IoU |
|---|---:|---:|
| BLUR | 0.564661 | 0.393399 |
| MOTION_BLUR | 0.423585 | 0.268702 |
| PIXELATION | 0.637152 | 0.467515 |
| BLOCK_MOSAIC | 0.657849 | 0.490145 |
| JPEG_ARTIFACT | 0.810323 | 0.681128 |
| SCRIBBLE | 0.758703 | 0.611217 |
| STICKER | 0.673814 | 0.508084 |
| OPAQUE_BLOCK | 0.697512 | 0.535523 |
| BLACK_BAR | 0.793916 | 0.658260 |
| PARTIAL_OCCLUSION | 0.856123 | 0.748440 |
| MISSING_COMPONENT | 0.948949 | 0.902858 |
| HEALTHY, report-only | 0.985574 | 0.971559 |

Motion blur is the weakest class. The saved visual also shows imperfect boundary/class localization, so the aggregate DEV pass must not be treated as proof that masks are sufficiently reliable for reference-guided pixel generation.

## Runtime, loader and safety evidence

- 308 official backbone tensors loaded with a strict key match;
- full trained checkpoint reload works offline with maximum absolute drift `0`;
- ONNX checker passed;
- PyTorch/ONNX argmax segmentation is exactly equal;
- maximum absolute ONNX logit difference: `3.81470e-5`;
- measured PyTorch CPU inference: `0.01781 s` per aligned 192x192 face;
- measured first-call ONNX Runtime CPU inference: `0.00675 s`;
- process RSS snapshot: `1,080,541,184` bytes under the 80% resource contract;
- output type: mask logits only; wrong-person final pixels `0`; provenance violations `0`;
- no network is required after the checkpoint/ONNX artifacts have been acquired;
- Windows and physical HP EliteBook execution: NOT_RUN.

## Decision and exact next action

Keep LR-ASPP as the current DEVELOPMENT mask leader, but do not integrate it into RefFace or the product. Build a substantially larger identity-disjoint DEVELOPMENT/VALIDATION mask bank, verify comparable per-class quality and boundary precision, and resolve or replace the checkpoint licensing path. The stopped small U-Net remains immutable evidence and is not a tuning baseline.
