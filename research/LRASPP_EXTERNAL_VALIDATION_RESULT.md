# Frozen LR-ASPP external DEVELOPMENT validation

Date: `2026-08-24`  
Exact Track B HEAD: `edc5da8b55c39815cb34e10da6058ee0d2f4bc90`  
Workflow run: `32676602851` (`SUCCESS`)  
Artifact: `lraspp-external-validation-1`, ID `9502870418`  
Artifact archive SHA-256: `1357c983343b22f81942b130ae359a0051c0d6b79750417d17d832a89cf6b19c`

## Frozen-artifact contract

- trained checkpoint unchanged: `d510e6991cca582c3696b6b9132bf3fdb7948e240f4bf136440d8b75046910f4`;
- ONNX unchanged: `708c7e9c074b2abf98dc95b8e74b3b76d687a63fb2a54a3e374db0bef37ae3a9`;
- retraining or tuning: false;
- threshold changes: none;
- prior ControlFace identities excluded: 16;
- external identities: 40, identity-disjoint;
- balance: 20 female, 20 male, five identities per African/Asian/Caucasian/Indian by sex stratum;
- age counts: 25 = 12, 50 = 17, 65 = 11;
- cases: 40 identities x 11 damage classes x 2 deterministic repetitions = 880;
- completed: 880; errors: 0; final holdout used: false.

Dataset: ControlFace10K at pinned revision `a03589de1a9e028b2d16fa1eb0e019a6930e817c`, CC BY 4.0. This is an explicit-identity synthetic domain, not a real-photo or production benchmark.

## Overall frozen gate

| Frozen check | Required | Observed | Result |
|---|---:|---:|---|
| Damage macro-F1 | `>=0.70` | `0.716639` | PASS |
| Damage macro-IoU | `>=0.55` | `0.579849` | PASS |
| Minimum per-damage-class F1 | `>=0.35` | `0.387499` | PASS |

Disposition recorded by the pre-run contract: `EXTERNAL_DEVELOPMENT_VALIDATION_PASS_NOT_PRODUCTION_QUALIFIED`.

Binary damage-mask metrics: precision `0.673451`, recall `0.928655`, F1 `0.780727`, IoU `0.640322`. The high recall and lower precision mean the model still over-masks healthy pixels; it must not yet authorize reference-guided generation.

## Per-class validation

| Class | F1 | IoU |
|---|---:|---:|
| BLUR | 0.594079 | 0.422555 |
| MOTION_BLUR | 0.387499 | 0.240310 |
| PIXELATION | 0.731000 | 0.576044 |
| BLOCK_MOSAIC | 0.776206 | 0.634262 |
| JPEG_ARTIFACT | 0.776610 | 0.634802 |
| SCRIBBLE | 0.766284 | 0.621118 |
| STICKER | 0.564655 | 0.393393 |
| OPAQUE_BLOCK | 0.628597 | 0.458360 |
| BLACK_BAR | 0.861574 | 0.756812 |
| PARTIAL_OCCLUSION | 0.840401 | 0.724734 |
| MISSING_COMPONENT | 0.956129 | 0.915946 |
| HEALTHY, report-only | 0.982631 | 0.965856 |

Motion blur remains the weakest damage class.

## Domain diagnostics

| Domain | Macro-F1 | Macro-IoU | Minimum class F1 | Status against global floors |
|---|---:|---:|---:|---|
| female | 0.715022 | 0.578223 | 0.419628 | all floors met |
| male | 0.718119 | 0.581790 | 0.354845 | all floors met |
| African | 0.713942 | 0.573922 | 0.455160 | all floors met |
| Asian | 0.696471 | 0.563216 | 0.243086 | macro-F1 and min-class below floor |
| Caucasian | 0.738065 | 0.603014 | 0.491030 | all floors met |
| Indian | 0.720710 | 0.583870 | 0.402389 | all floors met |
| age 25 | 0.717099 | 0.579929 | 0.413746 | all floors met |
| age 50 | 0.706629 | 0.570660 | 0.322288 | min-class below floor |
| age 65 | 0.731612 | 0.596422 | 0.487861 | all floors met |

Per-domain thresholds were not part of the pre-run acceptance gate, so these observations do not retroactively change its overall PASS. They are blocking robustness evidence for any downstream RefFace or product claim and must not be tuned against this already-observed set.

## Runtime and safety

- checkpoint/ONNX first-batch argmax: exactly equal;
- maximum absolute logit difference: `8.48770e-5`;
- ONNX CPU total: `4.4710 s`; `0.005081 s` per 192x192 face;
- process RSS snapshot: `521,285,632` bytes under the 80% resource contract;
- wrong-person final pixels: 0; provenance violations: 0;
- output: mask logits only; restoration passes: 0; rollbacks: 0; abstentions: 0;
- RefFace: NOT_RUN; Windows: NOT_RUN; HP EliteBook: NOT_RUN; Target95: NOT_MEASURED.

## Classification and next action

Overall DEVELOPMENT gate: PASS. Domain robustness: NOT_QUALIFIED. Production: NOT_QUALIFIED. The checkpoint redistribution license remains non-explicit.

Do not tune on these 40 identities and do not run RefFace. Resolve the checkpoint licensing route and pre-register domain-aware acceptance on a new identity-disjoint real-photo validation bank before any reference-guided generation test.
