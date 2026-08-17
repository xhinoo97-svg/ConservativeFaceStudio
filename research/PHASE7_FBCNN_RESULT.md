# Phase 7 — FBCNN JPEG specialist result

Date: 2026-08-17

## Scope

Development-only real CPU vertical slice. No V2/V3/V4 final holdout was used.

- Official source: `jiaxi-jiang/FBCNN`
- Pinned source commit: `54d1831927506b3247e2d4d245abb4f4dab1a1cd`
- Official release asset: `fbcnn_color.pth`
- Asset size: `287755111` bytes
- Observed SHA-256: `8b0e4ef23d59cf7ac934a342cb31a17619e4fa4a0b3374a9d78c5174312387e8`
- License recorded from upstream: Apache-2.0
- Input: legacy V1 DEVELOPMENT source `smartphone_02_lady`
- Test damage: single JPEG QF=20
- Runtime: PyTorch CPU, 80% CFS resource policy, 3/4 logical processors on the Actions host

## Measured result

| Metric | JPEG degraded | FBCNN | Direction |
|---|---:|---:|---|
| PSNR | 34.6184 dB | **36.7801 dB** | better |
| SSIM | 0.948646 | **0.963414** | better |
| SFace clean similarity | 0.957095 | **0.969095** | better |

FBCNN predicted input JPEG quality factor: approximately `31.04` for the aligned/cropped QF=20 test case.

Measured model load: approximately `0.703 s`.
Measured 512 inference after warm-up: approximately `7.646 s`.
Peak model-load RSS: approximately `956 MB`.
Peak inference RSS: approximately `1305 MB`.

The measured SFace score remains far above the frozen `0.363` identity threshold.

## Decision

`FBCNN_COLOR = LEADING_JPEG_CANDIDATE`, **not production QUALIFIED**.

Reason: unlike a generic face generator, the first specialist JPEG slice improved PSNR, SSIM and SFace simultaneously while preserving the visible face structure. This is exactly the type of damage-specific model the V2 architecture wants.

## Required next qualification

Before production promotion:

1. JPEG QF stratification, not one QF only.
2. double JPEG and social-media/screenshot recompression.
3. mixed JPEG + resize and JPEG + blur.
4. compare FBCNN-alone vs FBCNN-before-face-restorer.
5. Windows CPU execution.
6. HP EliteBook 1030 G3 timing/RAM.
7. common `FaceRestorerAdapter` lifecycle and one-heavy-model-at-a-time execution.
8. model-pack license/hash packaging review.

No final holdout may be used for these tuning decisions.
