# InstantRestore feasibility audit for ConservativeFaceStudio

Status: RESEARCH / NOT QUALIFIED
Audit date: 2026-08-18
Target: HP EliteBook 1030 G3, Windows, 16 GB RAM, CPU-first, <=80% total-PC resources

## Why it remains high priority

InstantRestore is one of the closest published models to the real CFS Paper Quality use case:

- one severely degraded MAIN face;
- a small set of same-person references (upstream demo accepts 1-4; paper describes ~4);
- single-step restoration rather than a long diffusion sampling chain;
- shared-image attention transfers identity-specific reference information;
- no per-identity fine-tuning is required at inference.

This makes it scientifically more relevant to CFS personalization than adding another generic blind face restorer.

## Verified upstream implementation facts

Official repository: `snap-research/InstantRestore`.

The audited upstream implementation currently:

- defines the core as `pix2pix_turbo`;
- loads `stabilityai/sd-turbo` tokenizer + UNet;
- loads `stabilityai/sd-vae-ft-mse` VAE;
- constructs a second `original_unet` and second `original_vae` for the reference path;
- constructs a CLIP text encoder;
- explicitly calls `.to("cuda")` on both UNets and both VAEs;
- creates timesteps on CUDA;
- uses `.cuda()` for the text encoder and prompt tokens;
- uses `torch.float16`, CUDA device and CUDA autocast in the public Gradio inference path;
- publishes an environment pinned around CUDA 11.7 / Python 3.10.

Therefore **upstream CPU compatibility is NOT established**. CFS must not call the model CPU-compatible merely because PyTorch modules can theoretically run on CPU.

## Resource implication

The model is treated as one logical specialist by the CFS scheduler, but internally it contains multiple large neural networks that coexist during inference. The CFS 80% rule applies to the complete process/system footprint, not the number of Python model objects.

Before a real CFS CPU test can start, an isolated research adapter must:

1. remove hard-coded CUDA device placement without changing weights/math;
2. run `float32` or another CPU-supported dtype unless parity proves a safe alternative;
3. prevent internet downloads after a verified model pack has been prepared;
4. record the SHA-256 of the InstantRestore checkpoint and every upstream base-model asset;
5. check whole-system RAM before/after every large submodel load;
6. measure whether the duplicate reference UNet/VAE are simultaneously required during inference;
7. unload the complete specialist before the next heavy CFS model can start.

If projected whole-system RAM exceeds 80% on the 16 GB target, the adapter must refuse execution rather than swap/OOM.

## License blocker

The GitHub repository currently exposes no recognized repository license / top-level LICENSE in the audited root tree. That means:

- CFS may investigate the published research implementation;
- CFS must **not** assume redistribution or commercial permission for code/checkpoints;
- InstantRestore cannot become a redistributable `QUALIFIED` model-pack dependency until terms are clarified from an authoritative source.

This blocker is independent of technical quality.

## Qualification experiment (only after DamageMaskNet/reference-bank gate)

Use one identity-disjoint DEVELOPMENT case with:

- 512 aligned degraded MAIN;
- exactly four same-person references at distinct poses when available;
- same SFace identity gate as other Paper Quality candidates;
- CFS resource governor at 0.80;
- no CUDA;
- no final holdout;
- output labelled `GENERATED_MODEL_INFERRED`.

Measure:

- model-pack bytes and hashes;
- process/system RAM before load, after load, peak inference, after unload;
- load seconds;
- inference seconds;
- SFace against clean MAIN and accepted full-reference consensus;
- PSNR/SSIM/LPIPS when ground truth exists;
- component reference agreement for eyes/nose/mouth;
- generated fraction and healthy-region drift.

## Decision rule

Promote from `RESEARCH` to `BENCHMARKING` only if an actual CPU forward pass completes under the 80% total-PC cap.

Promote toward production only if it then beats the reference-first + GPEN/GFPGAN/CodeFormer alternatives on severe personalized cases by enough margin to justify dependency, runtime and licensing cost.

If the CPU vertical slice cannot be made to execute with at most three evidence-based implementation attempts, stop and move to the next specialist rather than weakening the 80% resource rule.
