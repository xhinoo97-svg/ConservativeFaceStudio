# Evaluation plan

ConservativeFaceStudio evaluates two different goals separately: image restoration quality and identity preservation. A visually sharper image is not automatically a better conservative restoration.

## Core metrics

For paired synthetic tests where a clean reference exists:

- PSNR: pixel reconstruction fidelity.
- SSIM: structural similarity.
- LPIPS: perceptual difference; optional because it adds a learned model dependency.
- Landmark distance: geometric drift of eyes, nose, mouth and face contour.
- Face-ID cosine similarity: identity consistency when a legally usable recognition model is available.

For real photographs without ground truth:

- identity similarity against the user's other reference photographs;
- landmark consistency across aligned references;
- percentage of final pixels copied from each observed source;
- percentage of generated pixels (must be zero in strict mode);
- occlusion-mask coverage;
- per-block image hashes and provenance;
- manual review of unsupported-detail risk.

## Important benchmark finding

The EDFace blind-face-restoration benchmark paper evaluates blur, noise, low resolution, JPEG artifacts and combined degradation, and includes Average Face Landmark Distance (AFLD) and Average Face ID Cosine Similarity (AFICS). These two task-driven metrics map directly to this project's conservative objective and should be preferred over perceptual sharpness alone.

## Dataset policy

Do not vendor or redistribute benchmark image datasets from this repository unless redistribution is explicitly allowed.

### CelebA

Official terms restrict CelebA to non-commercial research and prohibit further distribution of the images. The repository must therefore contain only an adapter/instructions, never the images.

### FFHQ

The FFHQ dataset package is CC BY-NC-SA 4.0 and individual photographs carry their own Flickr/CC terms. It is suitable only when those terms fit the evaluation use. Do not bundle it.

### WIDER FACE

Useful for testing face detection under pose, scale and occlusion, not as a ground-truth restoration dataset. Distribution terms must be reviewed from the official source before automated acquisition.

### XQLFW / LFW derivatives

Useful for cross-quality identity robustness. Do not assume that the original LFW terms automatically cover every derivative; each benchmark's distribution terms must be checked separately.

## Synthetic regression set

The default CI should continue using generated images and user-free synthetic degradations because they are legally simple, deterministic and reproducible. Recommended degradations:

1. Gaussian and motion blur;
2. Poisson/Gaussian noise;
3. downscale + JPEG compression;
4. rectangular and irregular occlusions;
5. exposure shifts;
6. small affine transforms;
7. combinations of the above.

Each deterministic block should be tested for shape, dtype, finite values, provenance and bounded geometric/photometric drift. Generative modules require separate tests and must never be part of strict-mode acceptance criteria.
