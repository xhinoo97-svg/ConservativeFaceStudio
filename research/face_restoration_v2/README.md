# Conservative Face Studio V2 research

This directory is an isolated research workstream. It must never be imported by
the V1 application or included in the V1 Windows installer.

## Frozen baseline

- V1 merge: `2767513f95dde2d417e7c6f1faf2357149a1a32f`
- YuNet, SFace, face parsing, head pose and LaMa V1: frozen
- NAFNet V1 checkpoint: immutable baseline
- V1/V2/V3 release holdouts: consumed; prohibited for V2 development

## First milestone

1. Register legally reusable clean portraits with stable identity IDs.
2. Enforce identity-disjoint train, validation and hidden-holdout splits.
3. Generate deterministic face-targeted degradations from clean images.
4. Run the certified V1 pipeline on the new V2 validation split.
5. Select one NAFNet Face Deblur V2 hypothesis from baseline evidence.

No model training or production integration is part of this milestone.

