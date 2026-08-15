# Conservative Face Studio — Release State

## Current certification

- Branch: `feature/block-pipeline-v1`
- Pull request: [#1](https://github.com/xhinoo97-svg/ConservativeFaceStudio/pull/1) → `main`
- Certified candidate PR HEAD: `348c4d787a73d7b33cd339c0c14e29a4e162e067`
- Certified PR merge SHA: `7d826eaec32ff633df4bef1242be56484c4d25b7`
- Base `main` SHA: `5eff667373cd47c07ba14aaad2acafee6d5a61c1`
- `PRODUCT_COMPLETE_PRE_TUNING`: **TRUE**
- `ARCHITECTURE_FROZEN`: **TRUE**
- `RELEASE_CANDIDATE_QUALIFIED`: **TRUE**
- `RELEASE_READY`: **FALSE — state-only reconciliation CI must pass before merge**
- `REAL_PC_ACCEPTANCE`: **PENDING**
- TARGET95 policy: **REPORT ONLY**
- Current required gate: **STATE_RECONCILIATION_CI**

This state-only reconciliation does not change production code, models, thresholds, benchmark data,
or routing. The PR must not be merged until all required workflows triggered by this commit are
green on the resulting PR HEAD.

## Same-candidate workflow evidence

All required candidate workflows completed successfully:

- Windows build [#1194](https://github.com/xhinoo97-svg/ConservativeFaceStudio/actions/runs/31845859781): **SUCCESS**
- Female-domain benchmark [#462](https://github.com/xhinoo97-svg/ConservativeFaceStudio/actions/runs/31845859790): **SUCCESS**
- Release quality v2 [#12](https://github.com/xhinoo97-svg/ConservativeFaceStudio/actions/runs/31845859806): **SUCCESS**

GitHub records all artifacts above against candidate PR HEAD
`348c4d787a73d7b33cd339c0c14e29a4e162e067`. Workflow-generated evidence records the tested
merge SHA `7d826eaec32ff633df4bef1242be56484c4d25b7`.

## Release-quality evidence

Artifact `release-quality-v2-12`:

- Artifact ID: `9237031662`
- ZIP SHA256: `e36b1fcdbba2f6e0ae74e162e05e32a7aae7e135ef74671a8850d01b8a2a2d59`
- Calibration: **60/60 hard guardrails**, 0 runtime errors, 0 invalid provenance cases,
  0 wrong-person final pixels.
- Independent final holdout: **40/40 hard guardrails**, 0 runtime errors,
  0 invalid provenance cases, 0 wrong-person final pixels.
- Final holdout protocol: frozen before candidate evaluation, independent from consumed v1/v2
  source identities and clean-image checksums, candidate frozen after calibration and before the
  one-shot holdout run, and not used for tuning.
- Calibration report SHA256:
  `ebda0d3a6823c39185e59ed51593cc82dd497da62cdbf021d5b92ef0dfbdb947`
- Final-holdout report SHA256:
  `5a7e3d2c84905db0589b9e28799c9dcc97468b7b3ffd3d0028103706f9107ea1`
- Frozen v3 source manifest SHA256:
  `9310d78082b6a33bf79d7497f6823443ac922ee15576bf7dc2ec15bf17f9662a`

The earlier v1 (39/40) and v2 (38/40) holdouts remain consumed historical evidence and were not
reused to certify this candidate.

## Windows deliverables

Windows artifact metadata is same-candidate and unexpired:

- Validation: `9236572348`
- Production model updates: `9236565718`
- Release metadata: `9236563640`
- Portable Windows x64: `9236563205`
- Setup Windows x64: `9236560981`

Verified Windows results:

- complete tests and validation: **PASS**
- production model smoke: **6/6 PASS**
- failure injection: **19/19 PASS**
- reference counts 0 through 9: **PASS**
- portable package: **PASS**
- installer: **PASS**
- installed application: **PASS**
- offline installed application: **PASS**
- practical runtime: 120 completed, 0 runtime errors
- extended degradation matrix: 82 completed, 0 runtime errors

Deliverable checksums:

- `ConservativeFaceStudio-Setup-x64.exe`:
  `f032b17bb361c1eb4f5a58e615667abdb240e103699f9a0b39d70db91ed569a0`
- `ConservativeFaceStudio-Windows-x64.zip`:
  `47ec80efd04688047dcb70133f7d45486962c5c43df2a9d9a20191098f7b1a65`

The six-model registry SHA256 is identical in Windows and Release Quality:
`95c172ff825cb7c2d3f8024a8f450bf548d4b5231b3e35d6e798d1251d5ac234`.

## Release boundary

CI validates the Windows build and offline installed package on a GitHub-hosted Windows runner.
That is not a test on the user's physical PC. Therefore `REAL_PC_ACCEPTANCE` remains **PENDING**
and must not be represented as complete.

## Next exact action

Wait for every required workflow triggered by this state-only commit. If all required checks pass
on the same resulting PR HEAD, merge PR #1 without another state-file edit. If a required check
fails, diagnose its first failing step and apply only the minimum evidence-backed correction.
