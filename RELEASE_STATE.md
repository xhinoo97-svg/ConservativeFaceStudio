# Conservative Face Studio — Release State

- Branch: `feature/block-pipeline-v1`
- Last verified HEAD: `565f1ebc0af9660914fd7c0e40deb383a90aea60`
- Last fully green release commit: **none yet**
- Current first real blocker: **Windows build #1117 → Practical quality release gate**
- Classification: **ROOT_CAUSE**

## Verified gates

- Unit/integration tests: **PASS**
- Conservative validation suite: **PASS**
- CPU smoke benchmark: **PASS**
- Public practical runtime: **120 completed / 0 runtime errors**
- Extended degradation matrix runtime: **82 completed / 0 runtime errors**
- Practical quality: **FAIL — 0/83 target95-applicable cases >=95**

## Root cause from actual visual/block evidence

The remaining failure is not a missing model and not a runtime crash.

1. Sparse exact-canvas references are still rejected when their support is not sufficiently
   concentrated inside the small detected face ellipse, even when their photographed support
   lies entirely inside the frozen damage seed.
2. In cases where partial references *are* recovered, exact reference repair can succeed and
   then be rolled back because `observed_target_repair_runtime` resets pixels outside the
   current repair ROI to a runner-start anchor. That violates the accepted-context transaction
   model and triggers `outside_damage_change` in the core quality gate.

Example from the current artifact:
`peggy_whitson / face_scribble_ref2` recovered two partial references and produced 7,239 exact
reference pixels before Block 8 was rolled back by the outside-damage gate.

## Prepared logical fix batch

- Add a narrow seed-only exact-canvas verification fallback:
  - sparse donor only;
  - same image dimensions;
  - >=95% of clean observed support inside the frozen MAIN damage seed;
  - reference-damaged pixels excluded;
  - identity remains `not_enough_evidence`;
  - donor may not expand the damage seed.
- Make observed repair preserve the **current accepted context** outside its ROI instead of
  restoring a stale runner-start image.
- Add regression tests for both contracts.

Artifact audit against the current target95 dataset found **410/410** donor sheets satisfy the
new narrow exact-canvas geometry condition.

## Counterfactual validation on the current artifact

Using exact observed donor pixels on the accepted Block-7 context, the unchanged release metric
passes **83/83** target95-applicable cases; the lowest simulated score is approximately **96.67**.
This demonstrates that the blocker is evidence routing/transaction rollback, not the 95 threshold.

## Downstream

- Packaging: **BLOCKED**
- Installer: **NOT RUN**
- Offline installed-app validation: **NOT RUN**

## Next exact action

Create one logical commit from the verified current HEAD with the prepared quality fix. Update
the branch only after currently running workflows have completed, then let Windows CI run
uninterrupted to the next real release gate.
