# Matched-cancer fixed-five numeric amendment 07

Frozen at `2026-07-30T22:03:45Z`, after independent synthetic analyzer
red-team and before any scientific value or result was opened. Seed 32001 had
completed calibration and had only begun value-blind diagnostic setup.

This amendment defines floating-point boundary semantics for all fixed-five
analysis and verification rules. It supplements:

`results/matched_cancer_stage_20260730/fixed5_execution/FIXED5_EXECUTION_AMENDMENT_06.md`.

Amendment 06 is preserved byte-for-byte:

- bytes: `3,609`;
- SHA-256:
  `054629cb1ac5db1957382d8401b62a8e5d71418da2c862e3563ecfbd3a87fade`.

No scientific threshold, margin, endpoint, or direction changes.

## Canonical comparison policy

Every comparison to a locked numeric boundary uses:

- absolute tolerance: `1e-12`;
- relative tolerance: `0.0`; and
- finite IEEE-754 double-precision inputs.

For observed value `x` and boundary `b`:

- inclusive `x >= b` passes when `x > b` or
  `isclose(x, b, rel_tol=0.0, abs_tol=1e-12)`;
- inclusive `x <= b` passes when `x < b` or
  `isclose(x, b, rel_tol=0.0, abs_tol=1e-12)`;
- strict `x > b` passes only when `x > b` and the values are not close;
- strict `x < b` passes only when `x < b` and the values are not close; and
- `x` is treated as zero for sign counting when
  `isclose(x, 0.0, rel_tol=0.0, abs_tol=1e-12)`.

Thus a binary representation infinitesimally below mathematical `0.02`
passes an inclusive `>=0.02` materiality gate when it is within tolerance,
while a value infinitesimally inside mathematical `0.03` does not pass a
strict `<0.03` gate when it is within tolerance of the boundary.

The policy applies uniformly to:

- mean and median materiality;
- per-seed practical margins;
- 90% confidence-interval equivalence boundaries;
- mean, median, leave-one-out, cancer, and head-half directions;
- baseline harm gates;
- every overall and cancer-specific utility gate; and
- exact sign-test zero/nonzero classification.

Raw computed values and interval endpoints remain unrounded in the report.
The report records the tolerance and each Boolean gate. Display rounding never
enters a decision.

## Independent verification and tests

The analyzer and independent verifier implement the policy separately; the
verifier does not import the analyzer or a shared comparison helper.

Synthetic tests cover every locked boundary at:

- the mathematical boundary represented through subtraction;
- one value clearly inside/passing;
- one value clearly outside/failing; and
- positive and negative directions where applicable.

Tests must include `0.02`, `-0.02`, `0.03`, `-0.03`, zero/sign classification,
strict AUROC `0.60` and `0.57`, inclusive AUROC/AUPRC deltas `-0.02` and
`-0.05`, and inclusive ECE deltas `+0.02` and `+0.05`.

The fixed-five source manifest, analyzer report, independent verification,
and final completion receipt bind this amendment in addition to Amendments 01
through 06.
