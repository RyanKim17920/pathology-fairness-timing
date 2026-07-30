# Matched-cancer fixed-five sample-size amendment 01

Frozen at `2026-07-30T21:11:02Z`.

Status: sample-size and interpretation amendment frozen after the user capped
the study at five foundation-model seeds, and before any real diagnostic
prediction, diagnostic outcome, arm comparison, calibration metric value,
analyzer output, or inferential result was opened. At freeze time, seed 32001
was still in calibration. Monitoring had inspected only scheduler state,
artifact presence, and line counts.

This amendment supersedes only the fixed-48 production population, final
matrix cardinality, and strength of inference in:

- `results/matched_cancer_stage_20260730/DIAGNOSTIC_FIXED_FINAL_LOCK.md`;
- `results/matched_cancer_stage_20260730/DIAGNOSTIC_FIXED_FINAL_AMENDMENT_01.md`;
  and
- `results/matched_cancer_stage_20260730/fixed48_execution/FIXED48_EXECUTION_PROTOCOL.md`.

The estimand, B/P/H arms, BRCA/LUAD cohorts, downstream-label firewall,
representation and adapter schedules, paired initialization, diagnostic
heads, nested calibration, `theta` direction, practical thresholds, utility
gates, harm gates, and independent-verification requirements remain
unchanged.

## Fixed population and execution

The production population is exactly the first five previously locked
foundation-model seeds:

`32001, 32002, 32003, 32004, 32005`.

Seeds are processed in that order. A failed execution is retried at the same
seed in a new immutable attempt; no seed is substituted. Seeds 32006 through
32048 are excluded and must not be submitted under this amendment. The study
is not extended in response to any observed result.

The already-running seed-32001 canary remains eligible only if its success
receipt passes the unchanged fixed-48 V2 source, V3 authorization, and V2
feasibility controls. Those bound files remain immutable. Seeds 32002 through
32005 reuse that exact per-seed scientific implementation through an additive,
versioned fixed-five controller.

There is at most one queued or running matched-cancer study GPU allocation
across both the fixed-48 and fixed-five scheduler comments. Every study
allocation is named `main_1gpu`, requests exactly one GPU, uses no array or
dependency fan-out, and runs seeds serially.

## Fixed cardinality

The complete fixed-five prediction matrix has exactly:

`5 FM seeds × 3 arms × 4 heads × 5 nested rows × (328+281 patients)`

= **182,700 prediction rows**.

It contains exactly `5 × 3 × 2 × 4 = 120` seed/arm/cancer/head cells. The
nested-threshold audit contains exactly
`5 × 3 × 2 × 15 × 5 = 2,250` records. Heads, cancers, patients, folds, and
specificity targets are repeated measurements and never increase the five
independent FM-seed units.

## Effect-size screen

For every seed, compute the unchanged primary paired effect:

`theta_seed = mean_cancer(iEO(P)) - mean_cancer(iEO(H))`.

Positive values favor H and negative values favor P. The final report must
show all five `theta_seed` values, their mean, median, sample standard
deviation, minimum, maximum, 90% and 95% Student intervals with four degrees
of freedom, and all five leave-one-seed-out means. It must also show the two
fixed head-half directions, the BRCA and LUAD directions, baseline harm gates,
and all utility gates.

The primary fixed-five conclusion is a practical engineering screen:

1. **large/stable practical effect favoring H or P** requires all of:
   - `abs(mean(theta_seed)) >= 0.02`;
   - at least four of five nonzero seed effects have the mean direction;
   - every leave-one-seed-out mean has that same strict direction;
   - both fixed head halves and both cancers have that strict mean direction;
   - the favored arm passes every unchanged baseline-harm and utility gate.
2. **small across these five tested seeds** requires all of:
   - `abs(mean(theta_seed)) < 0.02`;
   - every `abs(theta_seed) < 0.03`; and
   - the complete five-seed 90% Student interval is strictly inside
     `(-0.03, +0.03)`.
3. Every other valid outcome is **unstable/insufficient**.

The report may additionally reproduce the original paired Student
equivalence/superiority calculations with `df=4`, but these are secondary and
cannot override the fixed-five screen. A two-sided exact sign/randomization
test over five paired seed effects has minimum attainable p-value `0.0625`;
heads, cancers, folds, specificity targets, and patients must not be
pseudoreplicated to manufacture a smaller p-value.

“Small across these five tested seeds” is deliberately bounded to the tested
models. Nonsignificance alone is not evidence of no effect, and none of the
three screen labels establishes broad population equivalence or superiority.

## Blinding and final look

Production monitoring remains outcome-blind. Only after all five independently
verified seed-success receipts exist may the fixed-five collector emit the
sealed 182,700-row matrix. The analyzer runs once, followed by a separate
verifier that reopens the raw matrix and independently recomputes every
quantity and conclusion.

There is no interim scientific look, optional extension, seed replacement,
result-dependent endpoint choice, or submission of the fixed-48 remainder.
