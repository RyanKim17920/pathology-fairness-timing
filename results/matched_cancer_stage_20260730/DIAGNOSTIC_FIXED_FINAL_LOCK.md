# Matched-cancer fixed-final diagnostic lock

Status: preregistered before opening any real diagnostic prediction, label, or
outcome file. The analyzer and independent verifier are developed only against
synthetic fixtures. Calibration representation outcomes cannot enter this
diagnostic analysis.

## Fixed study

The inferential units are exactly 48 paired foundation-model seeds, in immutable
order `32001` through `32048`. Failed production jobs are rerun at the same
seed; seeds are never substituted or added. There is one analysis after all 48
pairs are complete. There are no interim looks, optional stopping, adaptive
sample size, or result-dependent head/target selection.

The arms are:

- `B`: plain encoder, plain final adapter;
- `P`: race fairness in pretraining, plain final adapter; and
- `H`: plain encoder, race fairness in the final adapter.

The two targets are `BRCA` and `LUAD`, with fixed complete cohorts of 334 and
281 patients respectively. Every cell uses the same four final-head seeds
`42001`, `42002`, `42003`, and `42004`. The fixed head halves are
`{42001,42002}` and `{42003,42004}`. Heads reduce optimization noise and never
increase the inferential sample size.

## Sealed prediction-row contract

The analyzer has no default real-data location. It accepts one explicit JSONL
whose rows use schema `matched-cancer-diagnostic-prediction/v1` and exactly the
fields:

`schema, fm_seed, arm, cancer, head_seed, patient_id, y_true, race, fold,
role, outer_fold, inner_fold, probability`.

Allowed races are `Black` and `White`; outcomes are binary; probabilities are
finite in `[0,1]`; folds are exactly `0..4`. Each patient/cell/head has exactly
one `outer_test` row at its own fold and four `inner_calibration` rows, one for
each other outer fold. Patient IDs and outcome/race/fold metadata must match
across arms, heads, and all 48 FM seeds. Extra/missing fields, rows, cells,
patients, or seeds invalidate the study.

The raw prediction file SHA-256 and row count are part of the semantic report.

## Nested-calibration iEO

For every FM seed, arm, cancer, row role, patient, and fold context, average
the four head probabilities first. Thresholds are never fit separately by head
and never averaged after classification.

For each held-out outer fold and each White-specificity target
`{0.60, 0.625, ..., 0.95}`, use NumPy's linear empirical quantile of the nested
White-negative calibration probabilities as the arm-specific threshold.
Classify a held-out score positive when `score >= threshold`. Pool the five
held-out folds and define

`EO_s = max(|FPR_Black-FPR_White|, |TPR_Black-TPR_White|)`.

The cancer endpoint `iEO` is the unweighted mean of the 15 `EO_s` values. Empty
White-negative calibration sets, held-out folds, or pooled Black/White outcome
denominators invalidate the study. The report records, for every full-ensemble
cell, specificity and outer fold, the threshold, White-negative calibration
count, held-out White-negative count, and achieved held-out White specificity.

The two cancer iEO values are averaged equally within each FM seed. Define

`theta_seed = iEO(P)_seed - iEO(H)_seed`.

Positive theta favors `H`; negative theta favors `P`.

## Fixed-final inference and precedence

All inference is paired over the 48 `theta_seed` values using a Student
one-sample paired-FM analysis with 47 degrees of freedom.

Practical equivalence is tested first and takes precedence. It is established
only when the paired 90% confidence interval is strictly inside
`(-0.03,+0.03)`. Touching either boundary is not equivalence.

If equivalence fails, directional superiority requires every gate:

1. two-sided paired-FM `p < 0.05`;
2. observed `|mean(theta)| >= 0.02`;
3. both fixed two-head halves have a strict mean direction matching the
   four-head result;
4. both BRCA and LUAD mean theta values have that strict direction (`2/2`);
5. in each cancer, the favored arm's mean iEO minus `B` mean iEO is no greater
   than `+0.03`; and
6. the favored arm passes all utility gates below.

If neither equivalence nor every superiority gate passes, the result is
`inconclusive`. A favorable point estimate, nominal p-value, individual head,
or individual cancer cannot be promoted to a claim.

## Utility qualification

Utility is computed from the four-head-averaged outer-test probabilities.
Overall AUROC uses the standard tie-adjusted rank definition. Black AUPRC uses
the threshold-group average-precision definition. Black ECE uses ten
equal-width probability bins.

For the directionally favored arm, averages are equal-weighted across BRCA and
LUAD after averaging across the 48 FM seeds. All mean gates must pass:

- overall AUROC `> 0.60`;
- AUROC change versus `B >= -0.02`;
- Black AUPRC change versus `B >= -0.02`; and
- Black ECE change versus `B <= +0.02`.

No cancer may violate a bound by more than another `0.03`: cancer AUROC must be
`> 0.57`, cancer AUROC and Black-AUPRC changes must each be `>= -0.05`, and
cancer Black-ECE change must be `<= +0.05`.

## Independent verification

The analyzer and verifier are separate implementations. The verifier does not
import or execute the analyzer. It reopens the sealed prediction JSONL,
independently checks completeness and metadata identity, averages probabilities,
fits thresholds, recomputes iEO, utility, paired inference, every gate, and the
complete nested-threshold audit trail. It then requires complete semantic
agreement with the analyzer report, including the raw prediction SHA-256.

Only an analyzer report accepted by this independent verifier is reportable.
