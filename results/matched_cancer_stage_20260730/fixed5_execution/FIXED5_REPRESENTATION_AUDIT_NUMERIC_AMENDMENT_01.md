# Fixed-five representation-audit numeric amendment 01

Frozen on `2026-08-01`, before any representation-audit metric was computed
or opened. This amendment resolves implementation details left implicit in
`FIXED5_REPRESENTATION_AUDIT_LOCK.md`; it does not change the population,
checkpoints, seed budget, or result-contingent branch.

## Representation normalization and tile identity

Every 384- or 128-dimensional tile representation is L2-normalized per tile
before pooling, probing, or geometry. A zero or non-finite norm fails closed.
Patient representations are arithmetic means of the 16 normalized tile
vectors in the relevant view; they are not normalized a second time for the
probe, whose training-only standardizer remains part of the frozen pipeline.

`occurrence_index` is the zero-based position of a tile within that patient's
full deterministic cache/input order, before invalid entries are removed.
Only occurrences whose existing validated cache `keep_mask` is true enter the
hash ranking. The selected payload hash must match the payload read for frozen
inference. Any order, validity, or payload disagreement fails closed.

## Pooled cancer-information probe

BRCA and LUAD have disjoint TSS values, so leaving out a single TSS would
produce a one-class cancer test fold. Instead, form five deterministic grouped
outer folds as follows. Within each cancer separately, rank its TSS values by
SHA-256 of

`rep-audit-cancer-fold/v1|288850999|cancer|tss`

and allocate ranked TSS values round-robin to folds `0..4`. Each outer fold
therefore holds out whole TSS groups from both cancers. Use patient-mean
representations, train-only standardization, and the same logistic-regression
solver seed and C grid as the race probes. Select C within each outer training
set using the other four grouped fold labels as inner folds; exclude invalid
inner folds and require at least two. Pool each patient's one held-out cancer
probability and report AUROC. This probe is computed separately by seed,
layer, and tile view.

## Secondary geometry

Cross-race cosine mixing uses `k=5`. The patient statistic finds neighbors
among other patients in the same cancer. The tile statistic finds neighbors
among tiles in the same cancer and excludes every tile belonging to the query
patient. For either statistic, compute the opposite-race neighbor fraction per
query, average within query race, then average the Black and White values.

Cancer-conditioned energy distance uses Euclidean distance on patient means
and the unbiased U-statistic form: twice the mean cross-race distance minus
the two within-race means over distinct ordered pairs. Aligned representation
displacement reports the mean Euclidean distance and mean cosine distance over
the same selected tile identities and, separately, their patient means. Only
equal-dimensional layers are compared. Parameter displacement analogously
reports absolute Frobenius norm and the ratio to the baseline-state Frobenius
norm over aligned floating-point tensors; it is descriptive and never a gate.

## Exact gate aggregation

The four gate-eligible candidate/baseline contrasts are:

- `E_fair` versus `E_plain`;
- `A_temp_fair` versus `A_temp_plain`;
- final `P` versus final `B`; and
- final `H` versus final `B`.

Final `P` versus `A_temp_fair` is washout evidence only and cannot pass the
primary gate by itself.

For every seed/cancer/view/probe-level cell define leakage reduction as
`L_baseline - L_candidate`. A contrast passes race-leakage activity only when,
for every cancer, both views, and both patient/tile levels, at least four of
five seed reductions are at least `0.05` and their median is at least `0.05`.
This is the exact meaning of seed and median agreement; there is no p-value
stopping rule.

For the pooled cancer probe define information loss as
`AUROC_baseline - AUROC_candidate`. A contrast preserves cancer information
only when, in both tile views, at least four of five seed losses are no greater
than `0.02` and the median loss is no greater than `0.02`. A missing or
non-finite required value fails. The complete primary mechanism gate is the
conjunction of race-leakage activity and cancer-information preservation.

The frozen continuation budget remains at most four new FM seeds total: one
targeted mechanism seed only if the branch requires it, followed only after a
pass by exactly three serial replication seeds. Thus no analysis can request
48 seeds, and no study job may request more than one GPU or use a name other
than `main_1gpu`.
