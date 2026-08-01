# Fixed-five objective-mask terminology erratum

Frozen on `2026-08-01`, after job `369288` had produced the 35 compact
representation caches but before `metric_input.json`, `analysis.json`, or any
scientific audit value existed. Two independent subagents reproduced the pair
topology and reviewed the pinned upstream implementation. This addendum
clarifies terminology in the earlier value-blind continuation documents; it
does not alter the running audit or reinterpret the checkpoints it measures.

## Three distinct objectives

Every frozen replay batch has 128 presentations, balanced 32 per cancer/race
cell. The two global views produce 256 anchors, balanced 64 per cell. For each
anchor there are:

| Pair relation | Candidates |
|---|---:|
| same cancer, different race | 64 |
| different cancer, same race | 64 |
| same cancer, same race, excluding self | 63 |
| different cancer, different race | 64 |

The implemented `fair_supcon` uses every non-self pair, hence 255 denominator
candidates per anchor. It averages the log probability of each positive.

Official FAIR-Path commit
`04939510a50dbf9af7af72eb144e838259d192d5` excludes only pairs that are both
different-label and different-sensitive, hence 191 candidates per anchor in
this replay. It retains same-label/same-sensitive pairs and uses the log of
summed positive mass divided by the positive count:
`-(temperature / base_temperature) * log((sum_positive exp(logit)) /
(n_positive * sum_allowed exp(logit)))`. The function does not L2-normalize
features internally, and its dot-product path divides the dot product by the
square root of feature dimension before temperature scaling. Its constructor
defaults both temperatures to `0.07`; this records the API default, not a claim
about an unchecked upstream call site. Changing only the local mask is
therefore not an exact upstream reproduction. The pinned source is
[`loss.py`, lines 73--89](https://github.com/hms-dbmi/fairpath/blob/04939510a50dbf9af7af72eb144e838259d192d5/loss.py#L73-L89),
Git blob `905fb5d2f8fa1881dddae62d96f96e467da687f3`.

The proposed `relation_consistent_mask_v1` uses only the 64 same-cancer,
different-race positives and 64 different-cancer, same-race negatives, hence
128 candidates. It deliberately excludes the 63 same-cell pairs so that the
fairness term does not make them negatives while the separate cancer SupCon
term makes them positives. This is a handoff-faithful objective-conflict
ablation. It is not the published FAIR-Path denominator and must never be
called an exact FAIR-Path reproduction.

Its scalar is frozen as follows. Convert the representation to float, L2
normalize each row with `eps=1e-6`, and form pair logits as the dot product
divided by temperature `0.2`. For anchor `i`, define

- `positive(i,j) = same_cancer(i,j) and different_race(i,j) and i != j`;
- `negative(i,j) = different_cancer(i,j) and same_race(i,j) and i != j`; and
- `allowed = positive or negative`.

The log denominator is `logsumexp` over `allowed` pairs only. The per-anchor
fair loss is the negative mean of the positive log probabilities, preserving
the current implementation's mean-of-log-positive reduction. Average valid
anchors equally. An anchor is valid only when it has at least one positive and
one negative; omit invalid anchors and return a differentiable zero when none
are valid. Unknown cancer/race rows are excluded before forming pairs. The
cancer SupCon term is unchanged and total loss remains
`cancer + 0.1 * relation_consistent_mask_v1`. No downstream outcome is an
argument.

## Authoritative result-contingent branch

The later 128-candidate specification in
`FIXED5_FRESH_CONTINUATION_REVIEW.md` is authoritative for the last branch of
`FIXED5_REPRESENTATION_AUDIT_LOCK.md`. Activity and preservation are evaluated
separately. A race-activity subgate at one probe level covers four strata (two
cancers by two views). It passes only when, in every stratum, at least four of
five seeds and the five-seed median reduce oriented leakage by at least `0.05`.
The preservation gate covers two pooled cancer/view cells; both require
cancer-AUROC loss no worse than `0.02` in at least four of five seeds and at the
median. Here `L = max(AUROC, 1 - AUROC) - 0.5`; missing cells fail.

Branch precedence is frozen and evaluated in this exact order:

1. If final H passes both race-activity subgates plus H/B preservation, use H.
2. Otherwise, if final P passes both race-activity subgates plus P/B
   preservation, use P. Thus H is selected deterministically if both pass.
3. Otherwise, if either final layer has race activity but its preservation
   fails, stop for utility harm.
4. Otherwise, if fair E has patient or tile race activity, run
   carry-versus-fresh if E_fair/E_plain preservation passes; otherwise stop for
   harm.
5. Otherwise, if temporary fair A has patient activity, run carry-versus-fresh
   if A_temp_fair/A_temp_plain preservation passes; otherwise stop for harm.
   This explicitly includes patient-pass/tile-fail.
6. Otherwise, if temporary fair A has tile activity, run patient-mean training
   if its preservation passes; otherwise stop for harm.
7. Otherwise, temporary fair A has neither race activity. Run
   `relation_consistent_mask_v1` only if its preservation passes; otherwise
   stop for harm.

This removes any after-value choice between carry, granularity, masking, and
utility failure. The mask branch tests only post-hoc timing first, because it
is the one-stage, frozen-encoder intervention and the user capped new runs.

The complete fixed-five audit must be opened to select branch 7. After it does,
apply the following feasibility predicates using only the already-fixed B/H
control values; no H-mask candidate exists or is opened yet:

- for seed `32001`, all eight B leakage values must be at least `0.05`, all
  eight H leakage values must be strictly positive, and the median of those
  eight H values must be at least `0.03`;
- for each of the eight race strata across all five seeds, at least four B
  leakage values and the median B leakage must be at least `0.05`; and
- for each stratum, at least four H leakage values must be strictly positive
  and the median H leakage must be at least `0.03`.

If any predicate fails, record inadequate headroom and do not train a mask,
substitute a seed, or relax the gate. Otherwise reuse accepted FM seed `32001`,
its immutable replay, accepted plain encoder, fresh H adapter initialization,
and existing B/H controls. One `main_1gpu` allocation runs exactly one new
stage-training execution, `H_mask_32001`, at fixed `lambda=0.1`, matched to H
in adapter topology, schedule, exposure, and optimizer. Do not combine this
with adapter carry, a dose change, P timing, the 191-candidate official mask,
or another method.

The seed-32001 gate contains those eight race cells and two pooled cancer-AUROC
cells:

- every race cell must satisfy `L(B) - L(H_mask) >= 0.05`;
- every race cell must satisfy `L(H_all_pairs) - L(H_mask) > 0`, and the median
  of those eight paired reductions must be at least `0.03`; and
- both cancer cells must satisfy `AUROC(B) - AUROC(H_mask) <= 0.02`.

Downstream-task evaluation may veto an eligible representation package, but
cannot rescue a failed representation gate. After the H-mask representation is
hashed and frozen, evaluate B and H-mask with the four fixed diagnostic head
seeds and outer folds. Ensemble the four head probabilities first, then compute
nested iEO and utility; do not average four independently computed metric
values. The one-seed target safety gate requires:

- in each cancer, `iEO(H_mask) - iEO(B) <= +0.03`;
- across the equal-weight mean of BRCA and LUAD, AUROC strictly greater than
  `0.60`, AUROC and Black-AUPRC changes versus B each at least `-0.02`, and
  Black-ECE change versus B no greater than `+0.02`; and
- in each cancer, AUROC strictly greater than `0.57`, AUROC and Black-AUPRC
  changes versus B each at least `-0.05`, and Black-ECE change versus B no
  greater than `+0.05`.

All comparisons use Amendment 07's finite-double policy (`abs_tol=1e-12`,
`rel_tol=0`). If the target fails any representation, utility, or harm gate,
stop without substituting another seed.

Seed `32001` passes only if it clears both the representation and one-seed
safety gates above. If and only if it passes both, a
second serial `main_1gpu` allocation runs `H_mask` completely for existing
seeds `32002`, `32003`, `32004`, and `32005` in that order, with no value
opening or early stop between them. Each path is bound to that seed's accepted
plain encoder, immutable replay, fresh H adapter initialization, H schedule,
exposure, optimizer, and existing B/H controls. The final five-seed
confirmation requires, in every one of the eight race strata,
`L(B) - L(H_mask) >= 0.05` in at least four seeds and at the median; positive
`L(H_all_pairs) - L(H_mask)` in at least four seeds with median at least
`0.03`; and, in both cancer/view cells, preservation within `0.02` in at least
four seeds and at the median. Downstream utility and harm gates still apply.
The completed five-seed package uses the original fixed-five aggregate
utility/harm formulas and probability-first four-head ensembling, implemented
and checked by a new source-bound analyzer and independent verifier written
before confirmation values are opened.

The exact cardinality ceiling is:

| Quantity | Maximum |
|---|---:|
| new unique FM seeds | 0 |
| reused accepted FM seeds | 5 |
| new stage-training executions | 5 |
| new fairness-bearing paths | 5 |
| result-contingent packages | 1 target + 1 fixed confirmation |
| Slurm allocations | 2, never concurrently |

Every allocation is named `main_1gpu` and uses one GPU. There are no p-value
gates, optional seeds, or sequential significance stops; a scientific failure
is not rerun under another seed. No official-mask, dose, P-timing, or method
sequence is authorized within this continuation.

## Interpretation boundary

The current zero-new-seed audit remains a valid test of the checkpoints trained
with the 255-candidate all-pairs implementation. A negative result means that
specific package was inactive under the frozen effect-size gate. It does not
show that `relation_consistent_mask_v1`, official FAIR-Path, or FairCon in
general is inactive.
