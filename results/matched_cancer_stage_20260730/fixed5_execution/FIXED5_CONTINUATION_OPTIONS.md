# Fixed-five value-blind continuation options

Frozen at `2026-07-30T23:01:25Z`, before the fixed-five analyzer or any
scientific result was opened.

This plan was proposed by a fresh independent subagent that reviewed the
design, implementation, `FAIRPATH_HANDOFF.md`, and available checkpoint
topology without inspecting any real prediction, probability, calibration
value, scientific metric, outcome, analyzer report, or arm comparison.

It is a bounded contingency plan, not authorization to run every branch. It
exists before the final look so a small, negative, or unstable result does not
trigger an improvised “run more seeds” response.

## Primary mechanistic ambiguity

The straight-matched study controls architecture, initialization, exposure,
and cancer/race inputs, but it is not persistence-matched:

- P applies FairCon through the temporary Slot-1 adapter, then discards that
  adapter and trains a fresh cancer-only final adapter for 781 steps.
- H applies FairCon directly to the retained final adapter.

Therefore an H advantage or small P-H difference could reflect fairness
localization in P's discarded projection adapter, or washout through its fresh
adapter, rather than an intrinsic disadvantage of encoder-time fairness.

Other value-blind hypotheses are:

- nominal `lambda=0.1` produces different realized optimization doses when P
  can move E+A but H can move only A at a different stage learning rate;
- the pair-denominator construction differs from the published FAIR-Path NDL
  mask and may dilute the fairness signal;
- tile-level fairness may not survive patient-level mean pooling; or
- TP53 may have inadequate baseline disparity/headroom for either placement.

## Sequential funnel

Across all new-training branches below:

- at most one study GPU is queued or running;
- no optional extension or run-until-significance is allowed;
- branches 3, 4, and 5 are alternatives, not an automatic bundle;
- there are at most two one-seed exploratory blocks; and
- there is at most one fixed three-seed untouched-scenario confirmation.

Thus the entire continuation uses at most five new FM seeds.

### 1. Existing-checkpoint mechanism and headroom audit

Budget: zero new FM training; at most one serial embedding/cache pass.

Using only cancer, race, TSS, patient, and pixels on a patient-disjoint,
target-TSS-excluded audit population, measure at the pretrained E, Slot-1
plain/fair E, Slot-1 temporary A(E), and final B/P/H A(E):

- within-cancer leave-one-TSS-out race-probe AUROC;
- cancer-probe AUROC;
- cancer-conditioned MMD or energy distance between races;
- cross-race nearest-neighbor mixing;
- encoder and adapter state displacement; and
- accumulated fair/main gradient-norm ratios and gradient cosine.

Only after the authorized final look, use the already-produced B endpoint to
classify headroom. Do not rerun or retune the fixed-five analyzer.

Interpretation:

- Fair signal in temporary A but not E: projection-head localization.
- Signal in fair E but absent from final P A(E): fresh-adapter washout.
- No signal in temporary fair A: objective, dose, or granularity failure.
- Race leakage changes but TP53 iEO does not: proxy non-transfer or endpoint
  floor.
- Inadequate B utility or B iEO below 0.03 in both cancers: stop optimizing
  TP53 and preregister an untouched scenario with valid headroom.

### 2. Persisted-adapter matched analysis

Budget: zero new FM training; at most one serial diagnostic/cache job.

Use existing checkpoints:

- `B_keep`: Slot-1 plain E plus its Slot-1 adapter;
- `P_keep`: Slot-1 fair E plus its Slot-1 fair adapter; and
- `H_keep`: Slot-1 plain E plus the retained Slot-2 fair adapter.

P and H then expose the diagnostic probe to the adapter that actually received
781 fairness-stage steps. This is mechanistic/exploratory, not an independent
confirmation.

Advance only if both cancers and both fixed probe-seed halves agree, race-probe
AUROC improves by at least 0.05 versus matched B, and cancer AUROC loss is no
worse than 0.02.

- If `P_keep` differs materially from current P while H is stable, attribute
  the prior placement result primarily to discard/washout.
- If `P_keep` agrees with current P, retention is not the explanation; choose
  one of branches 3 or 4.

### 3. Denominator fidelity and realized-dose experiment

Budget: one exploratory FM seed; reuse common plain/B paths; at most three P
and three H fair paths, serially.

First change only the NDL denominator to exclude
different-cancer/different-race pairs, retaining `lambda=0.1` and the
persisted-adapter timing. Only if representation movement remains inadequate,
run the fixed ladder `lambda in {0.1, 0.3, 1.0}`. Do not insert values after
looking.

A separate realized-dose comparison may choose stage-specific lambda values
only to match a preregistered fair/main gradient-norm ratio; report it as
realized-dose matched, not as the nominal same-lambda estimand.

Go only if:

- race-probe AUROC improves by at least 0.05 in both cancers;
- cancer AUROC loss is no worse than 0.02;
- improvement is monotone at two consecutive doses;
- neither cancer collapses; and
- the smallest passing dose is selected.

If no dose passes, stop FairCon rather than spend more seeds.

### 4. Patient-level granularity ablation

Run only if branch 1 shows weak tile-to-patient transfer.

Budget: one exploratory FM seed; two new fair paths plus reusable B. This
shares, rather than expands, the exploratory-seed ceiling with branch 3.

Use 16 unique patients per batch, four per cancer/race stratum, with eight
fixed tiles each. Preserve 128 tiles per step and 99,968 presentations.
Mean-pool the eight embeddings before the unchanged cancer/FairCon loss. P
and H use the same immutable patient/tile replay and no TP53 access.

Advance only with the same 0.05 race-leakage and -0.02 cancer-preservation
gates in both cancers and both fixed audit halves.

### 5. One alternate matched method

Run only if FairCon lacks a measurable representation mechanism. This replaces
another FairCon branch; it does not authorize method shopping.

Budget: one exploratory FM seed, one P/H pair, one GPU.

Test DANN with a cancer-conditioned race adversary at the identical 128-D h.
Discard adversary parameters in both timings. P updates E+A through gradient
reversal; H updates only A. Cancer SupCon, replay, adapter, exposure, and
downstream-label firewall remain fixed.

If DANN cannot reduce held-out race separability by 0.05 without more than
0.02 cancer-AUROC loss in either cancer, stop the method family. FINO and
PCGrad require a later separate justification.

### 6. Untouched-scenario confirmation

Only one package may advance: persisted FairCon, corrected/dose-selected
FairCon, patient-level FairCon, or DANN.

Budget: exactly three FM seeds, serially in one `main_1gpu` allocation.

Confirm on the anticipated UCEC+COAD extension, or another completely frozen
untouched scenario if UCEC+COAD lacks valid denominators. Freeze cohort,
downstream target, thresholds, analyzer, and verifier before diagnostic
values. The downstream label appears only after E+A are hashed and frozen.

A large/stable engineering result requires:

- absolute mean theta at least 0.02;
- all three seed effects in the same strict direction;
- every leave-one-out mean, both cancers, and both head halves in that
  direction; and
- every existing harm and utility gate.

Otherwise report small/unstable without adding seeds or making a significance
claim.

## Priority

Run 1 then 2 first. They directly test discard/washout with artifacts already
being produced. Then choose at most one of 3, 4, or 5. Branch 6 is the only
new confirmatory study.

This funnel yields a tangible conclusion even if the current screen is
negative: it distinguishes adapter discard/washout, loss construction,
insufficient realized dose, tile/patient mismatch, proxy non-transfer, and
endpoint floor without resorting to 48 seeds or optional significance.
