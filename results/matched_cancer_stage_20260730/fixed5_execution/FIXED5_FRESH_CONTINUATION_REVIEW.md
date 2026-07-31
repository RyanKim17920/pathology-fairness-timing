# Fresh fixed-five continuation review

Reviewed outcome-blind on `2026-07-31`, after the five-seed execution had
started but before any prediction, metric, analyzer output, or scientific
result was opened.

This is a bounded contingency review, not authorization to run every branch.
Any continuation remains capped at five **new unique FM seeds**, runs serially
on one GPU, starts with zero- or one-seed mechanism checks, and never adds
seeds to chase significance.

## Core correction

The clean adapter-persistence test is a carry-versus-fresh factorial, not a
`P_keep` comparison alone.  On one new FM seed, reuse common Slot-1 paths and
compare:

- fresh Slot-2 adapters: `B_fresh`, `P_fresh`, `H_fresh`;
- Slot-1-initialized Slot-2 adapters: `B_carry`, `P_carry`, `H_carry`.

The mechanism contrast is

`[P_carry - H_carry] - [P_fresh - H_fresh]`.

`B_carry - B_fresh` controls for warm-starting itself.  Evidence for P
discard/washout requires P's race-leakage reduction to appear or persist under
carry, H to remain broadly stable, and B carry/fresh utility to remain within
0.02.  This one-seed diagnostic cannot establish placement superiority.

## Bounded sequence

1. **Zero-seed checkpoint audit.** Separate encoder persistence, temporary
   adapter localization, fresh-adapter washout, and direct post-hoc retention
   on a population fixed before outcomes are opened.  If it is not genuinely
   patient-disjoint and multi-TSS, label the result descriptive rather than
   held out.
2. **At most one carry-versus-fresh seed.** Run the factorial above only if
   the checkpoint audit leaves persistence/localization unresolved.
3. **At most one mechanism-targeted seed.** Choose exactly one of denominator
   fidelity, realized-dose matching, patient-level granularity, PCGrad, or
   conditional DANN from value-blind mechanism evidence.  Do not bundle
   mechanisms.
4. **Exactly three untouched-scenario seeds.** Advance only one frozen package
   to a predeclared scenario.  Predeclare an ordered fallback list and
   label-blind feasibility rule; call this an external replication screen,
   not population confirmation.

This consumes at most five new unique FM seeds: one carry diagnostic, one
targeted intervention, and three untouched-scenario seeds.  Zero-training
audits do not consume an FM seed.

## Audit definitions and gates

Orient race leakage as

`L = max(AUROC, 1 - AUROC) - 0.5`,

so label inversion cannot masquerade as improvement.  Use cancer-conditioned,
patient-level, leave-one-TSS-out cross-fitting and compare only matched layers
of equal dimension.  The primary mechanism gate requires:

- leakage reduction versus matched B of at least 0.05 in both cancers and
  both fixed probe halves;
- cancer-information or downstream-utility loss no worse than 0.02;
- patient-pooled and tile-level directions to agree; and
- across the existing five checkpoints, the direction in at least four seeds
  with an agreeing median.

Cancer-probe preservation is a pooled BRCA-versus-LUAD measure.  Within-cancer
preservation must instead use downstream task utility.

## Choosing the one targeted branch

- **Denominator fidelity:** choose when the all-pairs denominator creates
  irrelevant or conflicting negatives.  Compare the current denominator with
  a mask restricted to same-cancer/different-race positives and
  different-cancer/same-race negatives, holding adapter topology fixed.
- **Dose:** choose only when gradient geometry is benign but the realized fair
  update is too weak.  Match realized parameter updates, accounting for
  learning rate, clipping, AdamW preconditioning, and different parameter
  sets; raw gradient norms are insufficient.
- **Patient granularity:** choose when tile leakage moves but patient-pooled
  leakage does not.  Compare tile loss and patient-mean-pooled loss on the
  identical patient-blocked replay.
- **PCGrad/constrained update:** choose only when persistent fair/cancer
  gradient conflict accompanies utility harm.
- **Conditional DANN:** choose only when FairCon receives adequate realized
  dose but race separability does not fall.  Apply a cancer-conditioned
  adversary to the same 128-D representation with matched initialization and
  schedule, discarding it in both timing arms.

Stop a branch immediately when its representation or utility gate fails.

## Outcome-contingent decisions

- **Stable meaningful P over H:** if encoder leakage reduction survives the
  fresh P adapter, advance directly to untouched replication.  Use the carry
  diagnostic only if the signal is confined to the temporary adapter or
  disappears from the final representation.
- **Stable meaningful H over P:** run the carry factorial first.  If carry
  rescues P, replicate the carry system.  Otherwise choose denominator/dose
  only when no fair signal reaches E or temporary A; choose PCGrad only with
  demonstrated harmful gradient conflict.
- **Practically small/equivalent:** distinguish active equivalence (both P and
  H reduce leakage versus B) from joint inactivity.  Replicate the simpler
  active package in the first case; use one mechanism branch in the second.
  If B has negligible fairness headroom, stop TP53 and move only to the
  predeclared untouched scenario.
- **Unstable/inconclusive:** never add seeds to stabilize the estimate.  If
  representation mechanisms vary, choose fidelity/dose or granularity from
  the audit.  If representations are stable but TP53 varies by cancer/head,
  treat it as proxy non-transfer and do not optimize this endpoint.
- **Utility/harm failure:** do not replicate the favored current package.  If
  B lacks absolute utility, the scenario is unsuitable.  If leakage improves
  while iEO or utility worsens, treat this as proxy mismatch, not a dose issue.

## Flaws this review corrects

- `P_keep` versus `H_keep` alone is not causally matched because P's adapter
  trained with a moving encoder while H's trained against a frozen encoder.
- Denominator fidelity and adapter persistence must not change in the same
  branch.
- The current all-pairs SupCon denominator treats same-cancer/same-race pairs
  as negatives while cancer loss treats them as positives, creating a
  mechanistically plausible conflict worth isolating.
- A replay-unused patient subset is not automatically a common held-out set.
- Existing initial encoder-gradient diagnostics cannot recover accumulated
  encoder dose.
- One failed DANN seed can stop only that frozen DANN package, not the entire
  adversarial-method family.
- Terms such as “material,” “inadequate movement,” and “stable” must receive
  numeric definitions before a branch is opened.
