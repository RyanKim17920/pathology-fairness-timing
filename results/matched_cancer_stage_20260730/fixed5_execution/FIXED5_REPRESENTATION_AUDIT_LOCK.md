# Fixed-five representation-audit lock

Frozen on `2026-08-01`, after the fixed-five placement result was finalized
but before any representation-audit metric was computed or opened.

This is a zero-new-training-seed mechanism audit.  It may perform deterministic
inference over frozen checkpoints, but it may not update model parameters or
use TP53/diagnosis/outcome data.  Because the audit was specified after the
placement result and its patients already powered the downstream TP53
diagnostics, it is exploratory/descriptive.  It is nevertheless strictly
held out from representation training.

## Question

The fixed-five result showed a practically small P-versus-H placement effect,
but P and H were both nearly unchanged from B.  The audit distinguishes:

- encoder persistence: `E_fair` versus `E_plain`;
- temporary-adapter localization: `A_temp_fair(E_fair)` versus
  `A_temp_plain(E_plain)`;
- fresh-adapter washout: final P versus temporary fair A, descriptively;
- final P activity: final P versus final B; and
- direct H activity: final H versus final B.

Only equal-dimensional layers may be contrasted.

## Frozen checkpoints

Use only FM seeds `32001..32005` and these accepted calibration attempts:

| FM seed | attempt |
| --- | --- |
| 32001 | `attempt_03` |
| 32002 | `attempt_01` |
| 32003 | `attempt_01` |
| 32004 | `attempt_01` |
| 32005 | `attempt_01` |

For each attempt, use exactly:

- `slot1_plain/latest.pt` for `E_plain` and `A_temp_plain(E_plain)`;
- `slot1_fair/latest.pt` for `E_fair` and `A_temp_fair(E_fair)`;
- `B/latest.pt`, `P/latest.pt`, and `H/latest.pt` for the three final
  128-dimensional representations.

The 384-dimensional family contains only the matched plain/fair encoders.
The 128-dimensional family contains the two temporary adapters and final
B/P/H adapters.  Final B and H must share the exact plain encoder state;
final P must share the exact fair encoder state.  Completion receipts,
checkpoint hashes, and accepted-attempt ancestry must verify before inference.

## Diagnosis-free population

Use the fixed target-hospital Black/White cohort, joined only from:

- `data/metadata/brca_racepanel_folds.csv` with `fold=target`;
- `data/metadata/luad_hospital_folds.csv` with `fold=target`; and
- the file-implied cancer label.

The sanitizer may emit only `{patient_id, cancer, race, tss}`.  It must reject
TP53, diagnosis, outcome, target, or `y_true` fields at every downstream API.

Exact expected population:

- BRCA: 328 patients, 8 TSS, 118 Black and 210 White;
- LUAD: 281 patients, 11 TSS, 40 Black and 241 White;
- union: 609 patients and 19 non-overlapping TSS.

Every patient must occur in the frozen 979-patient representation exclusion
list and in none of the five training replays.  Any mismatch fails closed.

## Frozen tile views

Require at least 32 valid tiles for every patient.  Rank each valid occurrence
by the SHA-256 digest of the UTF-8 string

`rep-audit/v1|288850999|patient_id|payload_sha256|occurrence_index`.

Resolve ties by `(digest, payload_sha256, occurrence_index)`.  Keep the first
32.  Even ranks form view `A` and odd ranks form view `B`, giving two disjoint
16-tile views.  Tile identities and view assignment must be identical across
every compared layer and seed.  Merely changing probe initialization does not
count as a view replicate.

Existing validated final B/P/H caches may be subset and reused.  Missing
encoder and temporary-adapter embeddings require one serial frozen-inference
pass.  Raw E and its corresponding temporary A(E) should be emitted together
so each plain/fair encoder is evaluated once per seed.

## Probes and secondary geometry

For each FM seed, cancer, layer, and tile view:

1. Patient probe: mean-pool the 16 tiles, then fit a cancer-conditioned race
   probe with leave-one-TSS-out cross-fitting.
2. Tile probe: fit on 16 tiles per training patient with weight `1/16`, hold
   out the entire evaluation TSS, average tile probabilities to patients, and
   score patient AUROC.

Standardization and C selection occur only inside the outer training TSSs.
Use logistic regression with `C in {0.01, 0.1, 1, 10, 100}`, deterministic
solver seed `288850999`, and nested leave-one-training-TSS-out selection.
Select the smallest C on ties.  Every outer TSS is evaluated exactly once;
inner folds lacking both races are excluded and counted, and each outer fit
must retain at least two valid inner folds.  Score only pooled held-out patient
predictions.

Orient race leakage as

`L = max(AUROC, 1 - AUROC) - 0.5`.

Also report, without creating additional primary gates:

- pooled BRCA-versus-LUAD patient probe under patient/TSS blocking;
- cancer-conditioned patient energy distance between races;
- patient and tile cross-race k-nearest-neighbor mixing using cosine distance,
  excluding the same patient and equal-weighting races;
- aligned tile and patient-mean representation displacement;
- layerwise parameter displacement from the matched starting state;
- all 781 logged h-space main/fair gradient norms, ratios, cosines, conflict,
  losses, learning rates, and denominator-support counts.

Do not call h-space gradients cumulative encoder dose.  Only first-batch
encoder reachability, net parameter displacement, and combined optimizer state
are recoverable; accumulated fairness-specific encoder updates are not.

## Primary mechanism gate

A fair representation is active only if its oriented leakage decreases versus
the matched plain/B layer by at least `0.05`:

- in both BRCA and LUAD;
- in both disjoint tile views;
- for both patient and tile probes, with agreeing directions;
- in at least four of five FM seeds, with the median agreeing; and
- with pooled cancer-probe AUROC loss no worse than `0.02`.

No metric row, patient, tile, TSS, cancer, seed, or view may be dropped after
values are seen.  Missing required cells fail the gate.

## Frozen result-contingent branch

- If final P or H passes, spend no targeted mechanism seed.  Freeze the exact
  active package and consider only a genuinely unseen scenario.
- If fair E or temporary fair A passes but final P fails, use at most one new
  FM seed for the carry-versus-fresh B/P/H factorial.
- If tile leakage passes but patient leakage fails, use that one seed instead
  for patient-mean training on the identical replay.
- If FairCon is inactive even in temporary fair A and no granularity signal
  exists, use that one seed instead for denominator fidelity only at fixed
  `lambda=0.1`.
- If the selected one-seed mechanism experiment fails the same `0.05/0.02`
  gates, stop this FairCon package.  Do not try dose, PCGrad, and DANN in
  sequence.

Only a passing package may advance to exactly three serial FM seeds in a
genuinely unseen, predeclared scenario.  UCEC+COAD is not untouched in this
repository and may be labeled only an external replication screen.  Across
all continuation work, use at most four new FM seeds, one GPU at a time, with
every study job named `main_1gpu`.  Leave the fifth allowed seed unused.
