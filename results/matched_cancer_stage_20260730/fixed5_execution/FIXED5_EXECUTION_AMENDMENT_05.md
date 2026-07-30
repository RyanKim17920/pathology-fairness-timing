# Matched-cancer fixed-five execution amendment 05

Frozen at `2026-07-30T21:54:02Z`, after a second independent outcome-blind
execution review and before any scientific value or result was opened. Seed
32001 remained in calibration at freeze time.

This amendment resolves retry, orphan, lock-lifetime, and final-look recovery
details in:

`results/matched_cancer_stage_20260730/fixed5_execution/FIXED5_EXECUTION_AMENDMENT_04.md`.

Amendment 04 is preserved byte-for-byte:

- bytes: `5,458`;
- SHA-256:
  `604b11dd6e2bf8c7ee33e609b131fed31bd9e9b2c4fc0b22577b5b4c8502d676`.

Every term in Amendments 01 through 04 not explicitly changed below remains
in force. `FIXED5_SOURCE_MANIFEST_V1.json` binds all five amendments and every
implementation named here.

## Per-launch queue and allocation binding

Amendment 04's single prelaunch receipt is replaced by one immutable receipt
per submission attempt:

`control/prelaunch/FIXED5_PRELAUNCH_{launch_nonce}.json`.

The safe submitter creates a new 128-bit-or-stronger hexadecimal
`launch_nonce` while holding the shared submission lock. The receipt contains
that nonce and all Amendment 04 prelaunch fields. The nonce and exact receipt
path are exported to the Slurm allocation. A reused nonce or path fails.

At controller entry, after verifying the exported nonce, prelaunch receipt,
job name, legacy-compatible comment, one-task/one-GPU environment, and all
controls, the controller exclusively publishes:

`control/launch/FIXED5_LAUNCH_{launch_nonce}_JOB_{slurm_job_id}.json`.

Its schema is `matched-cancer-fixed5-launch/v1`; fields are status `running`,
the nonce, exact Slurm job ID, `job_name="main_1gpu"`,
`comment="matched_cancer_fixed48_20260730"`, one task, one allocated and
visible GPU, and `values_inspected=false`. It binds the prelaunch receipt,
source manifest, adoption authorization, controller, Slurm driver, and all
five amendments. Resubmission uses a new nonce and new launch receipt; prior
launch evidence is never overwritten.

## Attempt-scoped seed state

“Multiple starts fail” is scoped to one seed/attempt directory. Multiple
immutable interrupted or failed starts may exist across different attempt
numbers. Across one seed there may be at most one valid fixed-48 success and
at most one matching fixed-five completion.

Immediately before START publication, the controller:

1. verifies the source manifest, adoption, launch receipt, V2/V3/V2 controls,
   fixed-48 worker, and their exact identity snapshot;
2. scans excluded-seed state;
3. verifies canonical non-symlink ancestry and shared attempt numbering; and
4. verifies that no calibration or diagnostic artifact exists for the new
   attempt.

It publishes START exclusively, reverifies the complete identity snapshot and
excluded state, and only then invokes the worker. Drift between START and
worker invocation fails without invoking the worker.

Every calibration or diagnostic attempt for seed 32002 through 32005 must
have a pre-existing, same-number valid START. An attempt without START is an
orphan and invalidates the study. A START with no worker output, or START with
partial worker output but no success, is immutable interrupted/failed
evidence; retry uses a new attempt. START plus success but no COMPLETE may be
completed on resume only after full post-control verification. Bare success
is never adopted.

An execution attempt directory may contain exactly:

- a valid START;
- a valid START and valid COMPLETE; or
- an empty/temporary-only abandoned reservation created by a crashed
  exclusive publisher, provided no calibration or diagnostic path exists for
  that attempt.

An abandoned reservation is never reused or removed and consumes its attempt
number. Any other file, symlink, duplicate canonical receipt, partial
canonical receipt, or topology fails closed. Temporary receipt names cannot
serve as START or COMPLETE evidence.

## Execution-lock lifetime and excluded audit

The nonblocking canonical `control/serial_controller.lock` is acquired with
no symlink or ancestry redirection and remains held from controller entry
through seed execution, final collection, the single analyzer look,
independent verification, and final completion sealing. The controller calls
the finalization pipeline before releasing that lock; shell-level execution
after controller exit is not authorized.

The excluded-seed audit additionally binds:

- the seed-32001 adoption authorization and fixed-48 success;
- all four seed-32002-through-32005 START, COMPLETE, and fixed-48 success
  chains;
- the active launch receipt; and
- the source manifest and all five amendments.

Excluded-state absence is rescanned rather than trusted from the receipt at
every seed state scan and every finalization phase.

## Single-look, crash-resumable finalization

Each finalization attempt uses:

`finalization/attempt_kk/`

and may contain the exact canonical files:

- `FINALIZATION_START_RECEIPT.json`;
- `fixed5_predictions.jsonl` and its collection receipt;
- `analysis_report.json`;
- `independent_verification_report.json`; and
- `FINALIZATION_COMPLETE_RECEIPT.json`.

The START binds the complete five-seed/excluded-state chain, active launch,
source manifest, collector, analyzer, independent verifier, all five
amendments, and a pre-result independent continuation-options document. That
document proposes bounded follow-on experiments without inspecting current
scientific values; it is required before the analyzer can run.

Recovery rules are:

1. A verified collection and collection receipt are reused byte-for-byte.
2. A partial collection lacking its receipt abandons that finalization
   attempt; a new attempt may collect again because no analyzer look occurred.
3. Once any valid `analysis_report.json` exists, no new finalization attempt
   and no second analyzer invocation are allowed.
4. A valid analysis report without independent verification is resumed by
   verifying that same report; it is never regenerated.
5. A valid independent-verification report without final completion is
   reverified and sealed; neither analyzer nor verifier is rerun.
6. Any invalid, redirected, duplicate, or mismatched final artifact fails
   closed rather than being overwritten.

`FINALIZATION_COMPLETE_RECEIPT.json` binds the raw matrix and collection
receipt, single analysis report, independent-verification report, complete
seed and excluded-state ancestry, source manifest, active launch, continuation
options, and all five amendments.

Every future final verifier reopens and recomputes the raw matrix semantics
against the unchanged independent implementation, verifies the analysis
report and complete ancestry chain, rescans excluded state, and verifies the
source manifest. It does not rely only on a prior excluded-state receipt or
final completion flag.

## Legacy remainder limitation

No fixed-five code path calls the frozen fixed-48 remainder submitter, and
that remainder is deauthorized by Amendment 01. A deliberate later invocation
during a gap after a failed fixed-five allocation cannot be prevented without
mutating the frozen V2-bound script or maintaining an extra scheduler job,
both prohibited here. Such an invocation cannot be silently accepted:
orphan-attempt and excluded-seed scans invalidate the study before any further
seed, collection, analysis, or verification. Operational monitoring must
resubmit only through the fixed-five safe submitter.
