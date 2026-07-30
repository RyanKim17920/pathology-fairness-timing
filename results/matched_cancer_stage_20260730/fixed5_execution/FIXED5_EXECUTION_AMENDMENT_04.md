# Matched-cancer fixed-five execution amendment 04

Frozen at `2026-07-30T21:45:59Z`, after independent outcome-blind review of
Amendment 03 and before any scientific value or result was opened. Seed 32001
remained in calibration at freeze time.

This amendment completes:

`results/matched_cancer_stage_20260730/fixed5_execution/FIXED5_EXECUTION_AMENDMENT_03.md`.

Amendment 03 is preserved byte-for-byte:

- bytes: `3,040`;
- SHA-256:
  `f847e98d81b8f58e7bb08e5e2362803313a990a87f1045286afd24bf889a8347`.

Every term in Amendments 01 through 03 not explicitly changed below remains
in force. `FIXED5_SOURCE_MANIFEST_V1.json` must bind Amendments 01, 02, 03,
and 04.

## Prelaunch queue receipt

Immediately before the sole fixed-five `sbatch` call, while holding the shared
fixed48/fixed5 safe-submit lock, the fixed-five submitter scans all pending and
running jobs and requires zero exact matches for both the active legacy
comment and the superseded fixed-five comment. It exclusively publishes:

`control/FIXED5_PRELAUNCH_QUEUE_RECEIPT.json`

with schema `matched-cancer-fixed5-prelaunch-queue/v1`; status `pass`; both
exact comments; matching-job count zero; `values_inspected=false`; and exact
identities for the fixed-five source manifest, adoption authorization,
safe-submit source, Slurm driver, and Amendments 01 through 04. The controller
must verify this receipt at job start. A stale job with either comment,
missing receipt, redirected receipt, or receipt drift fails closed.

## Canonical start and completion receipts

For each seed `s` in 32002 through 32005 and fixed-48 attempt name
`attempt_kk`, the canonical fixed-five execution directory is:

`fixed5_execution/seed_{s}/attempt_kk/`.

The controller exclusively publishes, before invoking the frozen worker:

`FIXED5_START_RECEIPT.json`

with schema `matched-cancer-fixed5-seed-start/v1` and exact semantic fields:

- `status="started"`;
- `fm_seed=s`;
- `attempt_name="attempt_kk"`;
- `values_inspected=false`; and
- `excluded_seed_state_absent=true`.

It binds the fixed-five source manifest, adoption authorization, prelaunch
queue receipt, unchanged V2/V3/V2 controls, fixed-five controller, frozen
fixed-48 worker, and Amendments 01 through 04.

Only after the matching fixed-48 success and every post-control check pass,
the controller exclusively publishes in the same directory:

`FIXED5_COMPLETE_RECEIPT.json`

with schema `matched-cancer-fixed5-seed-complete/v1` and exact semantic fields:

- `status="complete"`;
- `fm_seed=s`;
- `attempt_name="attempt_kk"`;
- `values_inspected=false`; and
- `excluded_seed_state_absent=true`.

It binds the matching start receipt, matching fixed-48 success receipt, every
identity bound by the start receipt, the fixed-five controller, and Amendments
01 through 04.

A fixed-48 success is accepted only when its attempt name matches a valid
start receipt. A matching start plus valid success may be completed after a
controller crash, but a success without a pre-existing start is never
adopted or sealed after the fact. A start without success is immutable failed
or interrupted evidence; retry uses the lowest unused attempt number. A
completion without its matching start and success fails. Multiple starts,
completions, or successes for one seed fail.

Calibration, diagnostic, and fixed-five execution attempt numbers share one
namespace. All roots, seed directories, attempt directories, receipts, locks,
and control paths must be non-symlinks whose resolved ancestry remains inside
the exact canonical production root. State after an already successful
attempt fails closed.

## Excluded-seed audit and same-allocation finalization

Excluded-state validation rejects the existence of any path, including an
empty directory, file, or symlink, at:

- `calibration/seed_32006` through `calibration/seed_32048`;
- `diagnostic/seed_32006` through `diagnostic/seed_32048`; or
- `fixed5_execution/seed_32006` through
  `fixed5_execution/seed_32048`.

After all five seeds pass and before collection, the controller exclusively
publishes:

`control/FIXED5_EXCLUDED_SEED_AUDIT_RECEIPT.json`

with schema `matched-cancer-fixed5-excluded-seed-audit/v1`; status `pass`;
the exact excluded seed list; `excluded_state_present=false`;
`values_inspected=false`; and identities for the source manifest, adoption
authorization, controller, and Amendments 01 through 04.

The fixed-five Slurm script then runs, serially inside the same allocation and
while its legacy-compatible scheduler comment remains active:

1. final excluded-state validation;
2. the exact five-seed collector;
3. another excluded-state validation;
4. the fixed-five analyzer's single scientific look;
5. another excluded-state validation;
6. the independent verifier; and
7. a final excluded-state and source-manifest verification.

This supersedes, for fixed-five execution only, the fixed-48 controller rule
that prohibited final analysis in the production allocation. The collector,
analyzer, and verifier each independently rescan excluded state and verify the
clean audit receipt. They reject any later excluded-seed state on every future
reverification.

The Slurm allocation exits only after the independent verifier succeeds or
any step fails. Because it retains exact comment
`matched_cancer_fixed48_20260730` throughout finalization, the frozen
fixed-48 submitter continues to observe it and cannot start the remainder
during the collection-to-verification window.
