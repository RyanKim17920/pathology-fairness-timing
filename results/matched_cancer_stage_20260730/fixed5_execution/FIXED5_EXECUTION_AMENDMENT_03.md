# Matched-cancer fixed-five execution amendment 03

Frozen at `2026-07-30T21:40:40Z`, after an independent outcome-blind
controller red-team and before any scientific value or result was opened.
Seed 32001 remained in calibration at freeze time.

This amendment corrects the scheduler comment and resume evidence specified
by:

`results/matched_cancer_stage_20260730/fixed5_execution/FIXED5_SAMPLE_SIZE_AMENDMENT_02.md`.

Amendment 02 is preserved byte-for-byte:

- bytes: `3,539`;
- SHA-256:
  `62eed85dc089d38118b19b1dbe6a807009b61bc095d96dd1d6da451a6ae7807c`.

Every term in Amendments 01 and 02 not explicitly changed below remains in
force.

## Scheduler comment compatibility

The fixed-five allocation's exact Slurm comment is:

`matched_cancer_fixed48_20260730`.

This supersedes Amendment 02's fixed-five comment. The unchanged legacy
fixed-48 submitter checks exact equality with this old comment. Reusing it
therefore makes that frozen submitter observe the queued/running fixed-five
allocation and refuse another submission after the shared submission lock is
released.

The fixed-five safe-submit guard must continue rejecting any pending or
running job whose exact comment is either:

- `matched_cancer_fixed48_20260730`; or
- `matched_cancer_fixed5_20260730`.

The second comment is retained only as a fail-closed guard against an aborted
or pre-amendment fixed-five submission. No production job is submitted with
it under this amendment. The job name remains `main_1gpu`.

## Per-seed fixed-five execution receipts

For each controller seed 32002 through 32005, the fixed-five controller must
publish exactly one exclusive, immutable receipt only after:

1. the fixed-five source manifest and adoption authorization pass;
2. the unchanged fixed-48 V2 source, V3 authorization, and V2 feasibility
   controls pass;
3. the frozen fixed-48 worker finishes the seed;
4. the fixed-48 seed-success receipt passes its complete ancestry check; and
5. all fixed-five and fixed-48 controls pass again without identity drift.

The fixed-five execution receipt binds by exact identity:

- seed-specific fixed-48 success receipt;
- fixed-five V1 source manifest;
- fixed-five adoption authorization;
- unchanged fixed-48 V2/V3/V2 controls;
- fixed-five controller source; and
- Amendments 01, 02, and 03.

Resume accepts seed 32002 through 32005 as complete only when both its
fixed-48 success receipt and fixed-five execution receipt pass. A success
without a fixed-five receipt, a receipt without a success, multiple receipts,
identity drift, or a redirected/symlinked receipt fails closed.

## Excluded-seed state

Before submission, before and after each controller seed, and on resume, the
fixed-five controller rejects any calibration attempt, diagnostic attempt,
success receipt, or fixed-five execution receipt for seed 32006 through
32048 in the production root.

The controller never invokes or adopts those seeds. Their appearance is
contamination evidence and terminates fixed-five execution without collection
or analysis.
