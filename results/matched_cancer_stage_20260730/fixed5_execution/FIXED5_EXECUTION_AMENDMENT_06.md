# Matched-cancer fixed-five execution amendment 06

Frozen at `2026-07-30T21:58:20Z`, after final outcome-blind recovery review
and before any scientific value or result was opened. Seed 32001 remained in
calibration at freeze time.

This amendment closes two crash windows in:

`results/matched_cancer_stage_20260730/fixed5_execution/FIXED5_EXECUTION_AMENDMENT_05.md`.

Amendment 05 is preserved byte-for-byte:

- bytes: `7,434`;
- SHA-256:
  `3af62f4ec72b9e5a245426af3f7604ebc665ddb0f82d42edd1ad8639ab3ac2c4`.

All Amendment 05 terms not changed below remain in force. The fixed-five
source manifest and every final receipt bind all six amendments.

## Analyzer invocation barrier

The exact finalization topology additionally permits:

`ANALYZER_START_RECEIPT.json`.

Immediately before invoking the analyzer, after the raw matrix and collection
receipt pass and all controls/excluded-state checks pass, the controller
exclusively publishes this receipt. Its schema is
`matched-cancer-fixed5-analyzer-start/v1`; fields are status `started`, the
finalization attempt name, current launch nonce and Slurm job ID,
`scientific_values_opened=false`, and `analyzer_invocations_before=0`. It
binds the raw matrix and collection receipt, source manifest, analyzer,
finalization START, current launch/excluded audit or RESUME receipt,
continuation-options document, and all six amendments.

After `ANALYZER_START_RECEIPT.json` exists, the analyzer is invoked at most
once. A valid analysis report is resumed into independent verification
without re-invocation. If the process crashes or the analyzer fails after the
barrier but before publishing a valid analysis report, the study fails closed:
no new finalization attempt and no second analyzer invocation are authorized.

The final completion and every future verifier require and bind the analyzer
barrier, verify there is exactly one analysis report, and reject any second
barrier or report anywhere in the production root.

## Per-launch excluded audits and recovery ancestry

Amendment 04's single excluded-audit path is replaced by:

`control/excluded/FIXED5_EXCLUDED_{launch_nonce}.json`.

Every launch independently rescans absence and exclusively publishes exactly
one audit bound to that launch receipt. The first finalization START binds the
originating launch and its audit.

When a later allocation resumes an existing finalization attempt, before
opening or advancing any artifact it exclusively publishes in that same
attempt:

`FINALIZATION_RESUME_{launch_nonce}.json`.

Its schema is `matched-cancer-fixed5-finalization-resume/v1`; fields are
status `authorized`, finalization attempt name, recovery launch nonce and
Slurm job ID, and `values_inspected=false`. It binds:

- the original finalization START;
- the originating launch and excluded audit;
- every earlier RESUME receipt in nonce order;
- the current recovery launch and its excluded audit;
- every already-existing valid finalization artifact;
- the source manifest and all six amendments.

There is at most one RESUME per launch nonce. A recovery launch never replaces
the originating audit or any prior receipt. The analyzer barrier binds the
latest applicable START/RESUME chain. The independent-verification report and
final completion bind the complete ordered originating-and-recovery launch,
excluded-audit, and RESUME ancestry.

Every future verification replays each launch and excluded audit, every RESUME
link, the analyzer barrier, the single analysis report, and the complete raw
matrix semantics. A missing launch, audit, resume link, or identity mismatch
fails closed.
