# Fixed-48 canary failure amendment 01

Status: operational amendment frozen before any diagnostic prediction,
diagnostic outcome, arm comparison, analyzer output, or inferential result was
opened.

This amendment supplements, without changing the estimand, endpoints, seeds,
arms, training schedule, diagnostic heads, decision rules, or utility gates
in the fixed-48 execution protocol.

## Failed production canary

Slurm job `368179` ran the seed-32001 canary as `main_1gpu` with comment
`matched_cancer_fixed48_20260730`, one task, and exactly one requested and
allocated GPU. Calibration `attempt_02` completed all five serial runs and
passed both its root verifier and an independent transitive audit.

Diagnostic `attempt_02` then failed before loader completion, prediction
export, structural audit, phase completion, or seed-success sealing. The only
diagnostic artifacts are the deployment contract, deployment gate, and an
empty partial run directory. They remain immutable failure evidence and are
excluded from inference.

The failure was:

`ValueError: tile-view record 00000311 hardlink differs`

No downstream diagnostic value was produced or inspected.

## Root cause

The tile-view receipt was sealed on `login-1`, where the shared `/data`
virtiofs mount reports device number `36`. The compute node `n-1` reports the
same shared mount and files with device number `37`. Device numbers are local
mount identifiers and are not durable cross-host file identities.

For all 688 tile-view records on `n-1`:

- the persisted device number differed;
- the persisted inode and byte count matched;
- the current source and view shared the same device and inode;
- both paths had link count at least two; and
- no regular-file, symlink, size, inode, or source/view-pair invariant drifted.

For failed record `00000311`, both source and view also matched the sealed
SHA-256 exactly. The legacy verifier reproduced the failure on `n-1`; the same
688-record receipt passed on `login-1`.

## Authorized implementation correction

The production verifier may stop comparing a current `st_dev` with the device
number persisted on another host. It must continue to require:

1. regular, non-symlink source and view paths;
2. recorded byte-count and inode agreement for both paths;
3. current source/view `st_dev` equality;
4. current source/view `st_ino` equality;
5. link count of at least two for both paths; and
6. the existing path, membership, and sealed source/view identity agreement.

Tests must demonstrate that a recorded-device-only difference passes, while a
replaced view, current source/view device mismatch, inode mismatch, size
drift, single-link topology, symlink, or inventory drift fails.

## Versioned controls and retry

The first source manifest and its authorization/feasibility controls are
superseded for future execution but retained as failure ancestry. The retry
must use new immutable artifacts:

- `control/FIXED48_SOURCE_MANIFEST_V2.json`;
- `authorization/AUTHORIZATION_MANIFEST_V3.json`; and
- `control/FEASIBILITY_GATE_RECEIPT_V2.json`.

The fixed authorization must bind a newly sealed legacy authorization that
uses the corrected loader identity. The feasibility gate must be recreated
against that fixed authorization. The source manifest must bind this
amendment and the complete corrected runtime closure.

Because the success verifier requires calibration and diagnostic outputs to
share one attempt number, calibration `attempt_02` cannot be relabeled or
paired with a diagnostic-only retry. The next production canary is a complete,
paired `attempt_03` under the corrected frozen implementation. Failed
`attempt_02` is neither moved nor overwritten.

Before submitting `attempt_03`, the corrected suite, real 688-record
compute-host verification, source/authorization/feasibility verification,
one-GPU submission preflight, and a fresh independent red-team must all pass.
