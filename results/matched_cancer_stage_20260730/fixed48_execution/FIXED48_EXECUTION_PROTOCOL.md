# Matched-cancer fixed-48 execution protocol

Status: inferential and operational design frozen before opening any real
diagnostic prediction value, diagnostic outcome value, arm comparison, or
interim result. Implementation identities will be sealed in a separate source
manifest only after synthetic and static validation, and before the first
production execution under this protocol.

This protocol supplements, but does not alter, the estimand, endpoints,
decision precedence, utility gates, or independent-verification rules in:

- `results/matched_cancer_stage_20260730/DIAGNOSTIC_FIXED_FINAL_LOCK.md`; and
- `results/matched_cancer_stage_20260730/DIAGNOSTIC_FIXED_FINAL_AMENDMENT_01.md`.

## Uniform production population

The production inferential units remain exactly the 48 paired
foundation-model seeds `32001` through `32048`, in ascending order. Every seed
is executed through one final, versioned fixed-48 implementation. For
representation seed `s`, the deterministic namespaces are:

- balanced-replay seed: `s + 20000`;
- data-order seed: `s + 30000`; and
- stage-adapter initialization seed: `s + 40000`.

All other representation exposure, optimizer, architecture, arm, cohort,
fold, diagnostic-head, and analysis settings remain those already locked.
Seeds are never substituted or added.

Every seed must load the same immutable DINOv2 ViT-S/14 register-token
pretrained ancestor from the locally bound checkpoint with SHA-256
`f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb`.
Its independently computed encoder-state SHA-256 is
`ba9418ed2138e42250085b04e0502d621b072c4bb60240f2845a27fbf3184bd6`.
Both identities are checked during every calibration and again across all 48
seeds at final collection.

The earlier seed-32001 calibration and diagnostic attempt 02 were explicitly
run and audited as systems-only validation. No prediction value, label
summary, metric, arm comparison, analyzer, or final verifier was opened.
Nevertheless, those artifacts are excluded from inference. They are immutable
legacy evidence and may be used only to validate structure and provenance.
Seed 32001 will be rerun in a new fixed-48 output namespace using the same
final implementation as seeds 32002 through 32048.

## Scenario and source provenance

Each production seed uses the current scenario
`brca_luad_black_white_calibration_seed{s}`. The already sealed cohort source
bundles, tile ledgers, and tile view remain under their original legacy
seed-32001 provenance scenario and are never rewritten or relabeled.

The fixed-48 authorization and every per-seed deployment gate must:

1. bind the original authorization, source bundles, ledgers, tile view, and
   estimand amendment by exact file identity;
2. validate those data ancestors under their original provenance scenario;
3. bind the current-seed calibration root and B/P/H completion receipts under
   the current scenario; and
4. write current-seed cohort, diagnostic, export, collection, and success
   receipts that retain both ancestry chains.

The same patient IDs and outcome/race/fold metadata must be identical across
all arms, heads, and representation seeds.

## Attempt and resume semantics

Production outputs use immutable attempt directories:

`fixed48/{calibration,diagnostic}/seed_{s}/attempt_{k}`.

An existing file, symlink, attempt, or success receipt is never overwritten.
A failed or interrupted attempt remains as evidence. A retry uses the lowest
unused positive attempt number with the same seed namespaces. A seed advances
only after independent structural audits and a sealed
`SEED_SUCCESS_RECEIPT.json` bind its complete calibration and diagnostic
ancestry. Resume re-verifies every success receipt and never trusts a mutable
status flag or directory name.

## One-study-GPU execution

There may be at most one queued or running matched-cancer fixed-48 study GPU
job. Job arrays, dependency chains, and bulk seed submissions are prohibited.
The production controller:

1. acquires a nonblocking allocation-wide filesystem lock;
2. rejects array execution;
3. verifies exactly one allocated GPU and exactly one visible CUDA device;
4. verifies the frozen source manifest before starting and before every seed;
5. processes only the lowest incomplete seed, strictly ascending;
6. runs calibration, calibration audit, diagnostic, structural diagnostic
   audit, and success sealing serially;
7. stops at the first failed invariant without skipping a seed;
8. never invokes an analyzer, final verifier, or result summarizer; and
9. never invokes `sbatch` or launches another GPU allocation.

Execution uses two sequential queue entries: one seed-32001 production canary,
followed only after independent audit by one resumable serial allocation for
seeds 32002 through 32048. No second study allocation may be submitted while
either entry is pending or running.

## Outcome-blind production monitoring

Production monitoring may inspect only scheduler state, file/receipt
identities, schemas, coordinates, row counts, finite/type/range validity,
normalization invariants, training-audit topology, and completion state. It
must not aggregate outcomes, probabilities, fairness, utility, or arm
differences.

Before expensive production execution, a pass/fail-only feasibility gate must
confirm that every locked cancer/race/outcome/fold denominator required by the
fixed endpoint exists. The durable result exposes only PASS/FAIL and bound
source identities, not counts or labels.

## Preproduction requirements

Before the seed-32001 production canary:

1. freeze an exact source manifest for all new and reused runtime files;
2. pass all unit, tamper, source-redirection, legacy-path write-denial,
   downstream-label-firewall, resume, and one-GPU controller tests;
3. pass a full-cardinality synthetic
   `1,753,920`-row collection/analyzer/verifier exercise;
4. independently red-team the frozen implementation and source manifest; and
5. confirm there is no other queued or running fixed-48 study GPU job.

The canary is audited without inspecting scientific values. After it passes,
the source manifest is unchanged for the remaining 47 seeds.

## Final collection and the only inferential look

After all 48 success receipts exist, a cross-gate collector independently
verifies every seed-specific calibration, deployment, loader, export, and
collection ancestry; exact seeds `32001..32048`; all 1,152
seed/arm/cancer/head coordinates; cross-seed cohort metadata identity; and
exactly `1,753,920` prediction rows.

It emits one sealed fixed-final JSONL and a completion receipt binding:

- all 48 seed success and collection receipts;
- the final collector and frozen source manifest;
- the original lock, Amendment 01, and this execution protocol;
- the analyzer and independent verifier implementations; and
- the raw fixed-final prediction identity and row count.

Only then are the preregistered analyzer and independent verifier run once.
There are no interim looks, optional stopping, adaptive extension, or
“run-until-significance” decisions.
