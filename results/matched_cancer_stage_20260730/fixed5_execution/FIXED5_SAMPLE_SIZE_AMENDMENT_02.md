# Matched-cancer fixed-five sample-size amendment 02

Frozen at `2026-07-30T21:15:15Z`, after an independent, outcome-blind audit
of Amendment 01 and before any scientific value or result was opened.

This amendment corrects and completes:

`results/matched_cancer_stage_20260730/fixed5_execution/FIXED5_SAMPLE_SIZE_AMENDMENT_01.md`

Amendment 01 is preserved byte-for-byte:

- bytes: `5,188`;
- SHA-256:
  `c2d6071a49b675f6c014e56d3cedb2eb534f2530ef5ce9c9c5a4c3e7aea279f2`.

Every Amendment 01 term not explicitly changed below remains in force.

## Immutable fixed-five controls

The additive fixed-five implementation must be frozen in:

`/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730/control/FIXED5_SOURCE_MANIFEST_V1.json`.

The fixed-five source manifest must bind by exact identity:

- Amendments 01 and 02;
- the unchanged fixed-48 V2 source manifest;
- the unchanged fixed-48 V3 diagnostic authorization;
- the unchanged fixed-48 V2 feasibility receipt;
- a fixed-five adoption authorization that binds the verified seed-32001
  success receipt;
- the fixed-five controller, Slurm driver, and safe-submit guard;
- the fixed-five collector, analyzer, and independent verifier;
- their complete local import closure; and
- all fixed-five tests and synthetic preflight receipts used to authorize
  production.

The manifest is verified before submission, before and after every seed,
before collection, before analysis, and after independent verification. It is
never edited or overwritten. Any correction requires a new versioned
manifest and a new immutable attempt.

The adoption authorization is:

`/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730/authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json`.

It may be created only after seed 32001 has a valid success receipt under the
unchanged V2/V3/V2 controls. It binds that receipt, both fixed-five
amendments, and the exact ancestor controls without recomputing or relabeling
the already-running canary.

The fixed-five scheduler comment is exactly:

`matched_cancer_fixed5_20260730`.

The safe-submit guard rejects submission if any pending or running job carries
either `matched_cancer_fixed48_20260730` or
`matched_cancer_fixed5_20260730`. The job name remains exactly
`main_1gpu`.

## Corrected large/stable rule

The “large/stable practical effect” rule in Amendment 01 additionally
requires:

- `median(theta_seed)` has the same strict sign as `mean(theta_seed)`; and
- `abs(median(theta_seed)) >= 0.02`.

Amendment 01's sign-consistency bullet is replaced with:

“At least four of the five seed effects are strictly nonzero and have the
sign of `mean(theta_seed)`; a zero effect counts as nonmatching.”

These requirements prevent a single large outlier from producing a
“large/stable” label when the typical tested-seed effect is negligible.

## Superseded aggregate wording

Every inherited aggregate described as an average “across 48 FM seeds” is,
under the fixed-five analysis, an equal-weighted average across exactly seeds
32001 through 32005. The original paired Student analysis uses four degrees
of freedom instead of 47. All numerical practical, harm, utility, cancer, and
head-half gates otherwise remain unchanged.

When all five paired differences are nonzero, the minimum attainable
two-sided exact sign-test p-value is `0.0625`. Zero differences make that
minimum larger. The exact sign test is descriptive and cannot be replaced by
treating heads, cancers, folds, targets, or patients as independent units.
