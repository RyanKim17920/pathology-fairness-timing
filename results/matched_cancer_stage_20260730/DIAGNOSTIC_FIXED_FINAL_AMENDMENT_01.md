# Matched-cancer diagnostic amendment 01: race-eligible cohort cardinality

Frozen at `2026-07-30T06:59:27Z`, before opening any real prediction or outcome
value.

Provenance erratum frozen at `2026-07-30T07:03:13Z`: the formal re-audit
selected the source `fold` column, in addition to `patient_barcode` and `race`,
to establish target membership. The earlier two-column wording below has been
corrected before binding. No inferential value or cohort count changed.

Status: label-blind re-audit **PASS**. This amendment changes only the BRCA
race-eligible cohort cardinality and the implied fixed prediction-matrix row
count. Every endpoint, seed, arm, head, fold, threshold, inferential rule,
precedence rule, direction convention, harm gate, and utility gate in the
original lock remains unchanged.

## Preservation of the original lock

The original
`results/matched_cancer_stage_20260730/DIAGNOSTIC_FIXED_FINAL_LOCK.md` is
preserved byte-for-byte. Its pre-amendment identity is:

- bytes: `5,593`;
- SHA-256:
  `7f9d02a0f94a0a7153e018536d6470cdf54ad23a3f89a138458dc93fb7caa327`.

Where the original lock states BRCA `N=334`, this amendment supersedes only
that number with the Black/White-eligible BRCA `N=328`.

## Label-blind provenance

The audit used `patient_barcode` for identity/uniqueness, the source `fold`
column for target membership, and `race` for Black/White eligibility/counting.
No `tp53_status`, prediction, probability, endpoint, or other outcome-bearing
value was selected or inspected. Whole-file SHA-256 was computed solely as
immutable byte provenance.

| Cancer | Immutable cohort source | Bytes | SHA-256 | Allowed audit columns |
|---|---|---:|---|---|
| BRCA | `data/metadata/brca_racepanel_folds.csv` | 40,282 | `3420720508a4a7fff9f795a9b92b9ce7436f78cf96975158cf807594d031d9ef` | `patient_barcode`, `race`, `fold` |
| LUAD | `data/metadata/luad_hospital_folds.csv` | 13,341 | `06a8e5cdbb9594d4358b9979822d1b2ce7e6561acce04a4f16100894c993698e` | `patient_barcode`, `race`, `fold` |

The observed source header topology for both files is
`patient_barcode,tss,race,tp53_status,fold`; only the three explicitly allowed
columns above participated in this re-audit.

Here, source `fold` is a pre-existing non-outcome target-membership field. It
is not the generated diagnostic `fold`, `outer_fold`, or `inner_fold` value in
the sealed prediction-row contract, and it is not one of the diagnostic
five-fold assignments `0..4`.

## Corrected fixed cohorts

- BRCA raw target: 334 patients.
- BRCA eligible Black/White target: **328** patients:
  118 Black and 210 White.
- BRCA exclusions by race only: five Asian and one American
  Indian/Alaska Native patient.
- LUAD eligible Black/White target: **281** patients:
  40 Black and 241 White.
- LUAD receives no cardinality change.

These exclusions occur before diagnostic prediction analysis and are
deterministic consequences of the already-frozen analyzer row contract, which
permits only `Black` and `White`.

## Corrected fixed matrix

The production analyzer and independent verifier must require:

- cohort sizes `{"BRCA": 328, "LUAD": 281}`;
- identical patient IDs and outcome/race/fold metadata across all 48 FM seeds,
  all B/P/H arms, and all four heads; and
- exactly
  `48 FM seeds × 3 arms × 4 heads × 5 nested rows × (328+281 patients)`
  = **1,753,920 prediction rows**.

The nested threshold audit cardinality remains
`48 × 3 × 2 cancers × 15 specificity targets × 5 outer folds = 21,600`
records.

No optional stopping, outcome-dependent amendment, replacement seed, or
post-result cohort choice is authorized by this correction.
