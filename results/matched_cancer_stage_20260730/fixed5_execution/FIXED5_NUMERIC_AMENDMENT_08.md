# Matched-cancer fixed-five numeric amendment 08

Frozen at `2026-07-30T22:05:48Z`, after independent outcome-blind review of
numeric Amendment 07 and before any scientific value or result was opened.

Amendment 07 is preserved byte-for-byte:

- bytes: `2,952`;
- SHA-256:
  `91d3b56bbf87f97faf14bbcb31689d0cdb750befed184b92073d31c2052f7f5b`.

The `1e-12` absolute, zero-relative tolerance in Amendment 07 applies only to
final inferential, directional, harm, and utility gate comparisons, including
the secondary two-sided paired-Student `p < 0.05` gate.

It does not alter endpoint or metric computation. These retain their original
IEEE-754 semantics:

- score `>=` fitted threshold classification;
- linear quantile interpolation;
- AUROC ordering and exact tie equality;
- AUPRC score grouping;
- ECE bin edges and membership; and
- exact `se == 0` and variance handling.

Synthetic analyzer and independent-verifier tests additionally cover the
secondary p-value at, within `5e-13` of, and at least `2e-12` to either side
of `0.05`.

The fixed-five source manifest, reports, verifier, and final completion bind
this amendment together with Amendments 01 through 07.
