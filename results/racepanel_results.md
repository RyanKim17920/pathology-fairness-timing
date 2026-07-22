# Race-Panel Head-to-Head: Bake-in vs Post-hoc Debiasing (TP53, Black-vs-White EO)

TCGA hospital/race-holdout target set, TP53 readout. Equalized-odds gap
`EO = max(|FPR_Black − FPR_White|, |TPR_Black − TPR_White|)`, per-arm threshold
refit to White 80% specificity. Fixed (non-negative) AUPRC. Baseline EO 95% CI =
TSS-cluster bootstrap within target (10,000 draws). Arms: `A_cancercond` =
pretraining-time (bake-in), `B_contrastive_labelcond` = post-hoc head.

## Per-cohort table

| Cohort | Black N (events) | White N | Baseline EO | Baseline EO 95% CI | Gate | A_cancercond EO | Bake-in reduction | B_contrastive_labelcond EO | Post-hoc reduction | Contrast A−B | Guardrails |
|--------|------------------|---------|-------------|--------------------|------|-----------------|-------------------|----------------------------|--------------------|--------------|------------|
| BRCA | 118 (45) | 210 | 0.236 | [0.102, 0.362] | PASS | 0.222 | +0.014 | 0.236 | +0.000 | −0.014 | all PASS |
| UCEC | 47 (23) | 134 | 0.333 | [0.049, 0.684] | PASS | 0.302 | +0.031 | 0.258 | +0.075 | +0.044 | all PASS |
| COAD | 43 (27) | 133 | 0.098 | [0.036, 0.441] | PASS | 0.098 | +0.000 | 0.036 | +0.062 | +0.062 | A_cancercond FAIL* |

Reduction = baseline EO − arm EO (positive = disparity reduced). Contrast = EO(A_cancercond) − EO(B_contrastive_labelcond); positive = bake-in leaves a *larger* gap (worse).

\*COAD `A_cancercond` fails the minority-AUPRC guardrail (Black AUPRC 0.651 < baseline floor); all other arms in all cohorts pass AUPRC + calibration guardrails.

## Baseline gate

All three cohorts pass the baseline-EO gate (bootstrap 95% CI lower bound > 0), so each has a real disparity to remove. Caveat: EO is a non-negative max-of-|disparities|, so its CI is bounded below by 0 and the gate is lenient by construction; COAD's disparity is modest (EO 0.098, CI lower bound 0.036) and UCEC's is TPR-driven (FPRdisp only +0.127) with a wide CI at Black N=47.

## Meta-analysis (DerSimonian–Laird random-effects, k=3)

| Effect | Pooled | 95% CI | I² | Sign consistency |
|--------|--------|--------|-----|------------------|
| Bake-in reduction (baseline − A_cancercond) | +0.0150 | [−1.12, 1.15] | 0.0% | 2 positive, 1 zero, 0 negative |
| Post-hoc reduction (baseline − B_contrastive_labelcond) | +0.0457 | [−1.09, 1.18] | 0.0% | 2 positive, 1 zero, 0 negative |
| Head-to-head (A_cancercond − B_contrastive_labelcond) | +0.0308 | [−1.10, 1.16] | 2 positive, 1 negative |

Note: the pooling uses unit (placeholder) per-study variances because `hh_arms_table.json` does not carry per-effect sampling variances, so the reported CIs (±~1.13) and I²=0% are uninformative — read the pooled point estimates and sign-consistency, not the CIs.

## Verdict

**Post-hoc debiasing systematically matches or beats bake-in; bake-in does NOT systematically match or beat post-hoc.**

- Pooled post-hoc EO reduction (+0.046) is roughly 3× the pooled bake-in reduction (+0.015).
- Head-to-head is positive in 2 of 3 cohorts (UCEC +0.044, COAD +0.062): bake-in leaves the larger residual gap. The only cohort where bake-in edges post-hoc is BRCA, and by a trivial +0.014 (there post-hoc did essentially nothing, EO unchanged at 0.236).
- Bake-in never reduces disparity by more than +0.031 and does nothing in COAD (0.000); post-hoc reduces most where the gap is real (UCEC 0.333→0.258, COAD 0.098→0.036).
- Guardrails favor post-hoc: `A_cancercond` sacrifices minority AUPRC in COAD (guardrail FAIL), whereas the post-hoc contrastive head preserves or improves Black AUPRC everywhere.

Honesty caveats: (1) all disparities pass the baseline gate but the gate is lenient for a non-negative EO metric, and COAD's baseline gap is small; (2) with only k=3 cohorts and no per-effect variances, the random-effects CIs are uninformative and the pooled estimates should be read as directional, not significance-tested; (3) UCEC's baseline EO is TPR-driven, not FPR-driven, so the "disparity" there is a sensitivity gap rather than the FPR over-flagging seen in BRCA/COAD. Direction is consistent (post-hoc ≥ bake-in) but the effect magnitudes are small and the panel is under-powered for a formal pooled significance claim.

## Meta-analysis — REAL bootstrap variances (variance bug fix, supersedes above)

Per-study variance now = SE², SE = (CI_hi − CI_lo)/(2×1.96) from each cohort's paired TSS-cluster bootstrap (10k draws; COAD 9999). Bake-in and post-hoc reduction CIs were re-run on the panel preds (absent from the tables); head-to-head CIs came from `confirmatory_contrast` (reproduced exactly). Placeholder unit variances retired.

Per-cohort SE used:

| Effect | BRCA | UCEC | COAD |
|--------|------|------|------|
| Bake-in reduction | 0.0332 | 0.0455 | 0.0863 |
| Post-hoc reduction | 0.0472 | 0.0552 | 0.0675 |
| Head-to-head (A−B) | 0.0531 | 0.0517 | 0.0807 |

Pooled DerSimonian–Laird (k=3), real 95% CI:

| Effect | Pooled | 95% CI | I² |
|--------|--------|--------|-----|
| Bake-in reduction (baseline − A_cancercond) | +0.0180 | [−0.0322, +0.0683] | 0.0% |
| Post-hoc reduction (baseline − B_contrastive_labelcond) | +0.0384 | [−0.0237, +0.1005] | 0.0% |
| Head-to-head (A_cancercond − B_contrastive_labelcond) | +0.0238 | [−0.0421, +0.0898] | 0.0% |

With real variances, no pooled CI excludes 0 (all three cross zero); I²=0% (no between-cohort heterogeneity). Head-to-head pooled is positive (+0.024, post-hoc leaves the smaller residual gap) but its CI includes 0 — directionally favors post-hoc, not significant. Verdict: on this k=3 race panel the pre-vs-post difference is NOT statistically distinguishable; the earlier directional read (post-hoc ≥ bake-in) survives only as a non-significant trend.
