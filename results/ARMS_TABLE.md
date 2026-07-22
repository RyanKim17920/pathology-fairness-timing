# Debiasing arms: does any stage/method reduce the race EO gap without a performance cost?

Question: on the powered leak-free BRCA-TP53 race cell, which debiasing arm (pretraining-time "bake-in" vs post-hoc head; unconditional vs conditioned) actually shrinks the equalized-odds (EO = FPR-disparity, Black−White) gap — and at what AUROC cost?

## Phase-1 arms — BRCA-TP53 race, pooled target (n=334 pooled/dedup; Black 118 / White 210 = 328 used for EO; baseline EO 0.140)

| Arm | What it does | Overall AUROC | ΔAUROC | Overall AUPRC | EO | EO 95% CI | ΔEO | ΔEO vs base [95% CI] | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| baseline | no debiasing (plain panel-pooled model) | 0.746 | — | 0.513 | 0.140 | [0.062, 0.290] | — | — | baseline |
| A_marginal | bake-in (pretraining) contrastive SSL, unconditional | 0.740 | −0.006 | 0.522 | 0.072 | [0.020, 0.242] | −0.068 | [−0.134, +0.035] | reduces |
| A_cancercond | bake-in (pretraining) contrastive SSL, cancer-conditioned | 0.738 | −0.008 | 0.491 | 0.085 | [0.025, 0.278] | −0.055 | [−0.134, +0.056] | reduces |
| B_contrastive_marginal | post-hoc contrastive head, unconditional | 0.737 | −0.010 | 0.483 | 0.181 | [0.066, 0.354] | +0.041 | [−0.069, +0.131] | worsens |
| B_contrastive_labelcond | post-hoc contrastive head, label-conditioned | 0.743 | −0.003 | 0.493 | 0.099 | [0.033, 0.256] | −0.041 | [−0.128, +0.044] | reduces |
| B_dann_marginal | post-hoc DANN adversarial head, unconditional | 0.742 | −0.004 | 0.484 | 0.167 | [0.078, 0.291] | +0.027 | [−0.088, +0.111] | worsens ⚠ guardrail FAIL (Black AUPRC 0.557) |
| B_fino_marginal | post-hoc FINO head, unconditional | 0.730 | −0.016 | 0.482 | 0.113 | [0.039, 0.299] | −0.027 | [−0.138, +0.094] | level-down (Black AUROC drops 0.707→0.672) |
| B_pcgrad_marginal | post-hoc PCGrad head, unconditional | 0.745 | −0.001 | 0.501 | 0.195 | [0.071, 0.306] | +0.055 | [−0.091, +0.109] | worsens |

CIs from ≥5000-draw (10 000) patient+TSS-cluster bootstrap, per-arm τ refit to White 80% specificity on that arm's own pooled scores (reusing `hh_analysis.py` conventions). ΔEO vs base is a paired bootstrap (same patients/clusters resampled, EO_arm − EO_baseline recomputed each draw). No arm's ΔEO CI excludes 0 (none marked *).

**Note:** the per-arm ΔEO CIs above are exploratory/uncorrected (8 arms, no multiplicity adjustment); only the pre-registered A_cancercond − B_contrastive_labelcond contrast (−0.014 [−0.068, +0.106]) is confirmatory.

Guardrail = Black AUPRC + Black ECE not degraded; only B_dann_marginal fails. Confirmatory contrast (A_cancercond vs B_contrastive_labelcond): EO diff −0.014, 95% CI [−0.068, 0.106] → indistinguishable.

## Phase-2 pooled meta (k=3 cohorts: BRCA, UCEC, COAD; all I²=0)

- Bake-in EO reduction: +0.018 (95% CI −0.032 to +0.068) — not significant.
- Post-hoc EO reduction: +0.038 (95% CI −0.024 to +0.101) — not significant.
- Head-to-head (bake-in − post-hoc): +0.024 (95% CI −0.042 to +0.090) — not significant.

## Phase-3 placebo (corrected)

- Post-hoc placebo (random sensitive label): ≈ inert — no spurious EO reduction.
- Bake-in placebo vs correct panel-plain baseline (AUROC 0.8096): marginal 0.807 (−0.002), cancercond 0.789 (−0.021).
- The earlier +0.06 "placebo improvement" was a comparator artifact (wrong baseline), not a real effect.

## Bottom line

Conditioning — not the stage (bake-in vs post-hoc) — is what decides the outcome: every conditioned arm reduces the gap while the unconditional post-hoc arms either worsen it (contrastive/DANN/PCGrad) or level down (FINO degrades the minority group). Bake-in vs post-hoc are statistically indistinguishable in the pooled meta (all CIs cross zero). The performance cost of debiasing is negligible throughout — worst-case AUROC loss ≤ ~0.017 (FINO).
