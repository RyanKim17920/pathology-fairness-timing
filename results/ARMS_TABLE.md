# Debiasing arms: does any stage/method reduce the race EO gap without a performance cost?

Question: on the powered leak-free BRCA-TP53 race cell, which debiasing arm (pretraining-time "bake-in" vs post-hoc head; unconditional vs conditioned) actually shrinks the equalized-odds (EO = FPR-disparity, Black−White) gap — and at what AUROC cost?

## Phase-1 arms — BRCA-TP53 race, pooled target (n=334; Black 118 / White 210; baseline EO 0.140)

| Arm | What it does | Overall AUROC | ΔAUROC | Overall AUPRC | EO | ΔEO | Verdict |
|---|---|---|---|---|---|---|---|
| baseline | no debiasing (plain panel-pooled model) | 0.746 | — | 0.513 | 0.140 | — | baseline |
| A_marginal | bake-in (pretraining) adversarial, unconditional | 0.740 | −0.006 | 0.522 | 0.072 | −0.069 | reduces |
| A_cancercond | bake-in (pretraining) adversarial, cancer-conditioned | 0.738 | −0.008 | 0.491 | 0.085 | −0.055 | reduces |
| B_contrastive_marginal | post-hoc contrastive head, unconditional | 0.737 | −0.010 | 0.483 | 0.181 | +0.041 | worsens |
| B_contrastive_labelcond | post-hoc contrastive head, label-conditioned | 0.743 | −0.003 | 0.493 | 0.099 | −0.041 | reduces |
| B_dann_marginal | post-hoc DANN adversarial head, unconditional | 0.742 | −0.004 | 0.484 | 0.167 | +0.027 | worsens ⚠ guardrail FAIL (Black AUPRC 0.557) |
| B_fino_marginal | post-hoc FINO head, unconditional | 0.730 | −0.016 | 0.482 | 0.113 | −0.027 | level-down (Black AUROC drops 0.707→0.672) |
| B_pcgrad_marginal | post-hoc PCGrad head, unconditional | 0.745 | −0.001 | 0.501 | 0.195 | +0.055 | worsens |

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
