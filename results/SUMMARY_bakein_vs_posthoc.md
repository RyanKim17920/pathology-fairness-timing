# Bake-in vs. Post-hoc FM Debiasing -- Executive Summary

## 1. Question & Scenario
When a hospital adopts a vendor pathology foundation model, is it better to **bake fairness into the vendor FM** at pretraining time, or to let the adopting hospital **patch it post-hoc** via a local readout head? A deployment-focused reframing with the hospital (site) as the held-out unit. Pre-registered before any arm was run.

## 2. Setup
- **Leak-free site-holdout FMs**: pretrained on 9,055 TCGA patients with 8 target centers excluded; FM has never seen the target hospitals.
- **Powered cell**: BRCA-TP53, race (Black vs White).
- **Pooled held-out target hospitals**: 118 Black / 210 White (45 Black events, 42 White events).
- **Per-arm re-thresholding**: White 80% specificity on each arm's own OOF scores.
- **EO** = max(|FPR_Black - FPR_White|, |TPR_Black - TPR_White|).

## 3. Primary Result
Paired center-clustered bootstrap (10,000 draws), two-sided alpha=0.05:

| | EO |
|---|---|
| Baseline (no fairness) | **0.140** |
| A_cancercond (bake-in conditional FM) | **0.085** |
| B_contrastive_labelcond (post-hoc conditional head) | **0.099** |
| **A - B contrast** | **-0.014 [-0.068, +0.106]** |

**Verdict: INDISTINGUISHABLE** -- CI includes 0, guardrails pass for both arms. Valid pre-registered outcome. The observed gap (0.014 EO) is well inside the MDE (~0.07--0.13).

## 4. Key Mechanism Finding
**Conditioning (not stage) is decisive.** At the head level, a *marginal* debiasing objective worsens disparity and costs minority AUPRC; a *label-conditional* objective fixes EO and preserves it.

| | EO | Black AUPRC |
|---|---|---|
| B_contrastive_marginal | 0.181 (+0.041 vs baseline) | 0.562 |
| B_contrastive_labelcond | 0.099 (-0.041 vs baseline) | 0.569 (+0.007) |
| A_marginal | 0.072 | 0.587 |
| A_cancercond | 0.085 | 0.560 (-0.027) |

The marginal head patch raises EO *above* baseline (0.181 vs 0.140). The label-conditional head cuts EO and improves minority AUPRC (+0.007). At the FM stage (A), both marginal and conditional are close; conditional slightly costs minority AUPRC.

## 5. Absolute Performance Cost

| Arm | AUROC | AUPRC | AUROC delta | AUPRC delta |
|---|---|---|---|---|
| baseline | 0.746 | 0.513 | -- | -- |
| A_marginal | 0.740 | 0.522 | -0.006 | +0.009 |
| A_cancercond | 0.738 | 0.491 | -0.008 | -0.022 |
| B_contrastive_marginal | 0.737 | 0.483 | -0.009 | -0.030 |
| B_contrastive_labelcond | 0.743 | 0.493 | -0.003 | -0.020 |

Debiasing is cheap. Primary arms lose -0.008 and -0.003 AUROC. Conditional (B) is cheapest at -0.003. All arms above 0.730 AUROC (well above the 0.6 floor).

## 6. Limitations
- **n = 1 powered cell** -- one confirmatory test; no internal replication.
- **Pooled inference only** -- individual folds (15 Black events each) underpowered.
- **MDE ~0.07--0.13** -- close A-vs-B race was expected to read as indistinguishable; null does not prove equivalence.
- **No external replication** -- all data from held-out TCGA-BRCA hospitals.

## 7. In Progress
- Multi-cohort race panel (BRCA+UCEC+COAD, leak-free re-pretrain) for meta-analytic confirmatory test.
- Specificity/placebo test of performance cost under no-bias.
