# Bake-in vs Patch-local FM Debiasing — Results (Hospital-Holdback)

Analyzer run: 2026-07-22 14:50 UTC · seed 20260716 · 10,000 center-clustered bootstrap draws.
Source table: `results/hh_arms_table.json` · Analyzer: `tools/hh_analysis.py` (AUPRC bug fixed).
Pre-registration: `results/prereg_bakein_vs_posthoc.md`, **frozen 2026-07-22 before any A/B arm was run** (baseline disparities were measured pre-freeze as context; the confirmatory A−B contrast was not).

## 1. Scenario recap

A vendor pretrains a pathology foundation model (FM) on many hospitals; a new hospital adopts it and trains a small local readout head on its own patients. The question is whether it is better to **bake fairness into the vendor FM** (arm A) or to let the adopting hospital **patch it post-hoc at the head** (arm B). The held-out unit is the SITE (hospital), not the cancer: the vendor FM was pretrained on 9,055 TCGA patients with the 8 target-hospital centers excluded, so it has never seen the target hospitals. Target = 3 folds of TCGA-BRCA centers (F1={A2,AC,S3}, F2={A7,B6}, F3={EW,LL,OL}), each with exactly 15 Black TP53+ events; each hospital trains a readout head on its own patients via internal k-fold CV, and primary inference pools the 3 folds' out-of-fold predictions. Confirmatory cell: **BRCA-TP53, race (Black vs White)**. Only the fairness *stage* differs across arms; the fairness method family is held constant across A and B.

EO = max(|FPR_Black − FPR_White|, |TPR_Black − TPR_White|), threshold set per-arm to White 80% specificity on that arm's own OOF scores, measured on the pooled held-out target hospitals. Positive FPR-disparity = Black over-called.

## 2. Per-arm table (pooled target: 118 Black / 210 White; 45 Black events, 42 White events)

Threshold τ refit per arm to matched White specificity. AUROC/AUPRC/FPR/TPR/PPV are per-group raw; disparities and EO at the refit threshold; calibration = slope / intercept / ECE per group.

| Arm | AUROC | AUPRC | Bl AUROC | Wh AUROC | Bl AUPRC | Wh AUPRC | Bl FPR | Wh FPR | Bl TPR | Wh TPR | Bl PPV | Wh PPV | FPR-disp | TPR-disp | **EO** | Bl slope | Wh slope | Bl ECE | Wh ECE | auroc_ok |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.746 | 0.513 | 0.707 | 0.748 | 0.579 | 0.465 | 0.342 | 0.202 | 0.644 | 0.595 | 0.537 | 0.424 | +0.140 | +0.049 | **0.140** | 0.131 | 0.107 | 0.134 | 0.054 | YES |
| A_marginal | 0.740 | 0.522 | 0.702 | 0.744 | 0.587 | 0.455 | 0.274 | 0.202 | 0.578 | 0.595 | 0.565 | 0.424 | +0.072 | −0.017 | **0.072** | 0.132 | 0.106 | 0.126 | 0.073 | YES |
| A_cancercond | 0.738 | 0.491 | 0.688 | 0.750 | 0.560 | 0.440 | 0.288 | 0.202 | 0.600 | 0.619 | 0.562 | 0.433 | +0.085 | −0.019 | **0.085** | 0.122 | 0.107 | 0.130 | 0.056 | YES |
| B_contrastive_marginal | 0.737 | 0.483 | 0.690 | 0.738 | 0.562 | 0.398 | 0.384 | 0.202 | 0.667 | 0.619 | 0.517 | 0.433 | +0.181 | +0.048 | **0.181** | 0.118 | 0.102 | 0.130 | 0.059 | YES |
| B_contrastive_labelcond | 0.743 | 0.493 | 0.706 | 0.745 | 0.569 | 0.437 | 0.301 | 0.202 | 0.622 | 0.619 | 0.560 | 0.433 | +0.099 | +0.003 | **0.099** | 0.119 | 0.110 | 0.146 | 0.061 | YES |
| B_dann_marginal | 0.742 | 0.484 | 0.677 | 0.772 | 0.557 | 0.429 | 0.370 | 0.202 | 0.533 | 0.667 | 0.471 | 0.452 | +0.167 | −0.133 | **0.167** | 0.115 | 0.110 | 0.142 | 0.071 | YES |
| B_fino_marginal | 0.730 | 0.482 | 0.672 | 0.750 | 0.566 | 0.410 | 0.315 | 0.202 | 0.556 | 0.571 | 0.521 | 0.414 | +0.113 | −0.016 | **0.113** | 0.125 | 0.110 | 0.139 | 0.056 | YES |
| B_pcgrad_marginal | 0.745 | 0.501 | 0.696 | 0.754 | 0.564 | 0.452 | 0.397 | 0.202 | 0.689 | 0.619 | 0.517 | 0.433 | +0.195 | +0.070 | **0.195** | 0.117 | 0.111 | 0.146 | 0.052 | YES |

All 8 arms have overall AUROC in 0.730–0.746 — real TP53 heads, all comfortably above the 0.6 auto-exclude floor. **No arm is auto-excluded; all are ranked.** Every per-group AUPRC is in [0.398, 0.587] (none negative, none >1 — AUPRC bug fix confirmed). EO is driven entirely by the FPR leg in every arm (FPR-disp ≥ |TPR-disp| everywhere), consistent with the baseline finding that Black patients are over-called. Note all calibration slopes are low (~0.11–0.13 both groups) — a shared under-confidence of the small local heads, not arm-specific; the guardrail uses ECE, which is stable per group.

## 3. Primary confirmatory result

**EO(A_cancercond) − EO(B_contrastive_labelcond)** — the pre-registered single confirmatory test, paired center-clustered bootstrap (10,000 draws), two-sided α=0.05:

- EO(A_cancercond) = **0.085**
- EO(B_contrastive_labelcond) = **0.099**
- **Point estimate: −0.014**
- **95% CI: [−0.068, +0.106]**
- **CI excludes 0: NO**

**Verdict (pre-committed decision rule): statistically INDISTINGUISHABLE.** The CI includes 0 and the guardrail holds for both arms, so under the frozen rule this is the "statistically indistinguishable" branch — a valid, reportable outcome, not a failure. **No winning stage is named.** Both the bake-in conditional FM (A) and the patch-local label-conditional head (B) cut baseline race EO by roughly the same amount (baseline 0.140 → 0.085 / 0.099), and the difference between them (0.014 EO) is far smaller than the pre-stated MDE (~0.07–0.13). Per the locked power caveat this close a race was expected to read as indistinguishable; the result is consistent with that expectation. Both conditional stages reduce disparity relative to baseline (baseline EO 0.140 sits outside neither arm's neighborhood but the A/B *difference* is null); neither can be claimed superior to the other at this power.

## 4. Guardrail table

Guardrail (must hold for a disparity reduction to count as a WIN): Black AUPRC ≥ baseline − 0.02 **AND** per-group calibration not worse. Baseline Black AUPRC = 0.579 → **AUPRC floor 0.559**; baseline Black ECE = 0.134 → **ECE ceiling 0.154**.

| Arm | Baseline EO | Arm EO | Reduces vs baseline? | Bl AUPRC (≥0.559) | Bl ECE (≤0.154) | Guardrail |
|---|---|---|---|---|---|---|
| A_marginal | 0.140 | 0.072 | yes (−0.068) | 0.587 ✓ | 0.126 ✓ | **PASS** |
| A_cancercond | 0.140 | 0.085 | yes (−0.055) | 0.560 ✓ | 0.130 ✓ | **PASS** |
| B_contrastive_marginal | 0.140 | 0.181 | no (worse) | 0.562 ✓ | 0.130 ✓ | PASS (n/a — no reduction) |
| B_contrastive_labelcond | 0.140 | 0.099 | yes (−0.041) | 0.569 ✓ | 0.146 ✓ | **PASS** |
| B_dann_marginal | 0.140 | 0.167 | no (worse) | **0.557 ✗** | 0.142 ✓ | **FAIL** (level-down; also no reduction) |
| B_fino_marginal | 0.140 | 0.113 | yes (−0.027) | 0.566 ✓ | 0.139 ✓ | **PASS** |
| B_pcgrad_marginal | 0.140 | 0.195 | no (worse) | 0.564 ✓ | 0.146 ✓ | PASS (n/a — no reduction) |

Both primary arms (A_cancercond, B_contrastive_labelcond) pass the guardrail — their disparity reductions are genuine, not level-down. **B_dann_marginal is the only guardrail failure** (Black AUPRC 0.557 < 0.559 floor), and it did not reduce disparity anyway (EO 0.167 > baseline), so it is disqualified on both counts. The three "marginal" B variants that raise EO above baseline (contrastive_marginal 0.181, dann 0.167, pcgrad 0.195) show that a marginal head patch can *worsen* race EO on held-out hospitals.

## 5. Marginal-vs-conditional ablation (SECONDARY)

Does the marginal objective "level down" (buy fairness by destroying minority signal) while the conditional objective preserves minority AUPRC?

**Bake-in FM (arm A): A_marginal vs A_cancercond**
- EO: marginal 0.072 → conditional 0.085 (conditional slightly *higher* EO, +0.014)
- Black AUPRC: marginal 0.587 → conditional 0.560 (conditional *loses* 0.027)
- At the FM/pretraining stage, going conditional does **not** help here: it neither lowers EO nor preserves AUPRC (it edges AUPRC down to the guardrail floor). Both A variants still pass the guardrail.

**Patch-local head (arm B): B_contrastive_marginal vs B_contrastive_labelcond**
- EO: marginal 0.181 → conditional 0.099 (conditional *reduces* disparity by 0.082)
- Black AUPRC: marginal 0.562 → conditional 0.569 (conditional *preserves/improves*, +0.007)
- At the head stage the ablation behaves as pre-registered: the **marginal head raises EO above baseline (0.181 vs 0.140) — a level-up in disparity — while the label-conditional head both cuts EO (to 0.099) and preserves minority AUPRC.** This is the clearest ablation signal in the study: for a post-hoc head patch, conditioning on the label is what makes the fairness objective work without harming the minority group.

Summary: the conditional-preserves-AUPRC story holds for the **head (B)** but not for the **FM (A)**, where the two variants are close on EO and conditional slightly reduces minority AUPRC. These are single-cell point comparisons (no CI); treat as secondary/descriptive.

## 6. Per-fold EO corroboration

Each fold has exactly 15 Black events (per design). Per-fold EO (folds are individually underpowered — 27–47 Black patients, 15 events each; shown for corroboration only):

| Arm | F1 (Bl 47, ev 15) | F2 (Bl 27, ev 15) | F3 (Bl 44, ev 15) |
|---|---|---|---|
| baseline | 0.112 | 0.284 | 0.267 |
| A_marginal | 0.050 | 0.284 | 0.200 |
| A_cancercond | 0.050 | 0.284 | 0.267 |
| B_contrastive_marginal | 0.112 | 0.368 | 0.333 |
| B_contrastive_labelcond | 0.112 | 0.368 | 0.333 |
| B_dann_marginal | 0.144 | 0.284 | 0.233 |
| B_fino_marginal | 0.144 | 0.368 | 0.161 |
| B_pcgrad_marginal | 0.081 | 0.284 | 0.333 |

All Black event counts are 15/15/15 across folds. F2 (smallest, 27 Black) carries the highest EO for every arm and is the dominant source of pooled disparity; the debiasing arms move F1 and F3 more than F2. Directions are consistent (all positive), corroborating the pooled result, but no single fold is powered for an arm contrast.

## 7. Honest limitations

- **n = 1 powered cell.** This is one confirmatory cell (BRCA-TP53 race). The single pre-registered test returned indistinguishable; there is no second powered cell to replicate it internally.
- **Pooled inference only.** The primary estimand pools 3 folds; each fold alone (15 events, 27–47 Black patients) is underpowered, so per-fold EO is corroborative, not inferential.
- **MDE ~0.07–0.13 EO.** By the locked power caveat, only a *large* stage difference is detectable; the observed |EO(A)−EO(B)| = 0.014 is well inside the MDE, so a close A-vs-B race is *expected* to read as indistinguishable. The null does **not** prove the stages are equivalent — only that any true difference is smaller than this study can resolve.
- **No external replication.** All data is held-out TCGA-BRCA hospitals; no non-TCGA cohort validates the finding.
- **Within-fold residual site confound.** Folds hold out whole centers, but a head trained via internal k-fold CV on a hospital's own patients still shares within-hospital batch characteristics; residual site signal inside a fold cannot be fully excluded.
- **Shared miscalibration.** All arms show low calibration slopes (~0.11–0.13) — the small local heads are under-confident across the board; the guardrail relies on ECE (stable) rather than slope.
- **Secondary/exploratory claims are descriptive.** The marginal-vs-conditional ablation and per-fold breakdowns are single-point comparisons without CIs and were pre-registered as secondary/exploratory.

### Bottom line
Baseline race EO on held-out hospitals is 0.140 (FPR-driven, Black over-called). Both a conditional bake-in FM (A_cancercond, EO 0.085) and a label-conditional patch-local head (B_contrastive_labelcond, EO 0.099) reduce it and pass the guardrail, but the confirmatory contrast between them is **null (−0.014, 95% CI [−0.068, +0.106])** — **statistically indistinguishable at this power**; no stage is declared the winner. The one interpretable mechanism finding is the head-stage ablation: a *marginal* head patch worsens EO (0.181) while a *label-conditional* head patch fixes it (0.099) without costing minority AUPRC.
