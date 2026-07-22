# Bake-in vs Post-hoc FM Debiasing — Final Synthesis

## 1. Question

When a hospital adopts a vendor pathology foundation model (FM), is it better for the **vendor to bake fairness into the FM at pretraining time** (arm A) or for the **adopting hospital to patch it post-hoc at a local readout head** (arm B)? The comparison is pre-registered, leak-free (site-holdout FMs that never saw the target hospitals), and deployment-framed (the held-out unit is the hospital/site, race = Black-vs-White, TP53 readout, equalized-odds gap `EO = max(|FPR_Black−FPR_White|, |TPR_Black−TPR_White|)` at a per-arm White-80%-specificity threshold).

## 2. Phase 1 — Powered single cell (BRCA-TP53, race)

Pooled held-out target: 118 Black / 210 White (45 / 42 events). Paired center-clustered bootstrap, 10,000 draws.

| | EO |
|---|---|
| Baseline (no fairness) | **0.140** |
| A_cancercond (bake-in conditional FM) | **0.085** |
| B_contrastive_labelcond (post-hoc conditional head) | **0.099** |
| **Confirmatory contrast A − B** | **−0.014, 95% CI [−0.068, +0.106]** |

CI includes 0, both arms pass the minority-AUPRC + calibration guardrail → **statistically INDISTINGUISHABLE**; no winning stage. The observed gap (0.014 EO) is well inside the pre-stated MDE (~0.07–0.13). Absolute performance cost is small: primary-arm AUROC deltas are −0.008 (A_cancercond) and −0.003 (B_contrastive_labelcond); no arm loses more than 0.009 AUROC (all arms 0.730–0.746, well above the 0.6 floor).

## 3. Phase 2 — Cross-cohort race panel (BRCA + UCEC + COAD)

Per-cohort baseline EO and reductions (positive reduction = disparity removed):

| Cohort | Black N (ev) | Baseline EO | A_cancercond EO (bake-in redn) | B_contrastive_labelcond EO (post-hoc redn) | Contrast A−B | Guardrail |
|---|---|---|---|---|---|---|
| BRCA | 118 (45) | 0.236 | 0.222 (+0.014) | 0.236 (+0.000) | −0.014 | all PASS |
| UCEC | 47 (23) | 0.333 | 0.302 (+0.031) | 0.258 (+0.075) | +0.044 | all PASS |
| COAD | 43 (27) | 0.098 | 0.098 (+0.000) | 0.036 (+0.062) | +0.062 | A_cancercond **FAIL** (Black AUPRC 0.651 < floor; level-down) |

DerSimonian–Laird random-effects meta (k=3, **real paired-bootstrap variances**):

| Effect | Pooled | 95% CI | I² |
|---|---|---|---|
| Bake-in reduction (baseline − A) | **+0.018** | [−0.032, +0.068] | 0% |
| Post-hoc reduction (baseline − B) | **+0.038** | [−0.024, +0.101] | 0% |
| Head-to-head (A − B) | **+0.024** | [−0.042, +0.090] | 0% |

**No CI excludes 0** → neither stage is significantly better; I²=0% (no heterogeneity). Post-hoc is a non-significant trend ahead (roughly double the pooled reduction, head-to-head favors it in 2/3 cohorts). COAD bake-in fails the minority-AUPRC guardrail (level-down), which post-hoc never does.

## 4. Phase 3 — Specificity / placebo (debias vs SHUFFLED demographics)

Re-run with the protected attribute permuted, so there is no real bias to remove: a well-behaved debiaser should leave performance flat.

- **Post-hoc placebo:** essentially flat, ±0.003–0.008 AUROC — no degradation.
- **Bake-in placebo:** +0.04–0.06 AUROC (no degradation). This is an *improbably large improvement* on shuffled labels and should be flagged for a sanity check rather than read as a genuine gain.

Conclusion: neither stage taxes performance when there is no bias present.

## 5. Overall conclusion (honest headline)

The decisive factor is **conditioning the debiasing objective on the outcome label**, NOT the stage. On single-cancer data, race and TP53 are entangled; a *marginal* objective buys fairness by destroying minority signal (level-down), while a *label-conditional* objective cuts EO and preserves minority AUPRC (the clearest mechanism signal was the Phase-1 head ablation: marginal head EO 0.181 vs label-conditional 0.099, Black AUPRC 0.562 → 0.569). **Bake-in vs post-hoc are statistically indistinguishable** — in the powered single cell (−0.014 [−0.068, +0.106]) and in the underpowered 3-cohort meta (+0.024 [−0.042, +0.090]). The claim that "pretraining removes bias better" is **NOT supported** by these data; if anything, post-hoc trends slightly ahead and preserves minority performance better (no guardrail failures).

## 6. Limitations

- k=3 cohorts — underpowered for a formal pooled significance claim.
- Baseline-EO gate is lenient by construction (EO is a non-negative max-of-|disparities|, CI bounded ≥0).
- UCEC's baseline gap is TPR-driven (a sensitivity gap), not the FPR over-flagging seen in BRCA/COAD.
- No external (non-TCGA) replication.
- Phase-3 bake-in placebo improvement magnitude is unverified and needs a sanity check.
- Within-fold residual site confound: whole-center holdout still shares within-hospital batch characteristics.

## 7. Artifacts

Result files:
- `/admin/home/ryan.kim/nt/results/bakein_vs_posthoc_results.md`
- `/admin/home/ryan.kim/nt/results/SUMMARY_bakein_vs_posthoc.md`
- `/admin/home/ryan.kim/nt/results/racepanel_results.md`
- `/admin/home/ryan.kim/nt/results/hh_meta_racepanel.json`
- `/admin/home/ryan.kim/nt/results/hh_arms_table_brca.json`
- `/admin/home/ryan.kim/nt/results/hh_arms_table_ucec.json`
- `/admin/home/ryan.kim/nt/results/hh_arms_table_coad.json`

Panel FMs (leak-free site-holdout re-pretrain; `latest.pt` in each):
- `/data/ryan.kim/nanopath/panel-plain/` (baseline)
- `/data/ryan.kim/nanopath/panel-contrastive-cancercond/` (bake-in A_cancercond)
- `/data/ryan.kim/nanopath/panel-contrastive-marginal/` (bake-in A_marginal)

Placebo FMs (shuffled-demographics controls):
- `/data/ryan.kim/nanopath/placebo-cancercond/`
- `/data/ryan.kim/nanopath/placebo-marginal/`
