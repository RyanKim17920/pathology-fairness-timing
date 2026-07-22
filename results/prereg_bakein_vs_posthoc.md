---
# Pre-Registration — Bake-in vs Patch-local FM Debiasing (FROZEN before A/B arms)
Date frozen: 2026-07-22. Status: locked prior to running any bake-in (arm A) or post-hoc (arm B) debiasing. Baseline (no-fairness) disparities were measured before freezing and are recorded below as context; the A/B contrast has NOT been run.

## Scenario
A vendor pretrains a pathology foundation model (FM) on many hospitals. A new hospital adopts it and trains a small local readout head on its own patients. Question: is it better to bake fairness into the vendor FM (arm A) or to let the adopting hospital patch it locally post-hoc (arm B)?

## Deployment simulation
- Held-out unit = SITE (hospital), not cancer. Vendor FM pretrained on 9,055 TCGA patients EXCLUDING the 8 target-hospital centers (barcodes in exclude_target_centers.txt). The FM has seen breast from other centers; it has never seen the target hospitals.
- Target hospitals = 3 folds (Partition A) of TCGA-BRCA centers, each with exactly 15 Black TP53+ events:
  - F1 = {A2, AC, S3}; F2 = {A7, B6}; F3 = {EW, LL, OL}.
  - Pooled target: 118 Black (45 TP53+, 73 TP53-), 210 White.
- Each target hospital trains a readout head on its OWN patients via internal k-fold CV -> out-of-fold (OOF) predictions. Primary inference pools the 3 folds' OOF preds.

## Arms (only the fairness STAGE differs)
- Baseline: plain vendor FM + plain readout head.
- A (bake-in): fair vendor FM + plain readout head.
- B (patch-local): plain vendor FM + fair readout head.
Fairness method family held constant across A and B. Two conditioning variants are run for the contrastive objective:
  (i) marginal demographic contrast (the objective the shipped fair-*-nobrca checkpoints used), and
  (ii) conditional demographic invariance — cancer-conditional at pretraining (within-cancer: pull different-demographic together, keep cross-cancer separable); TP53-label-conditional post-hoc (within each TP53 class, since the hospital cohort is single-cancer breast).

## Primary estimand (CONFIRMATORY)
Cell: BRCA-TP53, RACE (Black vs White).
EO = max(|FPR_Black - FPR_White|, |TPR_Black - TPR_White|), threshold set per-arm to White 80% specificity on that arm's own OOF scores, measured on the pooled held-out target hospitals.
Primary contrast: EO(A) - EO(B), paired center-clustered bootstrap (>=5000 resamples), two-sided alpha=0.05.

## Guardrail (must hold for any disparity reduction to count as a WIN)
Minority (Black) AUPRC must not drop by more than delta = 0.02 absolute vs baseline, and per-group calibration (slope/intercept/ECE) must not worsen. A disparity reduction that fails the guardrail is reported as "level-down" (destroying the minority signal), not a fairness gain.

## Pre-committed decision rule
- A/B CI excludes 0 AND guardrail holds -> name the winning stage.
- A/B CI includes 0 AND guardrail holds -> "statistically indistinguishable" (a VALID, reportable outcome, not a failure).
- Guardrail fails for an arm -> that arm's reduction is "level-down", disqualified as a win.
- (Already established pre-freeze) baseline disparity persists on held-out hospitals -> effect is not pure site-batch.

## Power caveat (locked in)
Pooled target has 73 Black negatives (FPR quantization 1/73 ~ 0.014). Realistic 80%-power MDE for the A-vs-B contrast is ~0.07-0.13 EO once each arm's independent training noise is included (a proportional-shrinkage simulation gave an optimistic 0.014, judged unreliable). Therefore: a LARGE stage difference is detectable; a CLOSE race (|EO(A)-EO(B)| <~ 0.06) is expected to read as indistinguishable. This is stated in advance.

## Baseline disparities measured BEFORE freeze (context, no-fairness plain FM)
- BRCA-TP53 race: pooled-target FPR-disparity +0.318 (95% CI [+0.122, +0.434]); per fold +0.257 / +0.451 / +0.230 (1/3 folds individually significant, all positive). Whole-cohort +0.196. AUROC-gap ~0.056 (ns) — AUROC hides it.
- BRCA-TP53 age (young vs old): pooled-target +0.126 [+0.027, +0.213] (0/3 folds individually sig, all positive). SECONDARY.
- NSCLC sex (Female): -0.111 [-0.184, -0.048] under per-arm thresholding (opposite sign; note the OLD fixed-threshold pooled-sex +0.204 REVERSED under correct thresholding — a methods finding). EXPLORATORY.
- GBM sex/age, NSCLC age: not significant at power -> negative-control specificity checks (debiasing must not manufacture a gap).

## Confirmatory vs exploratory
- CONFIRMATORY: BRCA-TP53 race, EO(A)-EO(B), one test.
- SECONDARY: BRCA-TP53 age; the marginal-vs-conditional ablation (does marginal level-down while conditional preserves minority AUPRC?).
- EXPLORATORY: NSCLC sex; site-probe; Asian=Duke pure-site control; per-fold breakdowns.

## Analysis specification (locked)
Per-arm threshold refit to matched majority specificity; per-group raw AUROC/AUPRC/FPR/TPR/PPV + calibration reported for every arm; any arm with overall AUROC <= 0.6 auto-flagged and excluded from ranking; pooled-target primary + per-fold corroboration; patient- and center-clustered bootstrap; matched_pool post-hoc variants dropped (previously degenerate at AUROC ~0.5).
---