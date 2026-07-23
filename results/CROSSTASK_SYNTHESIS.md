# Cross-task fairness synthesis — does pretraining-time (bake-in) demographic debiasing generalize across tasks?

**Study (HARD_PLAN, 2026-07-23).** One shared plain foundation model (nanopath-JEPA) is pretrained with the target hospitals of **two** biomarker-readout tasks held out, then compared against (a) 11 **bake-in** FMs that add a demographic-debiasing objective *during pretraining* and (b) 8 **post-hoc** debiasing heads on the plain FM. Every evaluation is **leak-free** (held-out hospitals), sensitive axis = **race** (Black vs White), metric = fixed-threshold signed **FPR-disparity** (equalized-odds; White 80%-specificity threshold) with patient+TSS-cluster bootstrap **BCa** CIs (10,000 draws); AUROC-gap and minority AUPRC/calibration are secondary/guardrail.

- **Task 1 — BRCA-TP53** (held-out hospitals F1/F2/F3; Black 118, 45 events).
- **Task 2 — LUAD-TP53** (held-out hospitals = target fold; Black 40, 26 events).
- **Shared plain FM** = `vendor-plain-brca-luad-holdout` (BRCA+LUAD target hospitals excluded from pretraining: 615 barcodes / 246,060 tiles).

## Baseline (plain FM, leak-free)

| Task | signed FPR-disp [BCa] | AUROC-gap [BCa] | Black AUROC |
|---|---|---|---|
| BRCA-TP53 | **+0.126 [+0.005, +0.233]** (excludes 0) | +0.051 [−0.055, +0.179] | 0.701 |
| LUAD-TP53 | +0.229 [−0.204, +0.416] (wide, N=40) | **+0.248 [+0.004, +0.407]** (excludes 0) | 0.473 (≈random) |

Both tasks carry a **real** leak-free race disparity: BRCA significant on FPR-disparity; LUAD a large FPR effect (Black FPR 0.43 vs White 0.20) whose FPR-disparity CI is wide because of the small Black N, but whose **AUROC-gap is significant** and whose Black-group readout is near-random. (LUAD-sex was significant only under the leaky screen and is **null** leak-free — a leakage artifact — so race is the axis for both tasks.)

## Headline findings

**1. Bake-in fairness does NOT generalize across tasks.**
Of 11 pretraining-time debiasing FMs, **none** reduced the race FPR-disparity on both held-out tasks (point-estimate basis): 2 helped BRCA only, 9 helped neither. On **LUAD nothing helped**, and several bake-in objectives made it *worse* (contrastive-demographics, FINO-race, DANN-racesexage, PCGrad-race: +0.229 → +0.300). On **BRCA**, only contrastive-demographics nudged it down (+0.126 → +0.085), while FINO/DANN-racesexage/twocond-racesexage/PCGrad *increased* it to +0.15–0.20 — several with BCa CIs sitting entirely above the baseline point estimate.

**2. Bake-in debiasing degrades minority performance.**
**8 of 11** bake-in FMs **failed the minority guardrail on BRCA** (Black AUPRC dropped >0.02 and/or Black ECE worsened >0.02) — e.g. contrastive-cancer (−0.053 AUPRC), PCGrad-race (−0.058), twocond-racesexage (−0.045). Suppressing demographic signal in the encoder cost the minority group readout quality.

**3. Post-hoc debiasing is gentler but not a decisive winner.**
Post-hoc arms **almost all passed** the minority guardrails (only FINO failed on LUAD-ECE), i.e. they did not degrade Black-group performance the way bake-in did. But their disparity reductions are **small (≤0.07) and within noise** at these minority sample sizes. The single best LUAD reducer is `pcgrad_labelcond` (+0.229 → **+0.157**, CI still wide [−0.202, +0.369]); the best BRCA post-hoc arms (`pcgrad_*` +0.099) roughly match the best bake-in.

**4. Label-conditioning is mixed here — not the decisive lever it was single-task.**
Label-conditional post-hoc beat marginal in only **2 of 8** family×task cells (FINO and PCGrad, on LUAD). The prior single-task result ("conditioning the debiasing objective on the outcome label is what matters") does **not** robustly replicate across these two tasks at this power.

## Verdict

At leak-free, held-out-hospital evaluation on two biomarker-readout tasks:

> **Fairness "baked in" during pretraining does not transfer across tasks, and it tends to harm minority-group performance.** Post-hoc debiasing on the plain FM is the safer choice — it preserves minority AUPRC/calibration — but **neither stage produces a large, statistically robust cross-task reduction** in the race FPR-disparity at these minority sample sizes. The decisive factor in the earlier single-task study (outcome-label conditioning) does not generalize cleanly to a second task.

## Limitations (honest)

- **Power.** LUAD minority N=40 (26 events) makes its FPR-disparity CI wide; small debiasing deltas (±0.07) are not distinguishable from zero. BRCA is better powered (45 events) but its debiasing deltas are also small. Effects are pooled-target; per-fold cells are underpowered.
- **k=2 tasks.** "Cross-task generalization" rests on two cancer types; the negative result is consistent and directionally strong (bake-in never wins on both, often loses), but two tasks cannot establish a general law.
- **Single plain FM / single seed** for the shared baseline (per the plan's "cheap but valid" decision; the M0 metric fix resolved the CI-behavior concern that had motivated multi-seed retraining). Bake-in FMs are single-seed each.
- **One axis (race), one outcome family (TP53).** LUAD-sex was null leak-free; other axes/labels (UCEC/COAD × race/sex/age/stage) were fully null in task selection and excluded.
- **A cache-key bug** (embedding cache keyed on `latest.pt` basename, colliding across FMs) was found and fixed mid-study; all reported numbers are from the corrected, checkpoint-identity-keyed cache and were validated (new-FM preds differ from prior-FM preds).

## Artifacts

- Full numbers: `results/hh_crosstask_table.md`, `results/hh_crosstask_results.json`.
- Metric engine: `tools/hh_metrics.py` (signed FPR/TPR-disparity, EO, AUROC-gap, per-group AUROC/AUPRC/PPV/calibration; patient+TSS bootstrap, percentile + BCa; `--sensitive-axis race|sex|age`).
- Aggregator: `tools/hh_crosstask_analysis.py`. Drivers: `tools/hh_drivers/{m2_gate_eval,m3_full,m4_posthoc,m4_posthoc_onefold,m5_crosstask,m3_smoke}.sbatch`.
- FMs: `/data/ryan.kim/nanopath/{vendor-plain-brca-luad-holdout, m3_*}` (all exclude `data/metadata/exclude_brca_luad_centers.txt`). Preds: `/data/ryan.kim/nanopath/results/preds/hh_{m2_baseline,m4_*,m5_*}__*.jsonl`.
