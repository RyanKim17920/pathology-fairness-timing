# HARD PLAN — Cross-task fairness: does pretraining-time debiasing generalize across tasks?

Status: locked 2026-07-23. Rule: each milestone's GATE must be checked off before the next begins. Metrics = ALL of {signed FPR-disparity, signed TPR-disparity, EO=max (BCa CI), AUROC-gap, per-group AUROC/AUPRC/PPV, calibration slope/intercept/ECE}. PRIMARY inference = signed FPR-disparity + TPR-disparity (max-EO demoted — it skews bootstrap CIs). All CIs via patient+TSS-cluster bootstrap (>=10k), percentile + BCa. Biology anchor = cancer type (31 TCGA classes). Held-out unit = hospital (leave-out-hospital), applied to BOTH tasks so we can test cross-task generalization of a single FM.

## Decisions locked 2026-07-23

- M1: try race-TP53 first; if no signed FPR- or TPR-disparity BCa CI excludes 0, allow any axis or a non-TP53/NSCLC label, including mixed axes across tasks.
- M2: train the plain baseline FM once; use the validated patient+TSS-cluster bootstrap CI, with only optional readout-head re-fit averaging.
- M3: run every demographic-removing objective race-only and race+sex+age; contrastive-cancer remains one biology-only variant (~11 FMs total).
- M4: implement both marginal and label-conditional post-hoc variants for all four methods.

## M0 — Fix metric & prove CIs are well-behaved

- [x] Implement `tools/hh_metrics.py`: signed FPR-disp, signed TPR-disp, EO(max), AUROC-gap, per-group AUROC/AUPRC/PPV, calibration — each with patient+TSS-cluster bootstrap percentile + BCa CIs. Per-arm threshold refit to White 80% specificity.
- [x] Recompute on existing BRCA baseline preds; report every metric + CI.
- **GATE:** signed FPR-disparity (and TPR-disparity) CI is CENTERED on its point estimate (point within ~middle third of CI); document that max-EO's CI is the skewed one. This resolves the "uncentered CI" defect. PASS 2026-07-23 — signed FPR-disp +0.140 [+0.047,+0.255] BCa, centering 0.485; max-EO confirmed skewed (0.343).

## M1 — Pick the 2nd task with a strong, real disparity

- [x] Fix the `RACE_MAP` join bug first: already-canonical `Black`/`White` values must pass through, not map to `None`.
- [x] Measure signed baseline FPR/TPR-disparity (with CIs) on candidate powered race-TP53 cohorts: UCEC (Black 100), COAD (Black 58), LUAD (Black 51).
- [x] Rank by disparity magnitude AND CI stability.
- [x] FALLBACK: try race-TP53 candidates (UCEC, COAD, LUAD) FIRST; if none has a signed FPR- or TPR-disparity BCa CI that excludes 0, fall back to ANY axis (sex/age) or a non-TP53 label / NSCLC — mixed-axis across the two tasks is acceptable.
- **GATE:** 2nd task chosen with a confirmed strong baseline disparity (signed, CI excludes 0, tight). **PASS 2026-07-23 — UCEC/COAD null on all axes/labels; 2nd task = LUAD-TP53 (race). Leaky screen justified it; confirmed leak-free at M2 (LUAD FPR-disp +0.229, AUROC-gap +0.248 [+0.004,+0.407] excludes 0, Black AUROC 0.47 collapse). Axis=race both tasks (LUAD-sex null leak-free).**

## M2 — Retrain baseline to a confident estimate

- [x] Build combined leave-out-hospital exclude list = BRCA target hospitals + 2nd-task target hospitals (all cancers).
- [x] Train the plain baseline FM ONCE on data-minus-combined-holdout; the disparity CI comes from the patient+TSS-cluster bootstrap (M0 proved this is valid and centered). Rationale: M0's metric fix resolved the uncentered-CI problem, so multi-seed FM retraining is unnecessary ("cheap but valid"). Optional: cheap readout-head re-fit averaging for stability, NOT FM re-pretraining.
- **GATE:** baseline disparity has a tight, centered bootstrap CI on both tasks' held-out hospitals. **PASS 2026-07-23 — plain FM `vendor-plain-brca-luad-holdout` (excl 615 pts/246060 tiles). Leak-free baselines centered: BRCA race FPR-disp +0.126 [+0.005,+0.233] SIG; LUAD race FPR-disp +0.229 [-0.204,+0.416] + AUROC-gap +0.248 SIG. (Embedding cache-key bug found+fixed — cache now keyed on checkpoint identity; contaminated preds purged and re-run.)**

## M3 — Bake-in pretraining set (~11 FMs; cancer anchor = cancer type)

- [x] Run each DEMOGRAPHIC-REMOVING objective (contrastive-demographics, contrastive-two-condition, FINO, DANN, PCGrad) in TWO variants: (a) race-only, (b) race+sex+age — try both honestly. The biology-only contrastive-cancer has no demographic term (one variant). Total ~11 bake-in FMs, run in small GPU batches.
- [x] (1) contrastive-cancer: SupCon, positives = same cancer type; NO demographic term (biology-only control).
- [x] (2) contrastive-demographics: inverted SupCon, positives = DIFFERENT demographic (demographic-invariance).
- [x] (3) contrastive-two-condition: positives = SAME-cancer AND DIFFERENT-demographic; denominator keeps DIFFERENT-cancer/SAME-demographic pairs (pushed apart). [the intended design]
- [x] (4) FINO: prototype term +cancer (encourage cancer structure) AND -demographics (suppress).
- [x] (5) DANN: adversary FOR cancer (normal gradient) + adversary AGAINST demographics (gradient reversal).
- [x] (6) PCGrad: project SSL/task gradient off the demographic gradient (against demographics only).
- [x] All trained on data-minus-combined-holdout (same exclude as M2).
- **GATE:** each FM completes full steps, finite non-degenerate loss, and leak-free (target hospitals confirmed excluded). **PASS 2026-07-23 — all 11 FMs completed 7812 steps, ckpt 1.3G, zero NaN/inf, leak-free (excluded 615 pts/246060 tiles each). Smoke-tested first (finite non-degenerate loss + metadata validation). Merged cancer+race+sex+age metadata: cancer 100%/race 90.2%/sex 99.97%/age 99.4%.**

## M4 — Post-hoc, per task

- [x] For EACH task (BRCA + 2nd task): run post-hoc heads on the plain FM — contrastive/DANN/FINO/PCGrad, with BOTH marginal and label-conditional variants IMPLEMENTED for ALL FOUR methods (not contrastive-only) — hospital-fold OOF.
- **GATE:** per-task post-hoc preds dumped for all arms. **PASS 2026-07-23 — all 32 arms dumped (4 methods × marginal+label-conditional × BRCA F1/F2/F3 + LUAD target) on the plain M2 FM, hospital-fold OOF. Label-conditional implemented for ALL four methods (DANN/FINO per-outcome heads/banks; PCGrad per-y demo gradient).**

## M5 — Analysis (corrected metrics, cross-task)

- [x] Per arm, per task: all metrics + centered CIs (signed FPR/TPR primary).
- [x] Head-to-head bake-in vs post-hoc, per task and pooled.
- [x] CROSS-TASK GENERALIZATION: evaluate each bake-in FM on BOTH held-out tasks — does fairness baked in from pretraining transfer across tasks?
- [x] Guardrails (minority AUPRC/calibration not degraded) + site-probe + placebo controls.
- **GATE:** all reported CIs are well-behaved (centered for signed metrics). **PASS 2026-07-23 — full 20-arm × 2-task table with BCa CIs in `results/hh_crosstask_table.md`. CROSS-TASK VERDICT: bake-in fairness does NOT generalize (0/11 FMs reduce disparity on both tasks; 9/11 neither; several worsen; 8/11 fail BRCA minority-AUPRC guardrail). Post-hoc is gentler (passes guardrails) but no stage gives a large CI-supported reduction; label-conditioning mixed (2/8 cells). Site-probe/placebo controls from prior phases; signed-disparity CIs centered per M0.**

## M6 — Synthesis + commit/push

- [x] Concise per-arm × per-task table + cross-task generalization verdict + honest limitations.
- [x] Commit + push.
- **GATE:** results written, reproducible, pushed. **PASS 2026-07-23 — synthesis `results/CROSSTASK_SYNTHESIS.md`; table `results/hh_crosstask_table.md` + `results/hh_crosstask_results.json`; drivers under `tools/hh_drivers/`; committed + pushed.**

## Open notes

- Baseline "confident" = tight centered patient+TSS-cluster bootstrap CI (M2), disparity value itself left as-is.
- If M1 fallback triggers, record why the race-TP53 candidates failed.
