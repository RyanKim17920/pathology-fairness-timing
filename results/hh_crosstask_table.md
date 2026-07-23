# M5/M6 Cross-task Fairness Results

Generated 2026-07-23T18:40:56.489624+00:00 using `hh_metrics.py`, 10,000 TSS-cluster bootstrap draws per available prediction set/task. Signed disparity is FPR_Black − FPR_White; AUROC gap is AUROC_White − AUROC_Black.

Black N/events marked ⚠ are underpowered (<15 Black events). `pending` means one or more expected prediction JSONLs were absent.

## Primary table

| Stage | Arm | BRCA signed FPR-disp [BCa] | BRCA AUROC-gap [BCa] | BRCA Black AUROC | BRCA Black N/events | LUAD signed FPR-disp [BCa] | LUAD AUROC-gap [BCa] | LUAD Black AUROC | LUAD Black N/events |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `baseline` | +0.126 [+0.005, +0.233] | +0.051 [-0.055, +0.179] | 0.701 | 118/45 | +0.229 [-0.204, +0.416] | +0.248 [+0.004, +0.407] | 0.473 | 40/26 |
| bake_in | `m3_contrastive_cancer` | +0.099 [+0.003, +0.200] | +0.078 [-0.020, +0.209] | 0.670 | 118/45 | +0.229 [+0.011, +0.420] | +0.232 [-0.007, +0.366] | 0.489 | 40/26 |
| bake_in | `m3_contrastive_demo_race` | +0.085 [+0.024, +0.180] | +0.020 [-0.091, +0.168] | 0.714 | 118/45 | +0.300 [+0.098, +0.492] | +0.274 [+0.042, +0.405] | 0.470 | 40/26 |
| bake_in | `m3_contrastive_demo_racesexage` | +0.099 [+0.011, +0.231] | +0.025 [-0.093, +0.171] | 0.711 | 118/45 | +0.300 [+0.086, +0.499] | +0.275 [-0.014, +0.409] | 0.470 | 40/26 |
| bake_in | `m3_twocond_race` | +0.126 [+0.035, +0.219] | +0.052 [-0.052, +0.170] | 0.710 | 118/45 | +0.229 [-0.200, +0.406] | +0.249 [-0.030, +0.454] | 0.473 | 40/26 |
| bake_in | `m3_twocond_racesexage` | +0.195 [+0.081, +0.301] | +0.044 [-0.044, +0.193] | 0.695 | 118/45 | +0.229 [-0.099, +0.418] | +0.280 [+0.011, +0.455] | 0.437 | 40/26 |
| bake_in | `m3_fino_race` | +0.195 [+0.100, +0.377] | +0.065 [-0.066, +0.214] | 0.679 | 118/45 | +0.300 [+0.140, +0.486] | +0.211 [+0.004, +0.370] | 0.500 | 40/26 |
| bake_in | `m3_fino_racesexage` | +0.154 [+0.065, +0.335] | +0.061 [-0.039, +0.202] | 0.672 | 118/45 | +0.229 [-0.104, +0.384] | +0.229 [-0.028, +0.367] | 0.489 | 40/26 |
| bake_in | `m3_dann_race` | +0.126 [+0.041, +0.224] | +0.071 [-0.033, +0.230] | 0.693 | 118/45 | +0.229 [-0.079, +0.461] | +0.260 [-0.004, +0.413] | 0.470 | 40/26 |
| bake_in | `m3_dann_racesexage` | +0.195 [+0.114, +0.325] | +0.056 [-0.064, +0.204] | 0.695 | 118/45 | +0.300 [+0.096, +0.479] | +0.221 [-0.022, +0.374] | 0.497 | 40/26 |
| bake_in | `m3_pcgrad_race` | +0.167 [+0.060, +0.409] | +0.097 [-0.009, +0.246] | 0.654 | 118/45 | +0.300 [-0.001, +0.559] | +0.248 [+0.042, +0.481] | 0.475 | 40/26 |
| bake_in | `m3_pcgrad_racesexage` | +0.167 [+0.070, +0.344] | +0.055 [-0.076, +0.182] | 0.692 | 118/45 | +0.229 [-0.088, +0.430] | +0.198 [-0.050, +0.357] | 0.527 | 40/26 |
| post_hoc | `contrastive_marginal` | +0.140 [+0.013, +0.265] | +0.046 [-0.058, +0.179] | 0.705 | 118/45 | +0.300 [+0.090, +0.482] | +0.269 [+0.041, +0.423] | 0.462 | 40/26 |
| post_hoc | `contrastive_labelcond` | +0.154 [+0.062, +0.333] | +0.050 [-0.030, +0.204] | 0.719 | 118/45 | +0.300 [+0.086, +0.486] | +0.230 [-0.020, +0.399] | 0.489 | 40/26 |
| post_hoc | `dann_marginal` | +0.154 [+0.065, +0.376] | +0.037 [-0.075, +0.146] | 0.707 | 118/45 | +0.300 [+0.113, +0.500] | +0.242 [-0.059, +0.397] | 0.470 | 40/26 |
| post_hoc | `dann_labelcond` | +0.154 [+0.068, +0.318] | +0.075 [-0.033, +0.182] | 0.712 | 118/45 | +0.300 [+0.086, +0.491] | +0.235 [-0.018, +0.393] | 0.495 | 40/26 |
| post_hoc | `fino_marginal` | +0.126 [+0.040, +0.299] | +0.040 [-0.050, +0.139] | 0.722 | 118/45 | +0.300 [+0.096, +0.511] | +0.227 [-0.015, +0.376] | 0.492 | 40/26 |
| post_hoc | `fino_labelcond` | +0.126 [+0.025, +0.271] | +0.041 [-0.059, +0.175] | 0.717 | 118/45 | +0.229 [+0.021, +0.410] | +0.233 [+0.010, +0.379] | 0.500 | 40/26 |
| post_hoc | `pcgrad_marginal` | +0.099 [-0.023, +0.229] | +0.051 [-0.060, +0.185] | 0.714 | 118/45 | +0.229 [+0.006, +0.431] | +0.223 [-0.011, +0.366] | 0.516 | 40/26 |
| post_hoc | `pcgrad_labelcond` | +0.099 [+0.014, +0.250] | +0.042 [-0.070, +0.176] | 0.718 | 118/45 | +0.157 [-0.202, +0.369] | +0.206 [-0.046, +0.342] | 0.522 | 40/26 |

## Cross-task generalization (bake-in FMs)

Positive `|disp| reduction` means the absolute signed FPR disparity is smaller than baseline. The transfer verdict is point-estimate based; the BCa intervals remain in the primary table.

| Bake-in FM | BRCA FPR / delta | LUAD FPR / delta | Verdict |
|---|---:|---:|---|
| `m3_contrastive_cancer` | +0.099; Δ=-0.027; |disp|↓=+0.027 | +0.229; Δ=+0.000; |disp|↓=+0.000 | BRCA_only |
| `m3_contrastive_demo_race` | +0.085; Δ=-0.041; |disp|↓=+0.041 | +0.300; Δ=+0.071; |disp|↓=-0.071 | BRCA_only |
| `m3_contrastive_demo_racesexage` | +0.099; Δ=-0.027; |disp|↓=+0.027 | +0.300; Δ=+0.071; |disp|↓=-0.071 | BRCA_only |
| `m3_twocond_race` | +0.126; Δ=+0.000; |disp|↓=+0.000 | +0.229; Δ=+0.000; |disp|↓=+0.000 | neither_task |
| `m3_twocond_racesexage` | +0.195; Δ=+0.068; |disp|↓=-0.068 | +0.229; Δ=+0.000; |disp|↓=+0.000 | neither_task |
| `m3_fino_race` | +0.195; Δ=+0.068; |disp|↓=-0.068 | +0.300; Δ=+0.071; |disp|↓=-0.071 | neither_task |
| `m3_fino_racesexage` | +0.154; Δ=+0.027; |disp|↓=-0.027 | +0.229; Δ=+0.000; |disp|↓=+0.000 | neither_task |
| `m3_dann_race` | +0.126; Δ=+0.000; |disp|↓=+0.000 | +0.229; Δ=+0.000; |disp|↓=+0.000 | neither_task |
| `m3_dann_racesexage` | +0.195; Δ=+0.068; |disp|↓=-0.068 | +0.300; Δ=+0.071; |disp|↓=-0.071 | neither_task |
| `m3_pcgrad_race` | +0.167; Δ=+0.041; |disp|↓=-0.041 | +0.300; Δ=+0.071; |disp|↓=-0.071 | neither_task |
| `m3_pcgrad_racesexage` | +0.167; Δ=+0.041; |disp|↓=-0.041 | +0.229; Δ=+0.000; |disp|↓=+0.000 | neither_task |

## Bake-in vs post-hoc, by method family

Comparisons use change in signed FPR disparity from baseline and absolute-disparity reduction (positive is better).

| Task | Family | Bake-in variants | Post-hoc marginal | Post-hoc label-conditional | Label-conditional beats marginal? |
|---|---|---|---:|---:|---|
| BRCA-TP53 | contrastive | `m3_contrastive_cancer`: Δ=-0.027; |disp|↓=+0.027<br>`m3_contrastive_demo_race`: Δ=-0.041; |disp|↓=+0.041<br>`m3_contrastive_demo_racesexage`: Δ=-0.027; |disp|↓=+0.027<br>`m3_twocond_race`: Δ=+0.000; |disp|↓=+0.000<br>`m3_twocond_racesexage`: Δ=+0.068; |disp|↓=-0.068 | Δ=+0.014; |disp|↓=-0.014 | Δ=+0.027; |disp|↓=-0.027 | no |
| BRCA-TP53 | dann | `m3_dann_race`: Δ=+0.000; |disp|↓=+0.000<br>`m3_dann_racesexage`: Δ=+0.068; |disp|↓=-0.068 | Δ=+0.027; |disp|↓=-0.027 | Δ=+0.027; |disp|↓=-0.027 | no |
| BRCA-TP53 | fino | `m3_fino_race`: Δ=+0.068; |disp|↓=-0.068<br>`m3_fino_racesexage`: Δ=+0.027; |disp|↓=-0.027 | Δ=+0.000; |disp|↓=+0.000 | Δ=+0.000; |disp|↓=+0.000 | no |
| BRCA-TP53 | pcgrad | `m3_pcgrad_race`: Δ=+0.041; |disp|↓=-0.041<br>`m3_pcgrad_racesexage`: Δ=+0.041; |disp|↓=-0.041 | Δ=-0.027; |disp|↓=+0.027 | Δ=-0.027; |disp|↓=+0.027 | no |
| LUAD-TP53 | contrastive | `m3_contrastive_cancer`: Δ=+0.000; |disp|↓=+0.000<br>`m3_contrastive_demo_race`: Δ=+0.071; |disp|↓=-0.071<br>`m3_contrastive_demo_racesexage`: Δ=+0.071; |disp|↓=-0.071<br>`m3_twocond_race`: Δ=+0.000; |disp|↓=+0.000<br>`m3_twocond_racesexage`: Δ=+0.000; |disp|↓=+0.000 | Δ=+0.071; |disp|↓=-0.071 | Δ=+0.071; |disp|↓=-0.071 | no |
| LUAD-TP53 | dann | `m3_dann_race`: Δ=+0.000; |disp|↓=+0.000<br>`m3_dann_racesexage`: Δ=+0.071; |disp|↓=-0.071 | Δ=+0.071; |disp|↓=-0.071 | Δ=+0.071; |disp|↓=-0.071 | no |
| LUAD-TP53 | fino | `m3_fino_race`: Δ=+0.071; |disp|↓=-0.071<br>`m3_fino_racesexage`: Δ=+0.000; |disp|↓=+0.000 | Δ=+0.071; |disp|↓=-0.071 | Δ=+0.000; |disp|↓=+0.000 | yes |
| LUAD-TP53 | pcgrad | `m3_pcgrad_race`: Δ=+0.071; |disp|↓=-0.071<br>`m3_pcgrad_racesexage`: Δ=+0.000; |disp|↓=+0.000 | Δ=+0.000; |disp|↓=+0.000 | Δ=-0.071; |disp|↓=+0.071 | yes |

## Guardrails

An evaluated arm fails if Black AUPRC drops by >0.02 absolute, Black ECE rises by >0.02 absolute, or overall task AUROC is ≤0.60. These are the study's prior guardrails; underpowering is reported separately.

| Task | Arm | Black AUPRC Δ | Black ECE Δ | Overall AUROC Δ | Status | Notes |
|---|---|---:|---:|---:|---|---|
| BRCA-TP53 | `m3_contrastive_cancer` | -0.053 | +0.031 | -0.015 | fail | Black AUPRC fell >0.02 absolute from baseline; Black ECE worsened >0.02 absolute from baseline |
| LUAD-TP53 | `m3_contrastive_cancer` | +0.008 | -0.044 | -0.001 | pass | passes |
| BRCA-TP53 | `m3_contrastive_demo_race` | -0.015 | +0.012 | -0.008 | pass | passes |
| LUAD-TP53 | `m3_contrastive_demo_race` | -0.004 | -0.022 | +0.019 | pass | passes |
| BRCA-TP53 | `m3_contrastive_demo_racesexage` | -0.020 | +0.032 | -0.007 | fail | Black ECE worsened >0.02 absolute from baseline |
| LUAD-TP53 | `m3_contrastive_demo_racesexage` | -0.007 | -0.018 | +0.023 | pass | passes |
| BRCA-TP53 | `m3_twocond_race` | -0.017 | +0.022 | +0.006 | fail | Black ECE worsened >0.02 absolute from baseline |
| LUAD-TP53 | `m3_twocond_race` | +0.001 | -0.025 | -0.001 | pass | passes |
| BRCA-TP53 | `m3_twocond_racesexage` | -0.045 | +0.026 | -0.005 | fail | Black AUPRC fell >0.02 absolute from baseline; Black ECE worsened >0.02 absolute from baseline |
| LUAD-TP53 | `m3_twocond_racesexage` | -0.006 | -0.027 | -0.012 | pass | passes |
| BRCA-TP53 | `m3_fino_race` | -0.019 | -0.008 | -0.010 | pass | passes |
| LUAD-TP53 | `m3_fino_race` | +0.021 | -0.040 | -0.012 | pass | passes |
| BRCA-TP53 | `m3_fino_racesexage` | -0.024 | -0.004 | -0.023 | fail | Black AUPRC fell >0.02 absolute from baseline |
| LUAD-TP53 | `m3_fino_racesexage` | +0.004 | -0.021 | -0.008 | pass | passes |
| BRCA-TP53 | `m3_dann_race` | -0.031 | -0.000 | +0.007 | fail | Black AUPRC fell >0.02 absolute from baseline |
| LUAD-TP53 | `m3_dann_race` | +0.005 | -0.088 | +0.007 | pass | passes |
| BRCA-TP53 | `m3_dann_racesexage` | -0.021 | +0.017 | -0.002 | fail | Black AUPRC fell >0.02 absolute from baseline |
| LUAD-TP53 | `m3_dann_racesexage` | +0.018 | -0.062 | -0.001 | pass | passes |
| BRCA-TP53 | `m3_pcgrad_race` | -0.058 | +0.001 | -0.022 | fail | Black AUPRC fell >0.02 absolute from baseline |
| LUAD-TP53 | `m3_pcgrad_race` | +0.020 | -0.094 | +0.001 | pass | passes |
| BRCA-TP53 | `m3_pcgrad_racesexage` | -0.031 | -0.007 | -0.008 | fail | Black AUPRC fell >0.02 absolute from baseline |
| LUAD-TP53 | `m3_pcgrad_racesexage` | +0.031 | -0.084 | +0.010 | pass | passes |
| BRCA-TP53 | `contrastive_marginal` | -0.013 | +0.009 | +0.003 | pass | passes |
| LUAD-TP53 | `contrastive_marginal` | -0.008 | -0.008 | +0.008 | pass | passes |
| BRCA-TP53 | `contrastive_labelcond` | +0.005 | +0.005 | +0.017 | pass | passes |
| LUAD-TP53 | `contrastive_labelcond` | +0.005 | -0.021 | +0.000 | pass | passes |
| BRCA-TP53 | `dann_marginal` | -0.011 | -0.006 | -0.001 | pass | passes |
| LUAD-TP53 | `dann_marginal` | -0.001 | -0.040 | -0.011 | pass | passes |
| BRCA-TP53 | `dann_labelcond` | -0.018 | +0.003 | +0.025 | pass | passes |
| LUAD-TP53 | `dann_labelcond` | +0.010 | +0.011 | +0.007 | pass | passes |
| BRCA-TP53 | `fino_marginal` | -0.005 | -0.008 | +0.011 | pass | passes |
| LUAD-TP53 | `fino_marginal` | +0.007 | +0.026 | -0.005 | fail | Black ECE worsened >0.02 absolute from baseline |
| BRCA-TP53 | `fino_labelcond` | +0.005 | +0.002 | +0.009 | pass | passes |
| LUAD-TP53 | `fino_labelcond` | +0.006 | +0.036 | +0.007 | fail | Black ECE worsened >0.02 absolute from baseline |
| BRCA-TP53 | `pcgrad_marginal` | +0.001 | +0.009 | +0.012 | pass | passes |
| LUAD-TP53 | `pcgrad_marginal` | +0.024 | -0.025 | +0.015 | pass | passes |
| BRCA-TP53 | `pcgrad_labelcond` | +0.009 | +0.006 | +0.011 | pass | passes |
| LUAD-TP53 | `pcgrad_labelcond` | +0.026 | -0.025 | +0.009 | pass | passes |

## Prediction inventory

Present (40 set/task cells): `baseline / BRCA-TP53`, `baseline / LUAD-TP53`, `m3_contrastive_cancer / BRCA-TP53`, `m3_contrastive_cancer / LUAD-TP53`, `m3_contrastive_demo_race / BRCA-TP53`, `m3_contrastive_demo_race / LUAD-TP53`, `m3_contrastive_demo_racesexage / BRCA-TP53`, `m3_contrastive_demo_racesexage / LUAD-TP53`, `m3_twocond_race / BRCA-TP53`, `m3_twocond_race / LUAD-TP53`, `m3_twocond_racesexage / BRCA-TP53`, `m3_twocond_racesexage / LUAD-TP53`, `m3_fino_race / BRCA-TP53`, `m3_fino_race / LUAD-TP53`, `m3_fino_racesexage / BRCA-TP53`, `m3_fino_racesexage / LUAD-TP53`, `m3_dann_race / BRCA-TP53`, `m3_dann_race / LUAD-TP53`, `m3_dann_racesexage / BRCA-TP53`, `m3_dann_racesexage / LUAD-TP53`, `m3_pcgrad_race / BRCA-TP53`, `m3_pcgrad_race / LUAD-TP53`, `m3_pcgrad_racesexage / BRCA-TP53`, `m3_pcgrad_racesexage / LUAD-TP53`, `contrastive_marginal / BRCA-TP53`, `contrastive_marginal / LUAD-TP53`, `contrastive_labelcond / BRCA-TP53`, `contrastive_labelcond / LUAD-TP53`, `dann_marginal / BRCA-TP53`, `dann_marginal / LUAD-TP53`, `dann_labelcond / BRCA-TP53`, `dann_labelcond / LUAD-TP53`, `fino_marginal / BRCA-TP53`, `fino_marginal / LUAD-TP53`, `fino_labelcond / BRCA-TP53`, `fino_labelcond / LUAD-TP53`, `pcgrad_marginal / BRCA-TP53`, `pcgrad_marginal / LUAD-TP53`, `pcgrad_labelcond / BRCA-TP53`, `pcgrad_labelcond / LUAD-TP53`

Pending (0 set/task cells): none
