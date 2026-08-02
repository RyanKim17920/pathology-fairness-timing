# Fixed-five diagnosis-free representation audit result

Recorded on `2026-08-02` after Slurm job `369288` completed. This result uses
the five already-accepted FM seeds (`32001`--`32005`) and introduces no new FM
seed or stage-training execution.

## Run and artifact integrity

- Job name: `main_1gpu`
- Allocation: one task, one GPU, 16 CPUs; no array or concurrent study job
- State: `COMPLETED`, exit `0:0`, elapsed `05:39:43`, peak RSS about `4.8 GiB`
- Stderr: empty
- Population boundary: diagnosis/outcome-free sanitized fields only:
  `patient_id`, `cancer`, `race`, and `tss`
- Metric topology: 280 race-probe cells and 70 cancer-probe cells
- Raw metric SHA-256:
  `f93777719f738e1cf6f6a12eb350c8189cade9701e6aad535026987c2b0dc601`
- Analysis SHA-256:
  `0d5c116cba00eb425b83ba4b7236efd24f709b17080d10de981f554287439633`
- Packaged independent-verification SHA-256:
  `c037b2694cf88ab066a1d3fb164edb79f41803d73904f9b61f5b7c8a85f503ee`

Two independent subagents and the primary agent validated the complete raw
artifact. One validator recomputed all 280 race AUCs and all 70 cancer AUCs
from the held-out predictions. All recomputed values, hashes, cell topology,
compact-cache identities, and gate decisions agreed with the packaged report.
The report's `status=pass` denotes verification success, not scientific-gate
success.

## Frozen primary result

Positive leakage reduction means the candidate removed oriented linear-probe
race signal relative to its matched control. The frozen material-effect
threshold was `0.05` in every cancer by view stratum at both patient and tile
levels, in at least four of five seeds and at the five-seed median. Cancer
preservation required AUROC loss no worse than `0.02` in both views.

| Contrast | Patient activity | Tile activity | Cancer preservation | Classification |
|---|---:|---:|---:|---|
| `E_fair - E_plain` | fail | fail | pass | inactive |
| `A_temp_fair - A_temp_plain` | fail | fail | pass | inactive |
| `P - B` | fail | fail | pass | inactive |
| `H - B` | fail | fail | pass | inactive |

Across the 40 seed/cancer/view/level cells per contrast, the mean leakage
reductions were `-0.004307` for E, `-0.009672` for temporary A, `-0.005295`
for P, and `+0.001254` for H. The numbers of individual cells reaching `0.05`
were respectively `0/40`, `1/40`, `3/40`, and `0/40`. These individual-cell
counts are descriptive; the frozen stratum gate is stricter. Every cancer
preservation view passed in all five seeds.

This is a material-effect result, not a significance claim. There was no
optional-seed or run-until-significance procedure.

## Mechanism evidence and likely explanation

The logs show that the fairness path was reached rather than an obvious no-op:

- every fair slot-1 and H stream recorded all `781` steps per seed;
- the weighted fair/main gradient-norm ratio averaged `0.10937` at slot 1 and
  `0.10213` at H;
- fair/main gradient cosine averaged `0.94113` at slot 1 and `0.98723` at H,
  with zero logged conflicts in either set of 3,905 steps; and
- parameters moved: the five-seed median candidate/control Frobenius ratios
  were `0.00483` for E, `0.13260` for temporary A, `0.15666` for P/B, and
  `0.01756` for H/B.

Thus the fairness-bearing paths had finite logged fairness-gradient signal and
exposure plus nonzero candidate/control displacement, while their logged
h-space gradient was nearly collinear with the cancer objective and they did
not materially remove race-decodable geometry. Raw gradient norms do not
recover accumulated fairness-specific encoder updates, so they do not prove a
realized optimization dose or a causal mechanism. The evidence is consistent
with objective redundancy/dilution as the leading hypothesis for the earlier
tiny and unstable pre-training-versus-post-hoc result. The directly supported
conclusion is that neither timing produced an active representation-level
intervention. This does not establish that corrected-mask FAIR-Path, DANN, or
fairness methods in general are inactive.

## Frozen branch and terminal feasibility decision

The value-blind objective-mask erratum at commit `c94f688` predates the metric
artifact. Since temporary A has neither patient nor tile activity and its
cancer preservation passes, its exact branch precedence selects route 7.

The source-bound receipt is
`FIXED5_REPRESENTATION_AUDIT_BRANCH_RECEIPT.json`, SHA-256
`20b0b19aeb0831e1b5fecd49397eb681eb8f5b01f6b8b1ba06da115c0b3a8322`.
It records:

- selected route: `7`, `temporary_A_inactive`;
- H-mask feasibility: `false`, `inadequate_headroom`; and
- action: `no_training_inadequate_headroom`.

The failure is structural:

- target seed `32001` has only `3/8` B cells at or above `0.05`; all eight were
  required. Its B minimum is `0.001141` and median is `0.040799`;
- across five seeds, B headroom passes `2/8` strata and fails `6/8`; and
- H passes `7/8` strata, but BRCA/view-A/tile has median `0.028652`, below the
  required `0.03`.

Because oriented leakage cannot be negative, demanding a `0.05` reduction
where baseline B leakage is below `0.05` is mathematically impossible. The
frozen rule therefore forbids H-mask training, seed substitution, or relaxing
the effect-size gate. No follow-up GPU job was submitted.

## What is concluded, and what is not

The tangible verified conclusion is:

> The implemented 255-candidate all-pairs FairCon package was inactive at E,
> temporary A, P, and H under a diagnosis-free fixed-five audit, despite finite
> logged fairness-gradient signal/exposure, nonzero candidate/control
> displacement, and preserved cancer information. The existing cohort and
> controls also lack the predeclared headroom needed to test a material `0.05`
> corrected-mask improvement.

This does not test `relation_consistent_mask_v1`, because its feasibility gate
stopped the experiment before training. It also does not prove that post-hoc
timing is intrinsically superior; it explains why the observed timing
difference was not a reliable method effect.

## Strongest bounded next study recommendation

This is a new prospective recommendation, not authorization under
`c94f688`; that frozen current-cohort continuation terminated above.

Do not redefine the current endpoint by pooling strata or scaling the threshold
after opening these values. Before spending any of the five allowed new stage
paths, obtain an external, patient- and site-disjoint BRCA+LUAD Black/White
panel and run a baseline-only eligibility screen. Prospectively require:

1. at least 40 minority patients and four sites per cancer;
2. both fixed tile views;
3. B leakage at least `0.08` in every cancer/view/patient-or-tile stratum in at
   least four of five seeds and at the median;
4. cancer-probe AUROC at least `0.80`; and
5. one downstream endpoint selected from availability and prevalence before
   candidate outputs are opened.

Only an eligible panel should receive exactly five serial
`relation_consistent_mask_v1` H-stage paths, one per accepted seed, each in a
one-GPU job named `main_1gpu`, with no interim peeking. The primary mechanism
gate remains an absolute `0.05` leakage reduction in all eight strata in at
least four of five seeds and at the median, with cancer loss no worse than
`0.02`. Downstream transfer is a separate non-harm/utility claim and cannot
rescue a failed representation gate.

That external study is not currently executable from repository data. The
available CPTAC minority counts are inadequate, and previously used cancers
are not an untouched confirmation panel. Acquiring and authorizing a suitable
external panel is the next blocking input. Under the five-path cap, the next
claim can test corrected post-hoc mechanism efficacy and transfer, not a new
five-seed pre-training-versus-post-hoc timing comparison, which would require
two new arms and ten paths.
