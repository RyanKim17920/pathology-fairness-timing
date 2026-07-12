# pathology-fairness-timing

Pretraining-Time vs. Post-Hoc Fairness for Histopathology AI Models.

Head-to-head evaluation of **when** a debiasing intervention is applied — during
self-supervised pretraining vs. post-hoc on a frozen encoder — holding the
**method** fixed. Built on the nanopath-JEPA (DINOv2/iBOT) pipeline.

## Experimental grid
- **Methods:** FINO (prototype bank) · DANN (CE adversary + gradient reversal) ·
  Contrastive (fair-SupCon) · PCGrad (gradient projection)
- **Timing:** pretraining-time · post-hoc
- **Race weighting:** random · inverse-frequency (ablation axis)
- **Post-hoc data regime:** task-only · matched-pool
- Baseline: no debiasing.

## Data
- **Train:** TCGA-12K (9,389 patients) — SSL corpus; demographics (race/sex/age)
  drive the training-time constraints.
- **Validate (external, OOD):** CPTAC — lung (NSCLC, White-vs-Asian), GBM, CCRCC.
  Tiled locally from on-disk SVS; stored on the private HF dataset repo
  `ryankim17920/nanopath-fairness-tiles` (tiles never committed here).
- **Labels:** subtype (NSCLC), TP53 / grade / stage whole-cohort molecular labels.
- Metadata + leak-free race-stratified folds under `data/metadata/`.

## Layout
- `tools/fairness_eval.py` — per-subgroup AUROC / AUCΔ / ES-AUC / ECEΔ eval harness.
- `tools/post_hoc_debias.py` — post-hoc debiasing head (all 4 methods, 2 data regimes).
- `tools/tile_cptac_ondisk.py` — CPTAC SVS → parquet tiler.
- `tools/hf_tiles.py` — push/pull tile cohorts to the private HF dataset repo.
- `data/metadata/` — demographics, molecular labels, folds, holdout lists.

Pretraining-time method code + configs live in the nanopath worktree
`nanopath-tests/20260710_fino-fairness` (branch `exp/20260710-fino-fairness`).

## Data provenance & licensing
- **CPTAC** whole-slide images (source of the external tiles) are from The Cancer
  Imaging Archive (TCIA) collections CPTAC-LUAD, CPTAC-LSCC, CPTAC-GBM,
  CPTAC-CCRCC, released under **CC BY 3.0**. Tiles are a derivative (tiled +
  packed to parquet) redistributed under the same license with attribution; they
  live in the public HF dataset `ryankim17920/nanopath-fairness-tiles`, not here.
- **Demographics** (case-level race/sex/age) are from the NCI Genomic Data Commons
  (GDC) **open-access** clinical data; de-identified, no PHI.
- **No TCGA data** is redistributed in this repo or the HF dataset.
- Please cite CPTAC, the TCIA collection DOIs, and the GDC when using this work.
