---
license: cc-by-3.0
task_categories:
- image-classification
tags:
- medical
- pathology
- histopathology
- fairness
- cptac
pretty_name: nanopath fairness tiles (CPTAC external validation)
size_categories:
- 100K<n<1M
---

# nanopath-fairness-tiles

Pre-tiled histopathology patches from **CPTAC** whole-slide images, used as the
**external out-of-distribution validation set** for a study on pretraining-time
vs. post-hoc fairness in histopathology foundation models.

## Contents
Per-cohort folders, each `slides_full/<slide_id>.parquet` (one row per tile:
`case_id, slide_id, tile_idx, image`) + `labels.tsv`:

| cohort | organ / task | slides |
|---|---|---|
| `cptac_lung`  | NSCLC — LUAD vs LSCC subtype | 604 |
| `cptac_gbm`   | GBM — TP53 mutation status | 243 |
| `cptac_ccrcc` | ccRCC — PBRM1 mutation status | 245 |

Tiles: 512×512 px at 0.5 MPP (~20×), JPEG q95, background dropped
(brightness ≥ 230). Full-slide tiling (uncapped — every tissue tile kept).
`metadata/` holds case-level demographics (race/sex/age) and CV folds.

## Source & attribution (required by CC BY 3.0)
Images derive from the **Clinical Proteomic Tumor Analysis Consortium (CPTAC)**
collections hosted on **The Cancer Imaging Archive (TCIA)**
(CPTAC-LUAD, CPTAC-LSCC, CPTAC-GBM, CPTAC-CCRCC), released under
**CC BY 3.0**. This dataset is a derivative: slides were tiled and packed into
parquet; no image content was otherwise altered. Case-level demographics are
from the NCI **Genomic Data Commons (GDC)** open-access clinical data.

Please cite CPTAC, the relevant TCIA collection DOIs, and the GDC when using this
data, and retain this attribution. Original data © their respective providers
under CC BY 3.0.

## Not included
No TCGA data. No protected health information (all sources de-identified).
