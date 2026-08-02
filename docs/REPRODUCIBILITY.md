# Reproducing the study

This repository provides the training, data preparation, post-hoc intervention,
and evaluation paths for a matched comparison of fairness intervention timing.
It intentionally contains no result claims or private patient artifacts.

## 1. Install

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[research,test]'
```

## 2. Prepare the data

The default preparation command downloads the pinned 200-shard public
pretraining snapshot, retrieves current public TCGA clinical records from the
NCI GDC API, derives deterministic patient-level task folds, creates the BRCA
downstream holdout, and builds `fino_meta.json`:

```bash
python scripts/prepare_data.py all
```

This creates:

- `data/pretraining_tiles/shard-*.parquet`
- `data/pretraining_tiles/fino_meta.json`
- `data/pretraining_tiles/DATASET_RECEIPT.json`
- `data/metadata/tcga_demographics.csv`
- `data/metadata/downstream_holdout.txt`
- `data/metadata/METADATA_RECEIPT.json`

The two training recipes both use the generated holdout file, so downstream
patients are absent from both the plain and fairness-pretrained encoders.

The downstream TCGA-12K tile mirror is approximately 703 GB and is therefore an
explicit opt-in:

```bash
python scripts/prepare_data.py all --download-downstream
```

It is stored under `data/downstream_tiles`. Downloads are resumable through
`huggingface_hub`. Both tile datasets are pinned to exact repository revisions,
validated for file completeness, Parquet schema, non-empty row counts, and
recorded in machine-readable receipts. For a more expensive first-row payload
check, add `--deep-validate`.

The metadata receipt records the exact GDC query, retrieval time, a canonical
response digest, and SHA-256 digests for each derived artifact. This makes a
run auditable even if the live clinical API changes later.

Individual stages are also available:

```bash
python scripts/prepare_data.py tiles \
  --dataset pretraining --dest data/pretraining_tiles
python scripts/prepare_data.py clinical --holdout-task brca
python scripts/prepare_data.py validate \
  --dataset downstream --dir data/downstream_tiles --deep
```

Data sources and terms:

- Pretraining tiles: [MedARC Nanopath dataset](https://huggingface.co/datasets/medarc/nanopath).
- Downstream tiles: [MedARC TCGA-12K Parquet](https://huggingface.co/datasets/medarc/TCGA-12K-parquet), repackaged from GDC open-access TCGA slides.
- Clinical and demographic fields: [NCI GDC Cases API](https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/).
- Follow the [GDC data-access policy](https://docs.gdc.cancer.gov/Encyclopedia/pages/Data_Access_Policy/) and do not attempt participant re-identification.

The downloader uses only public/open-access endpoints and does not accept or
store a GDC authentication token. Researchers replacing these inputs with
controlled-access data remain responsible for their own approvals and secure
storage.

## 3. Run the matched pretraining arms

```bash
WANDB_MODE=offline python pretraining/train.py configs/pretrain_plain.yaml
WANDB_MODE=offline python pretraining/train.py \
  configs/pretrain_cancer_conditioned_race.yaml
```

The configs have identical datasets, patient exclusions, model, optimization,
augmentations, split seed, and compute budgets. The only substantive difference
is the intervention block. The fairness arm preserves cancer identity while
reducing race signal; it does not receive a downstream diagnosis label.

The first run downloads Meta's public DINOv2 register-token initialization into
the standard PyTorch cache. Subsequent runs reuse it.

## 4. Run the matched post-hoc arm

The post-hoc runner freezes the encoder and trains a plain head (`lambda=0`) and
an intervened head in the same invocation. Use the same `cancer_type` condition
as pretraining and the cross-cancer matched pool—not the downstream diagnosis:

```bash
python scripts/post_hoc_debias.py \
  --checkpoint outputs/pretrain-plain/latest.pt \
  --task brca \
  --tiles-dir data/downstream_tiles \
  --demographics-csv data/metadata/tcga_demographics.csv \
  --sensitive race \
  --adversary-data matched_pool \
  --method contrastive \
  --condition-col cancer_type \
  --lambda-adv 0.1 \
  --dump-predictions outputs/posthoc/predictions.jsonl \
  --out outputs/posthoc/summary.json
```

`--condition-col` accepts any number of categorical strata. Do not replace it
with `--condition-on-label` in the matched design: that alternate flag exposes
the downstream binary target to the fairness branch.

## 5. Evaluate pretraining checkpoints

```bash
python scripts/fairness_eval.py \
  --checkpoint outputs/pretrain-plain/latest.pt \
  --task brca \
  --tiles-dir data/downstream_tiles \
  --demographics-csv data/metadata/tcga_demographics.csv \
  --test-fold 0 \
  --dump-predictions outputs/eval/plain-fold0.jsonl \
  --out outputs/eval/plain-fold0.json
```

Repeat with the fairness-pretrained checkpoint on the same fold. Predictions
are patient-pooled and are suitable for paired tests because both arms use the
same generated patient folds.

## 6. Repeated post-hoc heads

For repeated head seeds, create an experiment declaration such as:

```json
{
  "study": "cancer-conditioned race intervention timing",
  "split_policy": "fixed five-fold patient split",
  "downstream_label_access": "task head only"
}
```

Save it as `outputs/posthoc/runtime-contract.json`, then wrap the same post-hoc
arguments:

```bash
python scripts/reliable_fairness_head.py \
  --head-seed 1 \
  --runtime-contract outputs/posthoc/runtime-contract.json \
  --checkpoint outputs/pretrain-plain/latest.pt \
  --task brca \
  --tiles-dir data/downstream_tiles \
  --demographics-csv data/metadata/tcga_demographics.csv \
  --sensitive race \
  --adversary-data matched_pool \
  --method contrastive \
  --condition-col cancer_type \
  --lambda-adv 0.1 \
  --out outputs/posthoc/seed-1.json
```

The wrapper fixes the outer split, separates split and head RNGs, validates the
embedding cache, records source and checkpoint identities, and writes outputs
atomically.

## Verification boundary

Automated tests cover objective behavior, model/head contracts, deterministic
metadata and fold construction, pinned downloader arguments, Parquet validation,
CLI loading, and absence of personal absolute paths. A scientific run still
requires the documented storage and compute budget; tests do not substitute for
running the complete experiment.
