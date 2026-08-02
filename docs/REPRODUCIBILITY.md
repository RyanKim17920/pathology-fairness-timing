# Reproducing the study

This repository provides the training, data preparation, post-hoc intervention,
and evaluation paths for studying fairness intervention timing. It intentionally
contains no result claims or private patient artifacts.

## 1. Install and record the environment

Python 3.10 or newer is required. Encoder pretraining is a Linux/CUDA workflow;
CPU execution is supported for unit tests and small plumbing smokes, not for the
documented pretraining run.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[research,test]'
python scripts/environment_receipt.py --out outputs/environment.json
```

The broad dependency bounds allow researchers to select the correct PyTorch
build for their CUDA platform. Retain the environment receipt with every run;
it records exact package, Python, PyTorch, CUDA, cuDNN, accelerator, and source
versions without recording user or filesystem paths.

## 2. Prepare the data

The default command prepares the pinned pretraining snapshot and clinical
metadata. It downloads approximately 119.75 GB, retrieves current public TCGA
clinical records from the NCI GDC API, derives deterministic patient-level task
folds, creates the BRCA downstream holdout, and builds `fino_meta.json`:

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

Both pretraining recipes consume the generated holdout, so the downstream BRCA
patients are excluded from both encoders. The primary timing comparison also
needs the TCGA-12K mirror: 11,368 Parquet files, 24,985,184 image patches, and
approximately 703.08 GB. A complete fresh study therefore needs at least 823 GB
for source tiles, plus checkpoints, caches, and temporary download space:

```bash
python scripts/prepare_data.py all --download-downstream
```

That full command also creates `data/metadata/COHORT_RECEIPT.json`, binding the
labeled GDC cohorts to patients and slides actually present in TCGA-12K. The
primary runner requires it and checks prediction coverage against it.

Downloads are resumable. Both tile datasets are pinned to exact repository
revisions and remote LFS content manifests. Validation enforces names, file and
row counts, byte totals, local size inventories, and exact Arrow field types,
hashes every file against the pinned LFS content manifest, then writes
machine-readable receipts. Initial validation therefore rereads every downloaded
byte once. Add `--deep-validate` for a first-row nonempty-payload check. Later
TCGA runs quickly recheck the receipt's name, size, modification/change-time,
device, and inode fingerprint; changed or replaced files must be fully
revalidated before use.

The metadata receipt records the exact GDC query, retrieval time, canonical
response digest, task/class counts, BRCA subtype inclusion and exclusion counts,
race availability, and SHA-256 digests for every derived artifact. The CSV
retains the diagnosis strings used by the deterministic IDC-versus-ILC mapping.
Because the live API may change, retain the generated CSV and receipt with each
study run. A hash-valid existing snapshot is reused rather than silently
refreshed; use `--refresh-metadata` only as an explicit new-snapshot decision.

Folds are deterministic and patient-level. They stratify each task label by
race when a race/label cell has at least five patients; smaller cells are
distributed within the label stratum. The modeled race groups are White, Black
or African American, and Asian. The CSV also retains the original GDC value in
`race_gdc` and records whether it was mapped, missing, not reported, or outside
the modeled categories. Unmodeled categories are not reassigned. GDC race is an
administrative social variable, not a biological proxy.

Individual stages are also available:

```bash
python scripts/prepare_data.py tiles \
  --dataset pretraining --dest data/pretraining_tiles
python scripts/prepare_data.py clinical --holdout-task brca
python scripts/prepare_data.py validate \
  --dataset downstream --dir data/downstream_tiles --deep
python scripts/prepare_data.py crosswalk
```

Data sources and terms:

- Pretraining tiles: [MedARC Nanopath dataset](https://huggingface.co/datasets/medarc/nanopath).
- Downstream tiles: [MedARC TCGA-12K Parquet](https://huggingface.co/datasets/medarc/TCGA-12K-parquet), repackaged from GDC open-access TCGA slides.
- Clinical fields: [NCI GDC Cases API](https://docs.gdc.cancer.gov/API/Users_Guide/Search_and_Retrieval/).
- Follow the [GDC data-access policy](https://docs.gdc.cancer.gov/Encyclopedia/pages/Data_Access_Policy/) and do not attempt participant re-identification.

The downloader uses only public/open-access endpoints and never accepts a GDC
authentication token. Researchers substituting controlled-access inputs remain
responsible for approvals and secure storage.

The pinned Nanopath repository currently has no dataset card or explicit
license file. Public downloadability is not itself a grant of reuse rights;
researchers must confirm the upstream TCGA/GDC and dataset terms for their use.

## 3. Run the matched pretraining arms

```bash
WANDB_MODE=offline python pretraining/train.py configs/pretrain_plain.yaml
WANDB_MODE=offline python pretraining/train.py \
  configs/pretrain_cancer_conditioned_race.yaml
```

The configs have identical datasets, patient exclusions, models, optimization,
augmentations, split seeds, and compute budgets. Their only substantive
difference is the intervention block. The fairness objective conditions its
race-pair construction on cancer identity; this constraint is not evidence that
cancer utility is preserved. The fairness branch receives no downstream
diagnosis label.

The first run downloads Meta's public DINOv2 register-token initialization and
verifies its pinned SHA-256 digest before loading it. Training refuses to
overwrite a nonempty output directory and verifies the dataset, metadata,
holdout, and FINO receipts before GPU work. It also requires a clean tracked Git
tree, records that commit in checkpoints, and snapshots only tracked source.

A configured resume must match the checkpoint's scientific config, source
commit, input receipts, and RNG record. It restores model, optimizer, counter,
and RNG state, but restarts the shuffled DataLoader iterator. Treat resume as
auditable fault recovery, not bitwise-identical continuation.

## 4. Run a head-matched timing comparison

Use the reliable runner for all three arms so head architecture, head seed,
folds, task supervision, and prediction format remain identical:

1. plain-pretrained encoder with `lambda=0` (control);
2. fairness-pretrained encoder with `lambda=0` (pretraining intervention);
3. plain-pretrained encoder with `lambda>0` (post-hoc intervention).

Create `outputs/timing/runtime-contract.json` prospectively:

```json
{
  "study": "cancer-conditioned race intervention timing",
  "fold_seed": 1337,
  "head_architecture": "linear-relu-dropout-linear",
  "planned_head_seeds": [1],
  "primary_estimand": "posthoc_auc_gap_minus_pretraining_auc_gap",
  "min_fairness_advantage": 0.0,
  "utility_noninferiority_margin": 0.02,
  "sensitive": "race",
  "method": "contrastive",
  "condition_col": "cancer_type",
  "temperature": 0.2,
  "posthoc_lambda": 0.1
}
```

Run the control:

```bash
python scripts/reliable_fairness_head.py \
  --head-seed 1 \
  --runtime-contract outputs/timing/runtime-contract.json \
  --checkpoint outputs/pretrain-plain/latest.pt \
  --task brca \
  --tiles-dir data/downstream_tiles \
  --demographics-csv data/metadata/tcga_demographics.csv \
  --sensitive race \
  --adversary-data matched_pool \
  --method contrastive \
  --condition-col cancer_type \
  --proto-temp 0.2 \
  --lambda-adv 0 \
  --dump-predictions outputs/timing/control-seed1.jsonl \
  --out outputs/timing/control-seed1.json
```

Run the pretraining intervention with the same head:

```bash
python scripts/reliable_fairness_head.py \
  --head-seed 1 \
  --runtime-contract outputs/timing/runtime-contract.json \
  --checkpoint outputs/pretrain-cancer-conditioned-race/latest.pt \
  --task brca \
  --tiles-dir data/downstream_tiles \
  --demographics-csv data/metadata/tcga_demographics.csv \
  --sensitive race \
  --adversary-data matched_pool \
  --method contrastive \
  --condition-col cancer_type \
  --proto-temp 0.2 \
  --lambda-adv 0 \
  --dump-predictions outputs/timing/pretraining-seed1.jsonl \
  --out outputs/timing/pretraining-seed1.json
```

Run the post-hoc intervention on the plain encoder:

```bash
python scripts/reliable_fairness_head.py \
  --head-seed 1 \
  --runtime-contract outputs/timing/runtime-contract.json \
  --checkpoint outputs/pretrain-plain/latest.pt \
  --task brca \
  --tiles-dir data/downstream_tiles \
  --demographics-csv data/metadata/tcga_demographics.csv \
  --sensitive race \
  --adversary-data matched_pool \
  --method contrastive \
  --condition-col cancer_type \
  --proto-temp 0.2 \
  --lambda-adv 0.1 \
  --dump-predictions outputs/timing/posthoc-seed1.jsonl \
  --out outputs/timing/posthoc-seed1.json
```

`--condition-col` may have any number of categorical strata. Do not replace it
with `--condition-on-label`, which exposes the downstream target to the fairness
branch. The auxiliary pool is selected deterministically at the patient level,
round-robin across cancer-condition × sensitive-attribute cells. Its seed,
cell counts, patient-set digest, and slide count are stored in the result.

The reliable runner verifies the metadata receipt and fold seed, separates split
and head RNGs, validates content-addressed pickle-free caches, records source and
checkpoint identities, and writes atomically. Repeat the three commands only
for the prospectively declared head seeds; the example caps the set at five.

## 5. Analyze the paired timing contrast

Pass corresponding prediction files in the same seed order:

```bash
python scripts/analyze_timing.py \
  --runtime-contract outputs/timing/runtime-contract.json \
  --control outputs/timing/control-seed1.jsonl \
  --pretraining outputs/timing/pretraining-seed1.jsonl \
  --posthoc outputs/timing/posthoc-seed1.jsonl \
  --control-results outputs/timing/control-seed1.json \
  --pretraining-results outputs/timing/pretraining-seed1.json \
  --posthoc-results outputs/timing/posthoc-seed1.json \
  --sensitive race \
  --bootstrap 2000 \
  --out outputs/timing/analysis.json
```

For multiple seeds, declare them before running and list every file after its
corresponding arm flag. The
analyzer requires identical patients, labels, and subgroup assignments; reports
each seed; verifies the reliable-run sidecars, prediction hashes, arm lambdas,
checkpoint roles, seed order, and outer folds; uses a hierarchical paired
bootstrap over patients and head seeds; and evaluates one primary timing
estimand plus an AUROC non-inferiority guardrail.
Positive primary values favor pretraining.

The default support thresholds—15 total and five patients per outcome class—are
eligibility heuristics, not a formal power analysis. Age analyses use a fixed
65-year cutoff across folds unless prospectively overridden.

## 6. Optional logistic-regression diagnostic

`scripts/fairness_eval.py` provides a conventional frozen-encoder logistic-
regression probe. It is useful for representation diagnostics but must not be
mixed with the MLP-head outputs in the primary timing comparison. Evaluation
commands require an existing checkpoint. Random initialization is available
only through `--allow-random-init` for plumbing smoke tests and must not be
reported as a scientific run.

## Verification boundary

Automated tests cover executed objective equivalence, model/head contracts,
deterministic metadata and folds, pinned manifests, receipt binding, cache
invalidation, baseline invariance, paired analysis, CLI loading, and absence of
personal paths. Continuous integration runs the suite on Python 3.10 and 3.12.
Tests do not substitute for the documented storage, compute, and complete runs.

The pretraining and post-hoc interventions share the same cancer-conditioned
race-pair definition and temperature, but they operate in different parameter
spaces: pretraining changes the encoder, while post-hoc changes a downstream
head on a frozen encoder and sees the task label through that head. This is an
operational timing study, not assumption-free causal attribution to timing.
The runtime contract and analyzer make the estimand, utility guardrail, seed
aggregation, confidence interval, and single-primary multiplicity policy
explicit; researchers remain responsible for justifying those choices.
