#!/bin/bash
#SBATCH --job-name=m4_1fold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=06:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/m4_1fold_%x_%A.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/m4_1fold_%x_%A.err
# Usage: sbatch --job-name=<tag> m4_posthoc_onefold.sbatch <task> <folds_csv> <fold>
set -uo pipefail
mkdir -p /data/ryan.kim/nanopath/runs/slurm /data/ryan.kim/nanopath/results/preds
TASK="$1"; CSV="$2"; FOLD="$3"
PY="/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/python"
source "/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/activate"
cd /admin/home/ryan.kim/nt
CKPT=/data/ryan.kim/nanopath/vendor-plain-brca-luad-holdout/latest.pt
COMMON="--checkpoint $CKPT --sensitive race --adversary-data task_only \
  --inner-splits 5 --tiles-dir /data/TCGA-12K-parquet \
  --molecular-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_molecular_labels.csv \
  --demographics-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv \
  --hospital-folds-csv /admin/home/ryan.kim/nt/data/metadata/${CSV} --hospital-fold ${FOLD}"
arm() { # method extra armtag
  local METHOD="$1" EXTRA="$2" TAG="$3"
  echo "========== $(date): M4-1fold ${TASK} ${FOLD} ${TAG} =========="
  $PY tools/post_hoc_debias.py $COMMON --task "$TASK" --method "$METHOD" --lambda-adv 1.0 $EXTRA \
    --dump-predictions "/data/ryan.kim/nanopath/results/preds/hh_m4_${TAG}__${TASK}__${FOLD}.jsonl" \
    --out "/data/ryan.kim/nanopath/results/hh_m4_${TAG}__${TASK}__${FOLD}.json" \
    && echo "OK ${TASK} ${FOLD} ${TAG}" || echo "FAILED ${TASK} ${FOLD} ${TAG}"
}
arm contrastive ""                     contrastive_marginal
arm contrastive "--condition-on-label" contrastive_labelcond
arm dann        ""                     dann_marginal
arm dann        "--condition-on-label" dann_labelcond
arm fino        ""                     fino_marginal
arm fino        "--condition-on-label" fino_labelcond
arm pcgrad      ""                     pcgrad_marginal
arm pcgrad      "--condition-on-label" pcgrad_labelcond
echo "========== $(date): M4-1fold ${TASK} ${FOLD} COMPLETE =========="
