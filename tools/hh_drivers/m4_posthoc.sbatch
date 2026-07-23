#!/bin/bash
#SBATCH --job-name=m4_posthoc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=08:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/m4_posthoc_%A.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/m4_posthoc_%A.err
set -uo pipefail
mkdir -p /data/ryan.kim/nanopath/runs/slurm /data/ryan.kim/nanopath/results/preds
PY="/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/python"
source "/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/activate"
cd /admin/home/ryan.kim/nt
# Post-hoc heads on the PLAIN M2 FM (reuses cached embeddings from the M2 gate eval).
CKPT=/data/ryan.kim/nanopath/vendor-plain-brca-luad-holdout/latest.pt
COMMON="--checkpoint $CKPT --sensitive race --adversary-data task_only \
  --inner-splits 5 --tiles-dir /data/TCGA-12K-parquet \
  --molecular-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_molecular_labels.csv \
  --demographics-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv"

arm() { # task csv fold method extra armtag
  local TASK="$1" CSV="$2" FOLD="$3" METHOD="$4" EXTRA="$5" TAG="$6"
  echo "========== $(date): M4 ${TASK} ${FOLD} ${TAG} =========="
  $PY tools/post_hoc_debias.py $COMMON --task "$TASK" \
    --hospital-folds-csv "/admin/home/ryan.kim/nt/data/metadata/${CSV}" \
    --hospital-fold "$FOLD" --method "$METHOD" --lambda-adv 1.0 $EXTRA \
    --dump-predictions "/data/ryan.kim/nanopath/results/preds/hh_m4_${TAG}__${TASK}__${FOLD}.jsonl" \
    --out "/data/ryan.kim/nanopath/results/hh_m4_${TAG}__${TASK}__${FOLD}.json" \
    && echo "OK ${TASK} ${FOLD} ${TAG}" || echo "FAILED ${TASK} ${FOLD} ${TAG}"
}

run_task() { # task csv fold
  local T="$1" C="$2" F="$3"
  arm "$T" "$C" "$F" contrastive ""                     contrastive_marginal
  arm "$T" "$C" "$F" contrastive "--condition-on-label" contrastive_labelcond
  arm "$T" "$C" "$F" dann        ""                     dann_marginal
  arm "$T" "$C" "$F" dann        "--condition-on-label" dann_labelcond
  arm "$T" "$C" "$F" fino        ""                     fino_marginal
  arm "$T" "$C" "$F" fino        "--condition-on-label" fino_labelcond
  arm "$T" "$C" "$F" pcgrad      ""                     pcgrad_marginal
  arm "$T" "$C" "$F" pcgrad      "--condition-on-label" pcgrad_labelcond
}

run_task brca_tp53 brca_hospital_folds.csv F1
run_task brca_tp53 brca_hospital_folds.csv F2
run_task brca_tp53 brca_hospital_folds.csv F3
run_task luad_tp53 luad_hospital_folds.csv target
echo "========== $(date): M4 POST-HOC COMPLETE =========="
