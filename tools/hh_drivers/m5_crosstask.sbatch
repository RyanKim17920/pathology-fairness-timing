#!/bin/bash
#SBATCH --job-name=m5_xtask
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=06:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/m5_%x_%A.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/m5_%x_%A.err
# Usage: sbatch --job-name=<FMNAME> m5_crosstask.sbatch <checkpoint> <FMNAME>
set -uo pipefail
mkdir -p /data/ryan.kim/nanopath/runs/slurm /data/ryan.kim/nanopath/results/preds
CKPT="$1"; FM="$2"
PY="/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/python"
source "/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/activate"
cd /admin/home/ryan.kim/nt
COMMON="--checkpoint $CKPT --method contrastive --lambda-adv 0.0 --sensitive race --adversary-data task_only \
  --inner-splits 5 --tiles-dir /data/TCGA-12K-parquet \
  --molecular-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_molecular_labels.csv \
  --demographics-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv"
run() { # task csv fold tag
  local TASK="$1" CSV="$2" FOLD="$3" TAG="$4"
  echo "========== $(date): M5 ${FM} ${TASK} ${FOLD} =========="
  $PY tools/post_hoc_debias.py $COMMON --task "$TASK" \
    --hospital-folds-csv "/admin/home/ryan.kim/nt/data/metadata/${CSV}" --hospital-fold "$FOLD" \
    --dump-predictions "/data/ryan.kim/nanopath/results/preds/hh_m5_${FM}__${TASK}__${TAG}.jsonl" \
    --out "/data/ryan.kim/nanopath/results/hh_m5_${FM}__${TASK}__${TAG}.json" \
    && echo "OK ${FM} ${TASK} ${FOLD}" || echo "FAILED ${FM} ${TASK} ${FOLD}"
}
# plain readout head on THIS bake-in FM, both held-out tasks
run brca_tp53 brca_hospital_folds.csv F1 F1
run brca_tp53 brca_hospital_folds.csv F2 F2
run brca_tp53 brca_hospital_folds.csv F3 F3
run luad_tp53 luad_hospital_folds.csv target target
echo "========== $(date): M5 ${FM} COMPLETE =========="
