#!/bin/bash
# driver_marg.sh: A_marginal arm x 3 folds (3 runs)
PYTHON="/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/python"
CKPT="/data/ryan.kim/nanopath/vendor-contrastive-marginal-siteholdout/latest.pt"
COMMON="--task brca_tp53 --sensitive race --adversary-data task_only \
  --hospital-folds-csv /admin/home/ryan.kim/nt/data/metadata/brca_hospital_folds.csv \
  --inner-splits 5 \
  --tiles-dir /data/TCGA-12K-parquet \
  --molecular-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_molecular_labels.csv \
  --demographics-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv"

source "/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/activate"
cd /admin/home/ryan.kim/nt
mkdir -p /data/ryan.kim/nanopath/results/preds

run_one() {
  local FOLD="$1"
  echo "========== $(date): A_marginal / ${FOLD} =========="
  $PYTHON tools/post_hoc_debias.py \
    --checkpoint "$CKPT" \
    $COMMON \
    --hospital-fold "$FOLD" \
    --method contrastive \
    --lambda-adv 0.0 \
    --dump-predictions "/data/ryan.kim/nanopath/results/preds/hh_A_marginal__brca_tp53__${FOLD}.jsonl" \
    --out "/data/ryan.kim/nanopath/results/hh_A_marginal__${FOLD}.json" \
    || echo "FAILED A_marginal ${FOLD}"
}

for F in F1 F2 F3; do
  run_one "$F"
done

echo "========== $(date): driver_marg.sh COMPLETE =========="