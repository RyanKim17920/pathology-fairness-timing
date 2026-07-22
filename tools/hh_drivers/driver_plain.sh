#!/bin/bash
# driver_plain.sh: FM_plain arms x 3 folds (18 runs)
# Caches FM_plain embeddings and reuses across all 18 runs.
PYTHON="/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/python"
CKPT="/data/ryan.kim/nanopath/vendor-plain-siteholdout/latest.pt"
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
  local ARM="$1" FOLD="$2" METHOD="$3" LAMBDA="$4" EXTRA="$5"
  echo "========== $(date): ${ARM} / ${FOLD} =========="
  $PYTHON tools/post_hoc_debias.py \
    --checkpoint "$CKPT" \
    $COMMON \
    --hospital-fold "$FOLD" \
    --method "$METHOD" \
    --lambda-adv "$LAMBDA" \
    $EXTRA \
    --dump-predictions "/data/ryan.kim/nanopath/results/preds/hh_${ARM}__brca_tp53__${FOLD}.jsonl" \
    --out "/data/ryan.kim/nanopath/results/hh_${ARM}__${FOLD}.json" \
    || echo "FAILED ${ARM} ${FOLD}"
}

for F in F1 F2 F3; do
  run_one "baseline"               "$F" "contrastive" "0.0" ""
  run_one "B_contrastive_marginal" "$F" "contrastive" "1.0" ""
  run_one "B_contrastive_labelcond" "$F" "contrastive" "1.0" "--condition-on-label"
  run_one "B_dann_marginal"        "$F" "dann"        "1.0" ""
  run_one "B_fino_marginal"        "$F" "fino"        "1.0" ""
  run_one "B_pcgrad_marginal"      "$F" "pcgrad"      "1.0" ""
done

echo "========== $(date): driver_plain.sh COMPLETE =========="