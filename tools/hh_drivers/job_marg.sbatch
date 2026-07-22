#!/bin/bash
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --job-name=hh_panel_marg
#SBATCH --output=/admin/home/ryan.kim/nt/tools/hh_drivers/job_marg.log
#SBATCH --error=/admin/home/ryan.kim/nt/tools/hh_drivers/job_marg.log

set -euo pipefail
source /admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/activate
PY=/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/python
SCRIPT=/admin/home/ryan.kim/nt/tools/post_hoc_debias.py
FM=/data/ryan.kim/nanopath/panel-contrastive-marginal/latest.pt
COMMON="--sensitive race --adversary-data task_only --inner-splits 5 \
  --tiles-dir /data/TCGA-12K-parquet \
  --molecular-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_molecular_labels.csv \
  --demographics-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv \
  --hospital-fold target"

run_arm() {
  local cohort=$1 arm=$2 task=$3 method=$4 lambda=$5 extra="$6" hf_csv=$7
  local json_out="/data/ryan.kim/nanopath/results/hh_panel_${cohort}_${arm}.json"
  local preds_out="/data/ryan.kim/nanopath/results/preds/hh_panel_${cohort}_${arm}__${task}__target.jsonl"
  mkdir -p /data/ryan.kim/nanopath/results/preds /data/ryan.kim/nanopath/results
  echo "=== START $(date) ${cohort} ${arm} ==="
  $PY $SCRIPT \
    --checkpoint "$FM" \
    --task "$task" \
    --method "$method" \
    --lambda-adv "$lambda" \
    --hospital-folds-csv "$hf_csv" \
    --dump-predictions "$preds_out" \
    --out "$json_out" \
    $COMMON $extra || echo "FAILED ${cohort} ${arm}"
  echo "=== END $(date) ${cohort} ${arm} ==="
}

for c in brca ucec coad; do
  if [ "$c" = "brca" ]; then hf="/admin/home/ryan.kim/nt/data/metadata/brca_racepanel_folds.csv"; fi
  if [ "$c" = "ucec" ]; then hf="/admin/home/ryan.kim/nt/data/metadata/ucec_hospital_folds.csv"; fi
  if [ "$c" = "coad" ]; then hf="/admin/home/ryan.kim/nt/data/metadata/coad_hospital_folds.csv"; fi
  run_arm "$c" A_marginal "${c}_tp53" contrastive 0.0 "" "$hf"
done

echo "=== ALL job_marg COMPLETE $(date) ==="
