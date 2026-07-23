#!/bin/bash
#SBATCH --job-name=luad_screen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=main
#SBATCH --time=04:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/luad_screen_%A.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/luad_screen_%A.err
mkdir -p /data/ryan.kim/nanopath/runs/slurm /data/ryan.kim/nanopath/results/preds
PYTHON="/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/python"
source "/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/activate"
cd /admin/home/ryan.kim/nt
echo "========== $(date): LUAD LEAKY SCREEN start =========="
$PYTHON tools/post_hoc_debias.py \
  --checkpoint /data/ryan.kim/nanopath/panel-plain/latest.pt \
  --task luad_tp53 --sensitive race --adversary-data task_only \
  --method contrastive --lambda-adv 0 \
  --hospital-folds-csv /admin/home/ryan.kim/nt/data/metadata/luad_hospital_folds.csv \
  --hospital-fold target --inner-splits 5 \
  --tiles-dir /data/TCGA-12K-parquet \
  --molecular-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_molecular_labels.csv \
  --demographics-csv /admin/home/ryan.kim/nt/data/metadata/tcga12k_demographics.csv \
  --dump-predictions /data/ryan.kim/nanopath/results/preds/hh_LEAKYSCREEN_luad_baseline__luad_tp53__target.jsonl \
  --out /data/ryan.kim/nanopath/results/hh_LEAKYSCREEN_luad_baseline.json \
  && echo "========== $(date): LUAD LEAKY SCREEN COMPLETE ==========" \
  || echo "========== $(date): LUAD LEAKY SCREEN FAILED =========="
