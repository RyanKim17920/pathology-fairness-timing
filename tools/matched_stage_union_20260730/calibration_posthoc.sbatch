#!/usr/bin/env bash
# One array task per calibration FM seed; default runs seed 31001 only.

#SBATCH --job-name=msu_cal_head
#SBATCH --array=0%1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=24:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/msu_cal_head_%A_%a.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/msu_cal_head_%A_%a.err

set -euo pipefail
umask 002

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/nanopath-tests/20260710_fino-fairness/.venv/bin/python
RUNNER="$REPO/tools/matched_stage_posthoc_runner.py"
CORE="$REPO/tools/matched_stage_posthoc.py"
ANALYSIS="$REPO/tools/matched_stage_union_20260730/dose_analysis.py"
OBJECTIVE="$REPO/tools/matched_stage_union_20260730/objectives.py"
CONFIG_DIR="$REPO/configs_vendor/matched_stage_union_20260730"
SEEDS="$CONFIG_DIR/calibration_seeds.tsv"
TARGETS="$CONFIG_DIR/targets.tsv"
DOSES="$CONFIG_DIR/posthoc_doses.tsv"
RUNTIME="$CONFIG_DIR/runtime_environment.json"
OUT_ROOT=/data/ryan.kim/nanopath/reruns/matched_stage_union_20260730
TILES=/data/TCGA-12K-parquet
MOLECULAR="$REPO/data/metadata/tcga12k_molecular_labels.csv"
DEMOGRAPHICS="$REPO/data/metadata/tcga12k_demographics.csv"

die() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -s "$1" && ! -L "$1" ]] || die "missing/empty/symlink file: $1"; }

preflight() {
  [[ -x "$PY" ]] || die "missing Python: $PY"
  for path in "$RUNNER" "$CORE" "$ANALYSIS" "$OBJECTIVE" "$SEEDS" "$TARGETS" "$DOSES" \
              "$RUNTIME" "$MOLECULAR" "$DEMOGRAPHICS"; do
    require_file "$path"
  done
  [[ -d "$TILES" && ! -L "$TILES" ]] || die "missing/symlink tile root: $TILES"
  "$PY" -m py_compile "$RUNNER" "$CORE" "$ANALYSIS" "$OBJECTIVE"
  [[ $(awk -F $'\t' 'NR>1{n++}END{print n+0}' "$TARGETS") -eq 4 ]] ||
    die "target matrix must have four rows"
  [[ $(awk -F $'\t' 'NR>1{n++}END{print n+0}' "$DOSES") -eq 6 ]] ||
    die "dose matrix must have six rows"
}

if [[ "${1:-}" == "--check-only" ]]; then
  [[ $# -eq 1 ]] || die "usage: bash $0 --check-only"
  preflight
  echo "OK: default is one GPU for calibration seed 31001 across 4 targets x 6 doses"
  exit 0
fi
[[ $# -eq 0 ]] || die "usage: sbatch $0"
preflight

INDEX=${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}
[[ "$INDEX" =~ ^[0-3]$ ]] || die "array index must be 0..3"
ROW=$((INDEX + 2))
FM_SEED=$(awk -F $'\t' -v row="$ROW" 'NR==row {print $2; exit}' "$SEEDS")
[[ "$FM_SEED" =~ ^3100[1-4]$ ]] || die "invalid FM seed"
HEAD_SEED=$((43000 + FM_SEED - 31000))
CHECKPOINT="$OUT_ROOT/calibration/fm/plain/seed_$FM_SEED/latest.pt"
MANIFEST="$OUT_ROOT/calibration/fm/plain/seed_$FM_SEED/manifest.json"
require_file "$CHECKPOINT"
require_file "$MANIFEST"

FINAL="$OUT_ROOT/calibration/posthoc/seed_$FM_SEED"
[[ ! -e "$FINAL" ]] || die "final output exists: $FINAL"
mkdir -p "$OUT_ROOT/staging" "$OUT_ROOT/calibration/cache/seed_$FM_SEED" \
  "$(dirname "$FINAL")"
STAGE=$(mktemp -d \
  "$OUT_ROOT/staging/posthoc_seed-${FM_SEED}_${SLURM_JOB_ID:-local}.XXXXXX")

while IFS=$'\t' read -r TARGET TASK FOLD FOLDS_REL EXPECTED; do
  [[ -z "$TARGET" || "$TARGET" == "target" ]] && continue
  FOLDS="$REPO/$FOLDS_REL"
  require_file "$FOLDS"
  while IFS=$'\t' read -r DOSE_ORDER DOSE; do
    [[ -z "$DOSE_ORDER" || "$DOSE_ORDER" == "order" ]] && continue
    CELL="$STAGE/$TARGET/dose_$DOSE"
    mkdir -p "$CELL"
    "$PY" "$RUNNER" \
      --head-seed "$HEAD_SEED" --study-task "$TARGET" \
      --checkpoint-manifest "$MANIFEST" \
      --runtime-contract "$RUNTIME" \
      --checkpoint "$CHECKPOINT" --task "$TASK" --sensitive race \
      --adversary-data task_only --method contrastive --lambda-adv "$DOSE" \
      --proto-temp 0.2 --proto-ema 0.9 --race-weight none \
      --hospital-folds-csv "$FOLDS" --hospital-fold "$FOLD" \
      --inner-splits 5 --epochs 20 --hidden 128 --lr 0.001 \
      --batch-size 256 --embed-batch-size 128 \
      --tiles-dir "$TILES" --molecular-csv "$MOLECULAR" \
      --demographics-csv "$DEMOGRAPHICS" \
      --cache-dir "$OUT_ROOT/calibration/cache/seed_$FM_SEED" \
      --dose-logging --out "$CELL/result.json"
  done < "$DOSES"
done < "$TARGETS"

"$PY" "$ANALYSIS" --root "$STAGE" --checkpoint "$CHECKPOINT" \
  --fm-seed "$FM_SEED" --head-seed "$HEAD_SEED"
mv -T "$STAGE" "$FINAL"
echo "COMPLETE: $FINAL"
