#!/usr/bin/env bash
# First bounded launch: indices 0-1 run plain/fair seed 31001 serially.
# Later calibration seeds require an explicit --array=2-7%1 submission.
# Read-only: bash tools/matched_stage_union_20260730/calibration_fm.sbatch --check-only

#SBATCH --job-name=msu_cal_fm
#SBATCH --array=0-1%1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=04:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/msu_cal_fm_%A_%a.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/msu_cal_fm_%A_%a.err

set -euo pipefail
umask 002

REPO=/admin/home/ryan.kim/nt
TRAIN_REPO="$REPO/vendor/matched_stage_train_20260730"
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
CONTRACT="$REPO/tools/matched_stage_union_20260730/contract.py"
OBJECTIVE="$REPO/tools/matched_stage_union_20260730/objectives.py"
INSTRUMENTATION="$REPO/tools/matched_stage_union_20260730/instrumentation.py"
CONFIG_DIR="$REPO/configs_vendor/matched_stage_union_20260730"
SEEDS="$CONFIG_DIR/calibration_seeds.tsv"
OUT_ROOT=/data/ryan.kim/nanopath/reruns/matched_stage_union_20260730

die() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -s "$1" && ! -L "$1" ]] || die "missing/empty/symlink file: $1"; }
require_exec() { [[ -x "$1" ]] || die "missing/non-executable program: $1"; }

preflight() {
  require_exec "$PY"
  for path in "$TRAIN_REPO/train.py" "$CONTRACT" "$OBJECTIVE" "$INSTRUMENTATION" "$SEEDS" \
              "$CONFIG_DIR/calibration_plain.yaml" \
              "$CONFIG_DIR/calibration_faircon.yaml"; do
    require_file "$path"
  done
  "$PY" -m py_compile "$TRAIN_REPO/train.py" "$CONTRACT" "$OBJECTIVE" "$INSTRUMENTATION"
  "$PY" "$CONTRACT" preflight
}

if [[ "${1:-}" == "--check-only" ]]; then
  [[ $# -eq 1 ]] || die "usage: bash $0 --check-only"
  preflight
  echo "OK: default launch is two serial one-GPU tasks for calibration seed 31001"
  exit 0
fi
[[ $# -eq 0 ]] || die "usage: sbatch $0"
preflight

INDEX=${SLURM_ARRAY_TASK_ID:?submit as a Slurm array}
[[ "$INDEX" =~ ^[0-7]$ ]] || die "array index must be 0..7"
SEED_ROW=$((INDEX / 2 + 2))
SEED=$(awk -F $'\t' -v row="$SEED_ROW" 'NR==row {print $2; exit}' "$SEEDS")
[[ "$SEED" =~ ^3100[1-4]$ ]] || die "invalid calibration seed at row $SEED_ROW"
if (( INDEX % 2 == 0 )); then
  ARM=plain
  BASE="$CONFIG_DIR/calibration_plain.yaml"
else
  ARM=faircon
  BASE="$CONFIG_DIR/calibration_faircon.yaml"
fi

FINAL="$OUT_ROOT/calibration/fm/$ARM/seed_$SEED"
[[ ! -e "$FINAL" ]] || die "final output already exists: $FINAL"
mkdir -p "$OUT_ROOT/staging" "$(dirname "$FINAL")"
STAGE=$(mktemp -d \
  "$OUT_ROOT/staging/fm_${ARM}_seed-${SEED}_${SLURM_JOB_ID:-local}.XXXXXX")
EFFECTIVE="$STAGE/effective_config.yaml"
EFFECTIVE_SOURCE=$(mktemp \
  "/tmp/msu_${ARM}_seed-${SEED}_${SLURM_JOB_ID:-local}.XXXXXX.yaml")
cleanup() { rm -f -- "$EFFECTIVE_SOURCE"; }
trap cleanup EXIT

"$PY" "$CONTRACT" make-config \
  --base "$BASE" --output "$EFFECTIVE_SOURCE" --stage "$STAGE" \
  --arm "$ARM" --seed "$SEED"

export WANDB_MODE=offline
cd "$TRAIN_REPO"
"$PY" train.py "$EFFECTIVE_SOURCE" "output_dir=$STAGE"
cp -- "$EFFECTIVE_SOURCE" "$EFFECTIVE"
"$PY" "$CONTRACT" validate-fm --stage "$STAGE" --arm "$ARM" --seed "$SEED"
mv -T "$STAGE" "$FINAL"
echo "COMPLETE: $FINAL"
