#!/usr/bin/env bash
# One fixed48 seed per job; exactly one GPU; five calibration runs are serial.

#SBATCH --job-name=mc48_cal_one
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=08:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/mc48_cal_%j.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/mc48_cal_%j.err

set -euo pipefail
umask 002
unset PYTHONOPTIMIZE
# Disable the reused trainer's optional external Labless submission hook.  An
# inherited token path must never trigger an out-of-contract side effect.
unset LABLESS_AUTOSUBMIT_FILE
export TORCH_HOME=/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730/control/torch_home

SEED=${1:?usage: calibration_one_seed.sbatch SEED ATTEMPT_NN}
ATTEMPT=${2:?usage: calibration_one_seed.sbatch SEED ATTEMPT_NN}
REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
PY_TARGET=/admin/home/ryan.kim/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
PYVENV_CONFIG=/admin/home/ryan.kim/nanopath/.venv/pyvenv.cfg
TRAIN_REPO="$REPO/vendor/matched_stage_train_20260730"
TRAIN="$TRAIN_REPO/train.py"
RUNNER="$REPO/tools/matched_cancer_fixed48_20260730/runner.py"
AUDITOR="$REPO/tools/matched_cancer_fixed48_20260730/auditor.py"
MANIFEST_BUILDER="$REPO/tools/matched_cancer_stage_20260730/manifest_builder.py"
COMPLETION="$REPO/tools/matched_cancer_stage_20260730/completion_receipt.py"
PRETRAINED="$TORCH_HOME/hub/checkpoints/dinov2_vits14_reg4_pretrain.pth"
PRETRAINED_SHA=f433177089a681826f849f194ece3bb48f4d63fb38d32fc837e3dc7a4e5641fb
ROOT="/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730/calibration/seed_${SEED}/attempt_${ATTEMPT}"

die() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -s "$1" && ! -L "$1" ]] || die "bad file: $1"; }
verify_pretrained() {
  require_file "$PRETRAINED"
  local actual
  read -r actual _ < <(sha256sum "$PRETRAINED")
  [[ "$actual" == "$PRETRAINED_SHA" ]] \
    || die "frozen DINOv2 pretrained checkpoint identity drift"
}

[[ "$SEED" =~ ^32(00[1-9]|0[1-3][0-9]|04[0-8])$ ]] \
  || die "seed must be 32001..32048"
[[ "$ATTEMPT" =~ ^[0-9]{2,}$ ]] || die "attempt must be two or more digits"
[[ "${SLURM_GPUS_ON_NODE:-}" == "1" ]] \
  || die "SLURM_GPUS_ON_NODE must be exactly one"
[[ "${SLURM_NTASKS:-1}" == "1" ]] || die "job must have exactly one task"
JOB_GPUS=${SLURM_JOB_GPUS:-}
VISIBLE_GPUS=${CUDA_VISIBLE_DEVICES:-}
[[ -n "$JOB_GPUS" && "$JOB_GPUS" != *,* \
  && "$JOB_GPUS" != *[[:space:]]* ]] \
  || die "SLURM_JOB_GPUS must contain exactly one GPU identifier"
[[ -n "$VISIBLE_GPUS" && "$VISIBLE_GPUS" != *,* \
  && "$VISIBLE_GPUS" != *[[:space:]]* ]] \
  || die "CUDA_VISIBLE_DEVICES must contain exactly one GPU identifier"

for path in "$TRAIN" "$RUNNER" "$AUDITOR" "$MANIFEST_BUILDER" "$COMPLETION"
do
  require_file "$path"
done
[[ -L "$PY" && "$(readlink -f "$PY")" == "$PY_TARGET" ]] \
  || die "Python venv launcher does not resolve to the frozen interpreter"
for path in "$PY_TARGET" "$PYVENV_CONFIG"
do
  require_file "$path"
done
[[ ! -w "$(dirname "$PRETRAINED")" ]] \
  || die "frozen checkpoint directory must not be writable"
verify_pretrained
[[ -x "$PY" ]] || die "missing Python executable: $PY"

cd "$REPO"
"$PY" - "$SEED" <<'PY'
import os
import sys
import torch

seed = int(sys.argv[1])
if seed not in range(32001, 32049):
    raise SystemExit("invalid seed")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"exactly one visible study GPU is required; got {torch.cuda.device_count()}"
    )
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not visible:
    raise SystemExit("CUDA_VISIBLE_DEVICES must bind the one study GPU")
PY

"$PY" -m py_compile "$RUNNER" "$AUDITOR" "$MANIFEST_BUILDER" "$COMPLETION"
"$PY" "$RUNNER" prepare --seed "$SEED" --root "$ROOT"
MANIFEST_CONFIG="$ROOT/configs/manifest_build.yaml"
REPLAY_MANIFEST="$ROOT/CALIBRATION_REPLAY_MANIFEST.json"
"$PY" "$MANIFEST_BUILDER" \
  --config "$MANIFEST_CONFIG" \
  --destination "$REPLAY_MANIFEST"
require_file "$REPLAY_MANIFEST"
export WANDB_MODE=offline

for RUN in slot1_plain slot1_fair B H P
do
  CONFIG=$("$PY" "$RUNNER" build --seed "$SEED" --root "$ROOT" --run "$RUN" \
    | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["path"])')
  READY=$("$PY" "$RUNNER" verify-ready --seed "$SEED" --root "$ROOT" --run "$RUN" \
    | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["path"])')
  [[ "$CONFIG" == "$READY" ]] || die "ready config differs from built config"
  verify_pretrained
  OUTPUT="$ROOT/$RUN"
  cd "$TRAIN_REPO"
  "$PY" "$TRAIN" "$CONFIG" "output_dir=$OUTPUT"
  cd "$REPO"
  require_file "$OUTPUT/latest.pt"
  require_file "$OUTPUT/summary.json"
  require_file "$OUTPUT/metrics.jsonl"
  [[ ! -e "$OUTPUT/COMPLETION_RECEIPT.json" ]] \
    || die "completion receipt already exists: $RUN"
  "$PY" "$COMPLETION" \
    --output-dir "$OUTPUT" \
    --effective-config-receipt "$CONFIG.receipt.json" \
    --destination "$OUTPUT/COMPLETION_RECEIPT.json"
done

"$PY" "$RUNNER" finalize --seed "$SEED" --root "$ROOT"
"$PY" "$AUDITOR" --seed "$SEED" --root "$ROOT"
require_file "$ROOT/ROOT_CALIBRATION_COMPLETION_RECEIPT.json"
require_file "$ROOT/INDEPENDENT_CALIBRATION_AUDIT_RECEIPT.json"
echo "COMPLETE: $ROOT"
