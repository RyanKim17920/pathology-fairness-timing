#!/usr/bin/env bash
# One allocation, one visible GPU, seeds 32002..32005 strictly serial.

#SBATCH --job-name=main_1gpu
#SBATCH --comment=matched_cancer_fixed48_20260730
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=21-00:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/mcs_fixed5_%j.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/mcs_fixed5_%j.err

set -euo pipefail
umask 002
unset PYTHONOPTIMIZE

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
PY_TARGET=/admin/home/ryan.kim/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
PYVENV_CONFIG=/admin/home/ryan.kim/nanopath/.venv/pyvenv.cfg
ROOT=/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730
FIXED48_MANIFEST="$ROOT/control/FIXED48_SOURCE_MANIFEST_V2.json"
FIXED5_MANIFEST="$ROOT/control/FIXED5_SOURCE_MANIFEST_V1.json"
AUTHORIZATION="$ROOT/authorization/AUTHORIZATION_MANIFEST_V3.json"
ADOPTION="$ROOT/authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json"
FEASIBILITY="$ROOT/control/FEASIBILITY_GATE_RECEIPT_V2.json"
SEED_WORKER="$REPO/tools/matched_cancer_fixed48_20260730/seed_worker.py"

: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${FIXED5_LAUNCH_NONCE:?missing launch nonce}"
: "${FIXED5_PRELAUNCH_RECEIPT:?missing prelaunch receipt}"
: "${FIXED5_SLURM_COMMENT:?missing exported Slurm comment}"
[[ "${SLURM_JOB_NAME:-}" == main_1gpu ]] || {
  echo "Slurm job name must be main_1gpu" >&2
  exit 2
}
[[ "$FIXED5_SLURM_COMMENT" == matched_cancer_fixed48_20260730 ]] || {
  echo "Slurm comment must use the legacy-compatible study tag" >&2
  exit 2
}
[[ "$FIXED5_LAUNCH_NONCE" =~ ^[0-9a-f]{32,}$ ]] || {
  echo "launch nonce must contain at least 128 bits" >&2
  exit 2
}
[[ "$FIXED5_PRELAUNCH_RECEIPT" == \
  "$ROOT/control/prelaunch/FIXED5_PRELAUNCH_${FIXED5_LAUNCH_NONCE}.json" ]] || {
  echo "prelaunch receipt path differs" >&2
  exit 2
}
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "job arrays are prohibited" >&2
  exit 2
}
[[ "${SLURM_NTASKS:-1}" == 1 ]] || {
  echo "allocation must contain exactly one task" >&2
  exit 2
}
for path in "$FIXED48_MANIFEST" "$FIXED5_MANIFEST" "$AUTHORIZATION" \
  "$ADOPTION" "$FEASIBILITY" "$SEED_WORKER" "$FIXED5_PRELAUNCH_RECEIPT"; do
  [[ -s "$path" && ! -L "$path" ]] || {
    echo "missing/empty/symlink production input: $path" >&2
    exit 2
  }
done
[[ -L "$PY" && "$(readlink -f "$PY")" == "$PY_TARGET" ]] || {
  echo "Python venv launcher does not resolve to the frozen interpreter" >&2
  exit 2
}
for path in "$PY_TARGET" "$PYVENV_CONFIG"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "missing or symlinked frozen Python runtime input: $path" >&2
    exit 2
  }
done
[[ -x "$PY" && -x "$PY_TARGET" ]] || {
  echo "missing executable Python interpreter: $PY" >&2
  exit 2
}

cd "$REPO"
"$PY" -m tools.matched_cancer_fixed5_20260730.serial_controller run \
  --production-root "$ROOT" \
  --source-manifest "$FIXED48_MANIFEST" \
  --fixed5-source-manifest "$FIXED5_MANIFEST" \
  --authorization-manifest "$AUTHORIZATION" \
  --adoption-authorization "$ADOPTION" \
  --feasibility-gate "$FEASIBILITY" \
  --seed-worker "$SEED_WORKER"
