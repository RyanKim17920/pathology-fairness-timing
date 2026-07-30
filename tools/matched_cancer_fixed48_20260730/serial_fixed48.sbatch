#!/usr/bin/env bash
# One allocation, one visible GPU, strictly serial fixed-48 execution.

#SBATCH --job-name=mcs_fixed48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=21-00:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/mcs_fixed48_%j.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/mcs_fixed48_%j.err

set -euo pipefail
umask 002
unset PYTHONOPTIMIZE

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
PY_TARGET=/admin/home/ryan.kim/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
PYVENV_CONFIG=/admin/home/ryan.kim/nanopath/.venv/pyvenv.cfg
ROOT=/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730
MANIFEST="$ROOT/control/FIXED48_SOURCE_MANIFEST.json"
AUTHORIZATION="$ROOT/authorization/AUTHORIZATION_MANIFEST_V2.json"
FEASIBILITY="$ROOT/control/FEASIBILITY_GATE_RECEIPT.json"
SEED_WORKER="$REPO/tools/matched_cancer_fixed48_20260730/seed_worker.py"

: "${FIXED48_RUN_MODE:?FIXED48_RUN_MODE must be canary or remainder}"
: "${SLURM_JOB_ID:?must run inside Slurm}"
[[ "$FIXED48_RUN_MODE" == canary || "$FIXED48_RUN_MODE" == remainder ]] || {
  echo "FIXED48_RUN_MODE must be canary or remainder" >&2
  exit 2
}
[[ -z "${SLURM_ARRAY_JOB_ID:-}" && -z "${SLURM_ARRAY_TASK_ID:-}" ]] || {
  echo "job arrays are prohibited" >&2
  exit 2
}
for path in "$MANIFEST" "$AUTHORIZATION" "$FEASIBILITY" "$SEED_WORKER"; do
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
"$PY" -m tools.matched_cancer_fixed48_20260730.serial_controller run \
  --mode "$FIXED48_RUN_MODE" \
  --production-root "$ROOT" \
  --source-manifest "$MANIFEST" \
  --authorization-manifest "$AUTHORIZATION" \
  --feasibility-gate "$FEASIBILITY" \
  --seed-worker "$SEED_WORKER"
