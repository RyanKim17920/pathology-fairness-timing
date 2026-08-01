#!/usr/bin/env bash
# One allocation and one visible GPU for the zero-new-seed representation audit.

#SBATCH --job-name=main_1gpu
#SBATCH --comment=matched_cancer_rep_audit_20260801
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=n
#SBATCH --time=3-00:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/mcs_rep_audit_%j.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/mcs_rep_audit_%j.err

set -euo pipefail
umask 002
unset PYTHONOPTIMIZE

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
PY_TARGET=/admin/home/ryan.kim/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
PYVENV_CONFIG=/admin/home/ryan.kim/nanopath/.venv/pyvenv.cfg
PIPELINE="$REPO/tools/matched_cancer_representation_audit_20260801/pipeline.py"

: "${SLURM_JOB_ID:?must run inside Slurm}"
: "${REP_AUDIT_OUTPUT_ROOT:?missing output root}"
: "${REP_AUDIT_PREFLIGHT_RECEIPT:?missing preflight receipt}"
: "${REP_AUDIT_LAUNCH_NONCE:?missing launch nonce}"
: "${REP_AUDIT_SLURM_COMMENT:?missing exported Slurm comment}"
[[ "${SLURM_JOB_NAME:-}" == main_1gpu ]] || {
  echo "Slurm job name must be main_1gpu" >&2
  exit 2
}
[[ "$REP_AUDIT_SLURM_COMMENT" == matched_cancer_rep_audit_20260801 ]] || {
  echo "Slurm comment differs from the frozen audit tag" >&2
  exit 2
}
[[ "$REP_AUDIT_LAUNCH_NONCE" =~ ^[0-9a-f]{32}$ ]] || {
  echo "launch nonce must contain exactly 128 bits" >&2
  exit 2
}
[[ "$REP_AUDIT_PREFLIGHT_RECEIPT" == \
  "$REP_AUDIT_OUTPUT_ROOT/PREFLIGHT_RECEIPT.json" ]] || {
  echo "preflight receipt path differs from the frozen attempt root" >&2
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
for path in "$PIPELINE" "$REP_AUDIT_PREFLIGHT_RECEIPT"; do
  [[ -s "$path" && ! -L "$path" ]] || {
    echo "missing, empty, or symlinked audit input: $path" >&2
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
  echo "missing executable Python interpreter" >&2
  exit 2
}

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
cd "$REPO"
"$PY" -m tools.matched_cancer_representation_audit_20260801.pipeline run \
  --output-root "$REP_AUDIT_OUTPUT_ROOT"
