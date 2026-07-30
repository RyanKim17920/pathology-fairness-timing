#!/usr/bin/env bash
# Submit at most one fixed-48 study allocation, with no arrays or dependencies.

set -euo pipefail
umask 002

MODE=${1:-}
[[ "$MODE" == canary || "$MODE" == remainder ]] || {
  echo "usage: $0 {canary|remainder}" >&2
  exit 2
}

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
PY_TARGET=/admin/home/ryan.kim/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
PYVENV_CONFIG=/admin/home/ryan.kim/nanopath/.venv/pyvenv.cfg
SBATCH=/opt/slurm/bin/sbatch
SQUEUE=/opt/slurm/bin/squeue
SLURM_USER=ryan.kim
JOB_NAME=mcs_fixed48
ROOT=/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730
MANIFEST="$ROOT/control/FIXED48_SOURCE_MANIFEST.json"
AUTHORIZATION="$ROOT/authorization/AUTHORIZATION_MANIFEST_V2.json"
FEASIBILITY="$ROOT/control/FEASIBILITY_GATE_RECEIPT.json"
DRIVER="$REPO/tools/matched_cancer_fixed48_20260730/serial_fixed48.sbatch"
CONTROL="$REPO/results/matched_cancer_stage_20260730/fixed48_execution/submission"
LOCK="$CONTROL/safe_submit.lock"
STATE="$CONTROL/state.tsv"

mkdir -p "$CONTROL"
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "A fixed-48 submission process is already active." >&2
  exit 3
fi

for path in "$SBATCH" "$SQUEUE" "$MANIFEST" "$AUTHORIZATION" \
  "$FEASIBILITY" "$DRIVER"; do
  [[ -e "$path" && ! -L "$path" ]] || {
    echo "missing or symlinked submission input: $path" >&2
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
if grep -Eiq '^[[:space:]]*#SBATCH[[:space:]]+--(array|dependency)(=|[[:space:]])' \
  "$DRIVER"; then
  echo "arrays and dependency chains are prohibited" >&2
  exit 2
fi

mapfile -t STUDY_JOBS < <(
  "$SQUEUE" -h -u "$SLURM_USER" -n "$JOB_NAME" -t PENDING,RUNNING -o '%i'
)
if ((${#STUDY_JOBS[@]} > 1)); then
  echo "invariant failure: multiple queued/running fixed-48 jobs" >&2
  exit 4
fi
if ((${#STUDY_JOBS[@]} == 1)); then
  printf '%s\talready_present\tjob_id=%s\n' \
    "$(date -u +%FT%TZ)" "${STUDY_JOBS[0]}" >"$STATE"
  echo "Fixed-48 job ${STUDY_JOBS[0]} is already queued or running."
  exit 0
fi

cd "$REPO"
"$PY" -m tools.matched_cancer_fixed48_20260730.serial_controller \
  submission-preflight \
  --mode "$MODE" \
  --production-root "$ROOT" \
  --source-manifest "$MANIFEST" \
  --authorization-manifest "$AUTHORIZATION" \
  --feasibility-gate "$FEASIBILITY"

# This is intentionally the sole submission call in this script.
JOB_ID=$(
  "$SBATCH" --parsable --export="ALL,FIXED48_RUN_MODE=$MODE" "$DRIVER"
)
[[ -n "$JOB_ID" && "$JOB_ID" != *$'\n'* ]] || {
  echo "submission returned an invalid job id" >&2
  exit 5
}
printf '%s\tsubmitted\tmode=%s job_id=%s\n' \
  "$(date -u +%FT%TZ)" "$MODE" "$JOB_ID" >"$STATE"
echo "Submitted fixed-48 $MODE allocation as job $JOB_ID."
