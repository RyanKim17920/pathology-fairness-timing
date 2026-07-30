#!/usr/bin/env bash
# Submit one serial fixed-five allocation only when neither study tag is live.

set -euo pipefail
umask 002

[[ $# -eq 0 ]] || {
  echo "usage: $0" >&2
  exit 2
}

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
PY_TARGET=/admin/home/ryan.kim/.local/share/uv/python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12
PYVENV_CONFIG=/admin/home/ryan.kim/nanopath/.venv/pyvenv.cfg
SBATCH=/opt/slurm/bin/sbatch
SQUEUE=/opt/slurm/bin/squeue
SLURM_USER=ryan.kim
FIXED48_COMMENT=matched_cancer_fixed48_20260730
FIXED5_COMMENT=matched_cancer_fixed5_20260730
ROOT=/data/ryan.kim/nanopath/reruns/matched_cancer_fixed48_20260730
FIXED48_MANIFEST="$ROOT/control/FIXED48_SOURCE_MANIFEST_V2.json"
FIXED5_MANIFEST="$ROOT/control/FIXED5_SOURCE_MANIFEST_V1.json"
AUTHORIZATION="$ROOT/authorization/AUTHORIZATION_MANIFEST_V3.json"
ADOPTION="$ROOT/authorization/FIXED5_ADOPTION_AUTHORIZATION_V1.json"
FEASIBILITY="$ROOT/control/FEASIBILITY_GATE_RECEIPT_V2.json"
DRIVER="$REPO/tools/matched_cancer_fixed5_20260730/serial_fixed5.sbatch"
# Sharing the legacy submission lock closes the cross-script race with the
# already-frozen fixed-48 submitter without editing that submitter.
SHARED_LOCK="$REPO/results/matched_cancer_stage_20260730/fixed48_execution/submission/safe_submit.lock"
CONTROL="$REPO/results/matched_cancer_stage_20260730/fixed5_execution/submission"
STATE="$CONTROL/state.tsv"

mkdir -p "$(dirname "$SHARED_LOCK")" "$CONTROL"
exec 9>"$SHARED_LOCK"
if ! flock -n 9; then
  echo "A matched-cancer submission process is already active." >&2
  exit 3
fi

for path in "$SBATCH" "$SQUEUE" "$FIXED48_MANIFEST" "$FIXED5_MANIFEST" \
  "$AUTHORIZATION" "$ADOPTION" "$FEASIBILITY" "$DRIVER"; do
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

SQUEUE_OUTPUT=$(
  "$SQUEUE" -h -u "$SLURM_USER" -t PENDING,RUNNING -o '%i|%k'
)
STUDY_JOBS=()
while IFS='|' read -r job_id comment extra; do
  [[ -n "$job_id" || -n "$comment" || -n "$extra" ]] || continue
  [[ -z "$extra" ]] || {
    echo "malformed scheduler output" >&2
    exit 4
  }
  if [[ "$comment" == "$FIXED48_COMMENT" || "$comment" == "$FIXED5_COMMENT" ]]; then
    [[ "$job_id" =~ ^[0-9]+([_.][0-9]+)?$ ]] || {
      echo "tagged scheduler row has an invalid job id" >&2
      exit 4
    }
    STUDY_JOBS+=("$job_id")
  fi
done <<<"$SQUEUE_OUTPUT"
if ((${#STUDY_JOBS[@]} > 0)); then
  printf '%s\trejected_live_study\tjob_ids=%s\n' \
    "$(date -u +%FT%TZ)" "${STUDY_JOBS[*]}" >"$STATE"
  echo "Refusing submission: matched-cancer study job(s) ${STUDY_JOBS[*]} are pending or running." >&2
  exit 3
fi

cd "$REPO"
"$PY" -m tools.matched_cancer_fixed5_20260730.serial_controller \
  submission-preflight \
  --production-root "$ROOT" \
  --source-manifest "$FIXED48_MANIFEST" \
  --fixed5-source-manifest "$FIXED5_MANIFEST" \
  --authorization-manifest "$AUTHORIZATION" \
  --adoption-authorization "$ADOPTION" \
  --feasibility-gate "$FEASIBILITY"

# The per-launch nonce and receipt are created while the shared lock remains
# held and immediately before the sole allocation submission.
LAUNCH_NONCE=$(
  "$PY" -m tools.matched_cancer_fixed5_20260730.launch_receipt new-nonce
)
[[ "$LAUNCH_NONCE" =~ ^[0-9a-f]{32,}$ ]] || {
  echo "nonce generator returned an invalid launch nonce" >&2
  exit 5
}
PRELAUNCH="$ROOT/control/prelaunch/FIXED5_PRELAUNCH_${LAUNCH_NONCE}.json"
"$PY" -m tools.matched_cancer_fixed5_20260730.launch_receipt \
  create-prelaunch \
  --destination "$PRELAUNCH" \
  --launch-nonce "$LAUNCH_NONCE" \
  --production-root "$ROOT" \
  --fixed5-source-manifest "$FIXED5_MANIFEST" \
  --adoption-authorization "$ADOPTION" \
  --fixed48-source-manifest "$FIXED48_MANIFEST" \
  --authorization-manifest "$AUTHORIZATION" \
  --feasibility-gate "$FEASIBILITY"

# This is intentionally the sole allocation submission in this script.
JOB_ID=$(
  "$SBATCH" --parsable \
    --export="ALL,FIXED5_LAUNCH_NONCE=$LAUNCH_NONCE,FIXED5_PRELAUNCH_RECEIPT=$PRELAUNCH,FIXED5_SLURM_COMMENT=$FIXED48_COMMENT" \
    "$DRIVER"
)
[[ "$JOB_ID" =~ ^[0-9]+([_;][A-Za-z0-9._-]+)?$ ]] || {
  echo "submission returned an invalid job id" >&2
  exit 5
}
printf '%s\tsubmitted\tjob_id=%s launch_nonce=%s prelaunch=%s\n' \
  "$(date -u +%FT%TZ)" "$JOB_ID" "$LAUNCH_NONCE" "$PRELAUNCH" >"$STATE"
echo "Submitted fixed-five serial allocation as job $JOB_ID."
