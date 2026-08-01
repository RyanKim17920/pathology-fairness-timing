#!/usr/bin/env bash
# Submit exactly one serial representation-audit allocation after a queue guard.

set -euo pipefail
umask 002

[[ $# -eq 0 ]] || {
  echo "usage: $0" >&2
  exit 2
}

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
SBATCH=/opt/slurm/bin/sbatch
SQUEUE=/opt/slurm/bin/squeue
SLURM_USER=ryan.kim
AUDIT_COMMENT=matched_cancer_rep_audit_20260801
FIXED48_COMMENT=matched_cancer_fixed48_20260730
FIXED5_COMMENT=matched_cancer_fixed5_20260730
OUTPUT_ROOT=/data/ryan.kim/nanopath/reruns/matched_cancer_representation_audit_20260801/attempt_01
DRIVER="$REPO/tools/matched_cancer_representation_audit_20260801/serial_audit.sbatch"
CONTROL="$REPO/results/matched_cancer_stage_20260730/fixed5_execution/representation_audit_submission"
STATE="$CONTROL/state.tsv"
SHARED_LOCK="$REPO/results/matched_cancer_stage_20260730/fixed48_execution/submission/safe_submit.lock"

mkdir -p "$CONTROL" "$(dirname "$SHARED_LOCK")"
exec 9>"$SHARED_LOCK"
if ! flock -n 9; then
  echo "A matched-cancer submission process is already active." >&2
  exit 3
fi

for path in "$PY" "$SBATCH" "$SQUEUE" "$DRIVER"; do
  [[ -e "$path" && ! -L "$path" ]] || {
    if [[ "$path" == "$PY" && -L "$path" && -x "$path" ]]; then
      continue
    fi
    echo "missing or invalid submission input: $path" >&2
    exit 2
  }
done
[[ ! -e "$OUTPUT_ROOT" ]] || {
  echo "frozen attempt output already exists: $OUTPUT_ROOT" >&2
  exit 3
}
if grep -Eiq '^[[:space:]]*#SBATCH[[:space:]]+--(array|dependency)(=|[[:space:]])' "$DRIVER"; then
  echo "arrays and dependency chains are prohibited" >&2
  exit 2
fi

SQUEUE_OUTPUT=$("$SQUEUE" -h -u "$SLURM_USER" -t PENDING,RUNNING -o '%i|%j|%k')
STUDY_JOBS=()
while IFS='|' read -r job_id job_name comment extra; do
  [[ -n "$job_id" || -n "$job_name" || -n "$comment" || -n "$extra" ]] || continue
  [[ -z "$extra" ]] || {
    echo "malformed scheduler output" >&2
    exit 4
  }
  if [[ "$job_name" == main_1gpu || "$comment" == "$AUDIT_COMMENT" || \
        "$comment" == "$FIXED48_COMMENT" || "$comment" == "$FIXED5_COMMENT" ]]; then
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
  echo "Refusing submission: study job(s) ${STUDY_JOBS[*]} are pending or running." >&2
  exit 3
fi

LAUNCH_NONCE=$("$PY" -c 'import secrets; print(secrets.token_hex(16))')
[[ "$LAUNCH_NONCE" =~ ^[0-9a-f]{32}$ ]] || {
  echo "nonce generator returned an invalid value" >&2
  exit 5
}
PRELAUNCH="$CONTROL/PRELAUNCH_${LAUNCH_NONCE}.json"
cd "$REPO"
"$PY" -m tools.matched_cancer_representation_audit_20260801.pipeline preflight \
  --output-root "$OUTPUT_ROOT" \
  --receipt "$PRELAUNCH" \
  --launch-nonce "$LAUNCH_NONCE"

# This is intentionally the sole allocation submission in this script.
JOB_ID=$(
  "$SBATCH" --parsable \
    --export="ALL,REP_AUDIT_OUTPUT_ROOT=$OUTPUT_ROOT,REP_AUDIT_PREFLIGHT_RECEIPT=$PRELAUNCH,REP_AUDIT_LAUNCH_NONCE=$LAUNCH_NONCE,REP_AUDIT_SLURM_COMMENT=$AUDIT_COMMENT" \
    "$DRIVER"
)
[[ "$JOB_ID" =~ ^[0-9]+([_;][A-Za-z0-9._-]+)?$ ]] || {
  echo "submission returned an invalid job id" >&2
  exit 5
}
printf '%s\tsubmitted\tjob_id=%s launch_nonce=%s prelaunch=%s output=%s\n' \
  "$(date -u +%FT%TZ)" "$JOB_ID" "$LAUNCH_NONCE" "$PRELAUNCH" "$OUTPUT_ROOT" >"$STATE"
echo "Submitted zero-new-seed representation audit as job $JOB_ID."
