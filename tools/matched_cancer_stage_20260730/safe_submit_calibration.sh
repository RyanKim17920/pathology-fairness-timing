#!/usr/bin/env bash
# Submit the frozen one-GPU calibration only when account-wide GPU pressure is safe.

set -euo pipefail
umask 002
export PATH="/opt/slurm/bin:/usr/bin:/bin"

SLURM_USER="ryan.kim"
JOB_NAME="mcs_cal_32001"
MAX_RUNNING_GPUS_BEFORE_SUBMIT=4
POLL_SECONDS=60
REPO="/admin/home/ryan.kim/nt"
SBATCH_FILE="$REPO/tools/matched_cancer_stage_20260730/calibration_two_slot.sbatch"
CONTROL_DIR="$REPO/results/matched_cancer_stage_20260730/calibration_submission"
LOCK_FILE="$CONTROL_DIR/safe_submit.lock"
STATE_FILE="$CONTROL_DIR/state.tsv"

mkdir -p "$CONTROL_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "A calibration submission monitor is already active."
  exit 0
fi

[[ -s "$SBATCH_FILE" && ! -L "$SBATCH_FILE" ]] || {
  echo "Missing or invalid sbatch file: $SBATCH_FILE" >&2
  exit 2
}

write_state() {
  local state=$1
  local detail=$2
  printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$state" "$detail" >"$STATE_FILE"
}

gpu_value() {
  local record=$1
  local field=$2
  local value=0
  if [[ "$record" =~ ${field}=[^[:space:]]*gres/gpu=([0-9]+) ]]; then
    value=${BASH_REMATCH[1]}
  fi
  printf '%s\n' "$value"
}

while true; do
  existing="$(squeue -h -u "$SLURM_USER" -n "$JOB_NAME" -o '%i' | head -n 1)"
  if [[ -n "$existing" ]]; then
    write_state "already_present" "job_id=$existing"
    echo "Calibration already exists in Slurm as job $existing."
    exit 0
  fi

  running_gpus=0
  while IFS= read -r job_id; do
    [[ -n "$job_id" ]] || continue
    record="$(scontrol show job -o "$job_id" 2>/dev/null || true)"
    [[ -n "$record" ]] || continue
    allocated="$(gpu_value "$record" "AllocTRES")"
    running_gpus=$((running_gpus + allocated))
  done < <(squeue -h -u "$SLURM_USER" -t RUNNING -o '%i')

  pending_multigpu=()
  while IFS= read -r job_id; do
    [[ -n "$job_id" ]] || continue
    record="$(scontrol show job -o "$job_id" 2>/dev/null || true)"
    [[ -n "$record" ]] || continue
    [[ "$record" == *"Reason=JobHeldUser"* ]] && continue
    requested="$(gpu_value "$record" "ReqTRES")"
    if ((requested > 1)); then
      pending_multigpu+=("$job_id:$requested")
    fi
  done < <(squeue -h -u "$SLURM_USER" -t PENDING -o '%i')

  if ((running_gpus <= MAX_RUNNING_GPUS_BEFORE_SUBMIT)) &&
     ((${#pending_multigpu[@]} == 0)); then
    write_state "submitting" "running_gpus=$running_gpus"
    cd "$REPO"
    submitted="$(sbatch --parsable "$SBATCH_FILE")"
    [[ -n "$submitted" ]] || {
      write_state "submission_failed" "empty_sbatch_response"
      exit 3
    }
    write_state "submitted" "job_id=$submitted running_gpus_before=$running_gpus"
    echo "Submitted calibration job $submitted with $running_gpus GPUs already running."
    exit 0
  fi

  pending_detail="none"
  if ((${#pending_multigpu[@]} > 0)); then
    pending_detail="$(IFS=,; echo "${pending_multigpu[*]}")"
  fi
  write_state \
    "waiting" \
    "running_gpus=$running_gpus pending_multigpu=$pending_detail"
  sleep "$POLL_SECONDS"
done
