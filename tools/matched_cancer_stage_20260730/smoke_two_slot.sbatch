#!/usr/bin/env bash
# One-GPU sequential smoke for B/P/H cancer-only timing arms.

#SBATCH --job-name=mcs_two_slot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=n
#SBATCH --time=00:30:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/mcs_two_slot_%j.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/mcs_two_slot_%j.err

set -euo pipefail
umask 002

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
TRAIN_REPO="$REPO/vendor/matched_stage_train_20260730"
TRAIN="$TRAIN_REPO/train.py"
BUILDER="$REPO/tools/matched_cancer_stage_20260730/config_builder.py"
CONTRACT="$REPO/tools/matched_cancer_stage_20260730/contract.py"
MANIFEST_BUILDER="$REPO/tools/matched_cancer_stage_20260730/manifest_builder.py"
COMPLETION="$REPO/tools/matched_cancer_stage_20260730/completion_receipt.py"
SPEC="$REPO/configs_vendor/matched_cancer_stage_20260730/smoke_contract.yaml"
ROOT="/data/ryan.kim/nanopath/reruns/matched_cancer_stage_20260730/smoke/job_${SLURM_JOB_ID:?}"
CONTRACT_RECEIPT="$ROOT/CONTRACT_RECEIPT.json"
REPLAY_MANIFEST="$ROOT/REPLAY_MANIFEST.json"

die() { echo "ERROR: $*" >&2; exit 2; }
require_file() { [[ -s "$1" && ! -L "$1" ]] || die "bad file: $1"; }

for path in \
  "$TRAIN" "$BUILDER" "$CONTRACT" "$MANIFEST_BUILDER" "$COMPLETION" "$SPEC"
do
  require_file "$path"
done
[[ -x "$PY" ]] || die "missing Python: $PY"
[[ ! -e "$ROOT" ]] || die "smoke root already exists: $ROOT"

cd "$REPO"
"$PY" -m py_compile \
  "$TRAIN" "$BUILDER" "$CONTRACT" "$MANIFEST_BUILDER" "$COMPLETION"
"$PY" -m unittest -v \
  tools.matched_cancer_stage_20260730.test_objectives \
  tools.matched_cancer_stage_20260730.test_replay \
  tools.matched_cancer_stage_20260730.test_receipts
mkdir -p "$ROOT/configs"
"$PY" -m tools.matched_cancer_stage_20260730.contract \
  --write-receipt "$CONTRACT_RECEIPT"
export WANDB_MODE=offline

MANIFEST_CONFIG="$ROOT/configs/manifest_build.yaml"
"$PY" "$BUILDER" \
  --contract "$SPEC" \
  --contract-receipt "$CONTRACT_RECEIPT" \
  --replay-manifest "$REPLAY_MANIFEST" \
  --manifest-build-only \
  --mode joint \
  --fair-weight 0.0 \
  --name matched-cancer-smoke-manifest-build \
  --output-dir "$ROOT/manifest_build_not_run" \
  --destination "$MANIFEST_CONFIG"
"$PY" "$MANIFEST_BUILDER" \
  --config "$MANIFEST_CONFIG" \
  --destination "$REPLAY_MANIFEST"

build_and_run() {
  local name=$1
  local mode=$2
  local weight=$3
  local checkpoint=${4:-}
  local parent_receipt=${5:-}
  local output="$ROOT/$name"
  local config="$ROOT/configs/$name.yaml"
  local config_receipt="$config.receipt.json"
  local args=(
    --contract "$SPEC"
    --contract-receipt "$CONTRACT_RECEIPT"
    --replay-manifest "$REPLAY_MANIFEST"
    --mode "$mode"
    --fair-weight "$weight"
    --name "matched-cancer-smoke-$name"
    --output-dir "$output"
    --destination "$config"
  )
  if [[ -n "$checkpoint" ]]; then
    args+=(--encoder-checkpoint "$checkpoint")
  fi
  if [[ -n "$parent_receipt" ]]; then
    args+=(--parent-completion-receipt "$parent_receipt")
  fi
  "$PY" "$BUILDER" "${args[@]}"
  if grep -Eiq 'tp53|molecular_labels|condition_on_label' "$config"; then
    die "downstream-label token in representation config: $config"
  fi
  cd "$TRAIN_REPO"
  "$PY" "$TRAIN" "$config" "output_dir=$output"
  cd "$REPO"
  "$PY" "$COMPLETION" \
    --output-dir "$output" \
    --effective-config-receipt "$config_receipt" \
    --destination "$output/COMPLETION_RECEIPT.json"
}

# Slot 1: one plain encoder reused by B/H, and one fair encoder for P.
build_and_run slot1_plain joint 0.0
build_and_run slot1_fair joint 0.1

PLAIN_CKPT="$ROOT/slot1_plain/latest.pt"
FAIR_CKPT="$ROOT/slot1_fair/latest.pt"
PLAIN_RECEIPT="$ROOT/slot1_plain/COMPLETION_RECEIPT.json"
FAIR_RECEIPT="$ROOT/slot1_fair/COMPLETION_RECEIPT.json"
require_file "$PLAIN_CKPT"
require_file "$FAIR_CKPT"
require_file "$PLAIN_RECEIPT"
require_file "$FAIR_RECEIPT"

# Slot 2: fresh identical final adapters. Encoder is frozen in every run.
build_and_run B adapter_only 0.0 "$PLAIN_CKPT" "$PLAIN_RECEIPT"
build_and_run H adapter_only 0.1 "$PLAIN_CKPT" "$PLAIN_RECEIPT"
build_and_run P adapter_only 0.0 "$FAIR_CKPT" "$FAIR_RECEIPT"

"$PY" - "$ROOT" <<'PY'
import json
import math
import sys
from pathlib import Path

from tools.matched_cancer_stage_20260730.receipts import (
    atomic_write_receipt,
    build_receipt,
    file_identity,
    verify_receipt,
)

root = Path(sys.argv[1])
names = ("slot1_plain", "slot1_fair", "B", "H", "P")
summaries = {
    name: json.loads((root / name / "summary.json").read_text())
    for name in names
}
stage = {name: summaries[name]["matched_stage"] for name in names}

assert stage["slot1_plain"]["mode"] == "joint"
assert stage["slot1_fair"]["mode"] == "joint"
assert all(stage[name]["mode"] == "adapter_only" for name in ("B", "H", "P"))
assert stage["slot1_plain"]["fair_weight"] == 0.0
assert stage["slot1_fair"]["fair_weight"] == 0.1
assert stage["B"]["fair_weight"] == 0.0
assert stage["H"]["fair_weight"] == 0.1
assert stage["P"]["fair_weight"] == 0.0

for summary in summaries.values():
    assert summary["steps_completed"] == 4
    assert summary["tile_presentations"] == 32
    assert summary["stop_reason"] == "max_train_samples"

for key in (
    "adapter_pre_sha256",
    "replay_sampler_sha256",
    "replay_patient_sha256",
    "replay_tile_sha256",
    "replay_augmentation_seed_sha256",
    "replay_manifest_file_sha256",
    "sample_batch_trace_sha256",
    "patient_batch_trace_sha256",
    "augmentation_seed_batch_trace_sha256",
    "augmentation_seed_manifest_trace_sha256",
):
    assert len({stage[name][key] for name in names}) == 1, key
for name in names:
    assert (
        stage[name]["augmentation_seed_manifest_trace_sha256"]
        == stage[name]["replay_augmentation_seed_sha256"]
    )

assert (
    stage["slot1_plain"]["encoder_pre_sha256"]
    == stage["slot1_fair"]["encoder_pre_sha256"]
)
assert stage["B"]["encoder_pre_sha256"] == stage["H"]["encoder_pre_sha256"]
assert stage["B"]["encoder_post_sha256"] == stage["H"]["encoder_post_sha256"]
assert all(stage[name]["encoder_unchanged"] for name in ("B", "H", "P"))
assert (
    stage["B"]["encoder_pre_sha256"]
    == stage["H"]["encoder_pre_sha256"]
    == stage["slot1_plain"]["encoder_post_sha256"]
)
assert (
    stage["P"]["encoder_pre_sha256"]
    == stage["slot1_fair"]["encoder_post_sha256"]
)

for name in ("slot1_plain", "slot1_fair"):
    reach = stage[name]["encoder_reachability"]
    update = stage[name]["encoder_first_update"]
    assert reach["encoder_stage_grad_finite"]
    assert reach["encoder_cancer_grad_norm"] > 0
    assert reach["encoder_fair_raw_grad_norm"] > 0
    if stage[name]["fair_weight"] > 0:
        assert reach["encoder_fair_weighted_grad_norm"] > 0
    else:
        assert reach["encoder_fair_weighted_grad_norm"] == 0
    assert update["encoder_first_positive_lr_update_norm"] > 0
    assert update["encoder_first_positive_lr_changed_tensors"] > 0

completion_receipts = {}
for name in names:
    completion_receipts[name] = verify_receipt(
        root / name / "COMPLETION_RECEIPT.json",
        expected_schema="matched-cancer-stage-completion/v1",
        expected_study_id="matched_cancer_stage_20260730",
        expected_scenario="brca_luad_black_white",
    )
    checkpoint = __import__("torch").load(
        root / name / "latest.pt", map_location="cpu", weights_only=False
    )
    assert "model" in checkpoint and "stage_adapter" in checkpoint
    rows = [
        json.loads(line)
        for line in (root / name / "metrics.jsonl").read_text().splitlines()
        if line.strip() and '"total"' in line
    ]
    assert len(rows) == 4
    for row in rows:
        for key in (
            "cancer",
            "race_fair",
            "total",
            "h_dose_main_grad_norm",
            "h_dose_fair_grad_norm",
        ):
            assert math.isfinite(float(row[key])), (name, key)
        assert float(row["cancer"]) > 0
        assert float(row["race_fair"]) > 0
        assert float(row["h_dose_main_grad_norm"]) > 0
        if stage[name]["fair_weight"] > 0:
            assert float(row["h_dose_fair_grad_norm"]) > 0
        else:
            assert float(row["h_dose_fair_grad_norm"]) == 0

plain_identity = file_identity(root / "slot1_plain" / "latest.pt")
fair_identity = file_identity(root / "slot1_fair" / "latest.pt")
assert stage["B"]["encoder_checkpoint_identity"] == plain_identity
assert stage["H"]["encoder_checkpoint_identity"] == plain_identity
assert stage["P"]["encoder_checkpoint_identity"] == fair_identity
assert completion_receipts["slot1_plain"]["identities"][
    "latest_checkpoint"
] == plain_identity
assert completion_receipts["slot1_fair"]["identities"][
    "latest_checkpoint"
] == fair_identity

root_receipt = build_receipt(
    schema="matched-cancer-stage-root-completion/v1",
    study_id="matched_cancer_stage_20260730",
    scenario="brca_luad_black_white",
    identities={
        "contract_receipt": file_identity(root / "CONTRACT_RECEIPT.json"),
        "replay_manifest": file_identity(root / "REPLAY_MANIFEST.json"),
        "runs": {
            name: file_identity(root / name / "COMPLETION_RECEIPT.json")
            for name in names
        },
    },
    fields={
        "status": "matched_cancer_two_slot_smoke_valid",
        "arms": list(names),
        "replay_manifest_payload_sha256": stage["B"][
            "replay_sampler_sha256"
        ],
    },
)
root_receipt_path = atomic_write_receipt(
    root / "ROOT_COMPLETION_RECEIPT.json", root_receipt
)
verify_receipt(
    root_receipt_path,
    expected_schema="matched-cancer-stage-root-completion/v1",
    expected_study_id="matched_cancer_stage_20260730",
    expected_scenario="brca_luad_black_white",
)

print(json.dumps({
    "status": "matched_cancer_two_slot_smoke_valid",
    "job_root": str(root),
    "arms": list(names),
    "replay_sha256": stage["B"]["replay_sampler_sha256"],
    "root_completion_receipt": str(root_receipt_path),
}, sort_keys=True))
PY

echo "COMPLETE: $ROOT"
