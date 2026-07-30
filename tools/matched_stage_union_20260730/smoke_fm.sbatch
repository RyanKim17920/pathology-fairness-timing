#!/usr/bin/env bash
# One-GPU, four-step smoke test for shared FairCon and gradient diagnostics.

#SBATCH --job-name=msu_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=n
#SBATCH --time=00:30:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/msu_smoke_%j.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/msu_smoke_%j.err

set -euo pipefail
umask 002

REPO=/admin/home/ryan.kim/nt
TRAIN_REPO="$REPO/vendor/matched_stage_train_20260730"
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
CONFIG="$REPO/configs_vendor/matched_stage_union_20260730/smoke_faircon.yaml"
CONTRACT="$REPO/tools/matched_stage_union_20260730/contract.py"
INSTRUMENTATION="$REPO/tools/matched_stage_union_20260730/instrumentation.py"
OBJECTIVE="$REPO/tools/matched_stage_union_20260730/objectives.py"
OUT_ROOT=/data/ryan.kim/nanopath/reruns/matched_stage_union_20260730/smoke

[[ -x "$PY" ]] || { echo "missing Python: $PY" >&2; exit 2; }
for path in "$TRAIN_REPO/train.py" "$CONFIG" "$CONTRACT" \
            "$INSTRUMENTATION" "$OBJECTIVE"; do
  [[ -s "$path" && ! -L "$path" ]] ||
    { echo "missing/empty/symlink file: $path" >&2; exit 2; }
done
"$PY" -m py_compile "$TRAIN_REPO/train.py" "$CONTRACT" \
  "$INSTRUMENTATION" "$OBJECTIVE"
cd "$REPO"
"$PY" -m unittest -v \
  tools.matched_stage_union_20260730.test_instrumentation \
  tools.matched_stage_union_20260730.test_objectives
"$PY" "$CONTRACT" preflight

STAGE="$OUT_ROOT/job_${SLURM_JOB_ID:?}"
[[ ! -e "$STAGE" ]] || { echo "smoke output exists: $STAGE" >&2; exit 2; }
mkdir -p "$STAGE"
export WANDB_MODE=offline
cd "$TRAIN_REPO"
"$PY" train.py "$CONFIG" "output_dir=$STAGE"
"$PY" - "$STAGE" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text())
if summary.get("stop_reason") != "max_train_samples":
    raise SystemExit("smoke did not reach its sample cap")
rows = [
    json.loads(line)
    for line in (root / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
dose = [row for row in rows if "dose_fair_main_grad_ratio" in row]
if not dose:
    raise SystemExit("smoke emitted no dose diagnostics")
required = (
    "main", "fair", "dose_main_grad_norm", "dose_fair_grad_norm",
    "dose_fair_main_grad_ratio", "dose_grad_cosine",
)
if any(
    key not in row or not math.isfinite(float(row[key]))
    for row in dose
    for key in required
):
    raise SystemExit("smoke dose diagnostics are incomplete/nonfinite")
if not any(float(row["dose_fair_grad_norm"]) > 0 for row in dose):
    raise SystemExit("smoke fairness gradient is identically zero")
print(json.dumps({"status": "smoke_valid", "dose_rows": len(dose)}))
PY
echo "COMPLETE: $STAGE"
