#!/usr/bin/env bash
#SBATCH --job-name=m3smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=00:20:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/m3smoke_%A.out

set -euo pipefail

REPO_DIR="/admin/home/ryan.kim/nanopath-brca-src"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: sbatch $0 <config.yaml> [output_dir]" >&2
  exit 2
fi

CONFIG_FILE="$1"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "config not found: ${CONFIG_FILE}" >&2
  exit 2
fi
CONFIG_FILE="$(realpath -- "${CONFIG_FILE}")"

CONFIG_NAME="$(basename -- "${CONFIG_FILE}")"
CONFIG_NAME="${CONFIG_NAME%.yaml}"
CONFIG_NAME="${CONFIG_NAME%.yml}"
OUTPUT_DIR="${2:-/data/ryan.kim/nanopath/m3smoke/${CONFIG_NAME}}"

cd "${REPO_DIR}"

# Match submit/train_1gpu.sbatch: prefer this checkout's venv, then the main checkout's.
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

# train.py accepts only output_dir as a CLI override. Materialize the other
# smoke settings in a temporary derived recipe without changing the source YAML.
SMOKE_CONFIG="$(mktemp "${TMPDIR:-/tmp}/m3_smoke_${SLURM_JOB_ID:-local}_XXXXXX.yaml")"
cleanup() {
  rm -f -- "${SMOKE_CONFIG}"
}
trap cleanup EXIT

python - "${CONFIG_FILE}" "${SMOKE_CONFIG}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

source_path = Path(sys.argv[1])
smoke_path = Path(sys.argv[2])
cfg = yaml.safe_load(os.path.expandvars(source_path.read_text()))

batch_size = int(cfg["train"]["batch_size"])
cfg["train"]["max_train_samples"] = batch_size * 30
cfg["train"]["max_train_flops"] = 777_680_710_103_040
cfg["train"]["log_every"] = 10
cfg["train"]["save_every"] = None
cfg["train"]["eval_every"] = 1_000_000
cfg["train"]["val_batches"] = 1
cfg["train"]["resume"] = None
cfg["probe"]["enabled"] = False

smoke_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(
    "SMOKE_EFFECTIVE_OVERRIDES="
    f"train.max_train_samples={batch_size * 30} "
    "train.max_train_flops=777680710103040 "
    "train.log_every=10 train.save_every=null "
    "train.eval_every=1000000 train.val_batches=1 "
    "train.resume=null probe.enabled=false",
    flush=True,
)
PY

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "SMOKE_CONFIG=${SMOKE_CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

# output_dir is the sole key=value override supported by this train.py.
python train.py "${SMOKE_CONFIG}" "output_dir=${OUTPUT_DIR}"

# Make success conditional on the requested smoke contract, not just process exit.
python - "${OUTPUT_DIR}" <<'PY'
import json
import math
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
summary = json.loads((output_dir / "summary.json").read_text())
steps = int(summary["steps_completed"])
if not 20 <= steps <= 50:
    raise RuntimeError(f"smoke completed {steps} steps; expected 20-50")

train_rows = []
for line in (output_dir / "metrics.jsonl").read_text().splitlines():
    row = json.loads(line)
    if "total" in row:
        train_rows.append(row)

if not train_rows:
    raise RuntimeError("smoke emitted no logged training losses")
losses = [float(row["total"]) for row in train_rows]
if not all(math.isfinite(loss) for loss in losses):
    raise RuntimeError(f"smoke emitted non-finite training loss: {losses}")
if not any(abs(loss) > 1e-8 for loss in losses):
    raise RuntimeError(f"smoke emitted only degenerate zero training loss: {losses}")

print(
    f"SMOKE LOSS OK steps={steps} "
    f"first_total={losses[0]:.6g} last_total={losses[-1]:.6g}",
    flush=True,
)
PY

echo "SMOKE OK config=${CONFIG_FILE} output_dir=${OUTPUT_DIR}"
