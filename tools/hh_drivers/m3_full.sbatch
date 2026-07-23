#!/usr/bin/env bash
#SBATCH --job-name=m3full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=n
#SBATCH --time=04:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/m3full_%A.out

set -euo pipefail

REPO_DIR="/admin/home/ryan.kim/nanopath-brca-src"

if [[ $# -ne 2 ]]; then
  echo "usage: sbatch $0 <config.yaml> <output_dir>" >&2
  exit 2
fi

CONFIG_FILE="$1"
OUTPUT_DIR="$2"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "config not found: ${CONFIG_FILE}" >&2
  exit 2
fi
CONFIG_FILE="$(realpath -- "${CONFIG_FILE}")"

cd "${REPO_DIR}"

# Match m3_smoke.sbatch: prefer this checkout's venv, then the main checkout's.
VENV_DIR="${REPO_DIR}/.venv"
if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
  VENV_DIR="${HOME}/nanopath/.venv"
fi
source "${VENV_DIR}/bin/activate"

# train.py accepts only output_dir as a CLI override. Derive a temporary recipe
# that disables probes and resume without changing any full-run training values.
FULL_CONFIG="$(mktemp "${TMPDIR:-/tmp}/m3_full_${SLURM_JOB_ID:-local}_XXXXXX.yaml")"
cleanup() {
  rm -f -- "${FULL_CONFIG}"
}
trap cleanup EXIT

python - "${CONFIG_FILE}" "${FULL_CONFIG}" <<'PY'
import os
import sys
from pathlib import Path

import yaml

source_path = Path(sys.argv[1])
full_path = Path(sys.argv[2])
cfg = yaml.safe_load(os.path.expandvars(source_path.read_text()))

cfg["probe"]["enabled"] = False
cfg["train"]["resume"] = None

full_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(
    "FULL_EFFECTIVE_OVERRIDES=probe.enabled=false train.resume=null",
    flush=True,
)
PY

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "FULL_CONFIG=${FULL_CONFIG}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

python train.py "${FULL_CONFIG}" "output_dir=${OUTPUT_DIR}"

echo "M3 FULL DONE ${OUTPUT_DIR}"
