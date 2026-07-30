#!/usr/bin/env bash
# One-GPU diagnostic template. No loader executes until both gates verify.

#SBATCH --job-name=mcs_diag_32001
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=n
#SBATCH --time=24:00:00
#SBATCH --output=/data/ryan.kim/nanopath/runs/slurm/mcs_diag_32001_%j.out
#SBATCH --error=/data/ryan.kim/nanopath/runs/slurm/mcs_diag_32001_%j.err

set -euo pipefail
umask 002
unset PYTHONOPTIMIZE

REPO=/admin/home/ryan.kim/nt
PY=/admin/home/ryan.kim/nanopath/.venv/bin/python
DEPLOY="$REPO/tools/matched_cancer_diagnostic_20260730/deployment.py"
EXPORTER="$REPO/tools/matched_cancer_diagnostic_20260730/exporter.py"
DEPLOY_MODULE=tools.matched_cancer_diagnostic_20260730.deployment
EXPORTER_MODULE=tools.matched_cancer_diagnostic_20260730.exporter
CONTRACT="$REPO/tools/matched_cancer_diagnostic_20260730/deployment_seed32001.json"

: "${CALIBRATION_ROOT_RECEIPT:?required}"
: "${COMPLETION_B:?required}"
: "${COMPLETION_P:?required}"
: "${COMPLETION_H:?required}"
: "${AUTHORIZED_PATHS_JSON:?required}"
: "${DIAGNOSTIC_OUTPUT_ROOT:?required}"
: "${SLURM_JOB_ID:?required}"

[[ ! -e "$DIAGNOSTIC_OUTPUT_ROOT" ]] || {
  echo "output root already exists: $DIAGNOSTIC_OUTPUT_ROOT" >&2
  exit 2
}
mkdir -p "$DIAGNOSTIC_OUTPUT_ROOT"
GATE="$DIAGNOSTIC_OUTPUT_ROOT/DEPLOYMENT_GATE_RECEIPT.json"

cd "$REPO"
"$PY" -m py_compile "$DEPLOY" "$EXPORTER"
"$PY" -m unittest -v \
  tools.matched_cancer_diagnostic_20260730.test_runner \
  tools.matched_cancer_diagnostic_20260730.test_integration \
  tools.matched_cancer_diagnostic_20260730.test_exporter \
  tools.matched_cancer_diagnostic_20260730.test_deployment \
  tools.matched_cancer_diagnostic_20260730.test_vetted_loader \
  tools.matched_cancer_diagnostic_20260730.test_authorization_builder
"$PY" -m "$DEPLOY_MODULE" preflight \
  --contract "$CONTRACT" \
  --calibration-root "$CALIBRATION_ROOT_RECEIPT" \
  --completion-b "$COMPLETION_B" \
  --completion-p "$COMPLETION_P" \
  --completion-h "$COMPLETION_H" \
  --authorization-manifest "$AUTHORIZED_PATHS_JSON" \
  --write-gate "$GATE"
"$PY" -m "$DEPLOY_MODULE" run \
  --gate-receipt "$GATE" \
  --output-root "$DIAGNOSTIC_OUTPUT_ROOT/run"

SEALED="$DIAGNOSTIC_OUTPUT_ROOT/sealed"
mkdir -p "$SEALED"
exports=()
for cancer in BRCA LUAD; do
  for arm in B P H; do
    for head_seed in 42001 42002 42003 42004; do
      cell="$DIAGNOSTIC_OUTPUT_ROOT/run/diagnostics/$cancer/$arm/head_seed_$head_seed"
      output="$SEALED/seed32001_${arm}_${cancer}_${head_seed}.jsonl"
      "$PY" -m "$EXPORTER_MODULE" export \
        --nested-predictions "$cell/nested_predictions.jsonl" \
        --diagnostic-receipt "$cell/DIAGNOSTIC_RECEIPT.json" \
        --deployment-gate-receipt "$GATE" \
        --loader-root-receipt \
          "$DIAGNOSTIC_OUTPUT_ROOT/run/LOADER_ROOT_RECEIPT.json" \
        --destination "$output" \
        --fm-seed 32001 \
        --arm "$arm" \
        --cancer "$cancer" \
        --head-seed "$head_seed"
      exports+=("$output")
    done
  done
done
"$PY" -m "$EXPORTER_MODULE" collect \
  --exports "${exports[@]}" \
  --destination "$DIAGNOSTIC_OUTPUT_ROOT/seed32001_predictions.jsonl" \
  --fm-seeds 32001
