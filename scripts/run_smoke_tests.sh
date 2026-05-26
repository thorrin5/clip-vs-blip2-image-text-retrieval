#!/bin/bash
# Run structural checks without starting full training or evaluation jobs.
# Thesis defense use:
#   - validates config shape, lightweight dataset loading, metric logic, and
#     expected launcher files;
#   - does not produce thesis metrics;
#   - CHECK_MODEL_IMPORTS=1 checks dependencies, while FULL_MODEL_SMOKE=1 loads
#     large models and should only be used inside the configured ML environment.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONDA_ENV="${CONDA_ENV:-bakalarka}"
CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/bin/activate}"

cd "$PROJECT_ROOT"
mkdir -p outputs/logs

if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
  if [[ ! -f "$CONDA_INIT" ]]; then
    echo "Set CONDA_INIT to your conda activate script before running smoke tests."
    exit 1
  fi
  source "$CONDA_INIT" "$CONDA_ENV"
fi

echo "========================================"
echo "Canonical scaffold smoke tests"
echo "========================================"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(python --version)"

SMOKE_ARGS=(--config configs/experiment_config.yaml)
if [[ "${CHECK_MODEL_IMPORTS:-0}" == "1" || "${FULL_MODEL_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--check-model-imports)
fi
if [[ "${FULL_MODEL_SMOKE:-0}" == "1" ]]; then
  SMOKE_ARGS+=(--full-model-smoke)
fi

python -m src.eval.smoke_test "${SMOKE_ARGS[@]}"

echo "Smoke tests completed."
