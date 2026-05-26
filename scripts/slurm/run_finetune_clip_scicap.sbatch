#!/bin/bash
#SBATCH --job-name=clip_scicap_ft
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=96G

# CLIP SciCap fine-tuning job.
# Resources: one GPU, 8 CPU cores, 96 GB RAM, 24 hour limit.
# Environment: activates the configured Conda environment.
# Launches: SciCap split preparation and src.train.finetune_clip_scicap.
# Outputs: checkpoint, history JSON, raw fine-tuned metrics, table rows, logs.
# Expensive: trains and evaluates a checkpoint; submit intentionally on a GPU node.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-bakalarka}"
CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/bin/activate}"
cd "$PROJECT_ROOT"
mkdir -p outputs/scicap/logs outputs/scicap/raw_results outputs/scicap/tables outputs/scicap/checkpoints

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Git commit: $(git rev-parse HEAD 2>/dev/null || echo 'not-a-git-repository')"
echo "GPU info:"
nvidia-smi || true
if [[ ! -f "$CONDA_INIT" ]]; then
  echo "Set CONDA_INIT to your conda activate script before submitting this job."
  exit 1
fi
source "$CONDA_INIT" "$CONDA_ENV"
python --version
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY

python -m src.prepare_scicap_split

CLIP_EPOCHS="${CLIP_EPOCHS:-30}"
CLIP_MIN_EPOCHS="${CLIP_MIN_EPOCHS:-8}"
CLIP_PATIENCE="${CLIP_PATIENCE:-6}"
CLIP_MIN_DELTA="${CLIP_MIN_DELTA:-0.01}"

echo "CLIP fine-tuning SciCap settings:"
echo "  epochs=${CLIP_EPOCHS}"
echo "  min_epochs=${CLIP_MIN_EPOCHS}"
echo "  early_stopping_patience=${CLIP_PATIENCE}"
echo "  early_stopping_min_delta=${CLIP_MIN_DELTA}"

python -m src.train.finetune_clip_scicap \
  --config configs/experiment_config.yaml \
  --epochs "${CLIP_EPOCHS}" \
  --min-epochs "${CLIP_MIN_EPOCHS}" \
  --early-stopping-patience "${CLIP_PATIENCE}" \
  --early-stopping-min-delta "${CLIP_MIN_DELTA}"
