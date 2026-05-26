#!/bin/bash
#SBATCH --job-name=blip2_scicap_ft
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=160G

# BLIP-2 SciCap fine-tuning job.
# Resources: one GPU, 8 CPU cores, 160 GB RAM, 48 hour limit.
# Environment: activates the configured Conda environment.
# Launches: SciCap split preparation and src.train.finetune_blip2_scicap.
# Outputs: trainable checkpoint, history JSON, raw fine-tuned metrics, table rows, logs.
# Expensive: this is the heaviest thesis job and should not run during audits.
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

BLIP2_EPOCHS="${BLIP2_EPOCHS:-20}"
BLIP2_MIN_EPOCHS="${BLIP2_MIN_EPOCHS:-8}"
BLIP2_PATIENCE="${BLIP2_PATIENCE:-5}"
BLIP2_MIN_DELTA="${BLIP2_MIN_DELTA:-0.01}"
BLIP2_RERANK_TOP_K="${BLIP2_RERANK_TOP_K:-128}"
BLIP2_PAIR_BATCH_SIZE="${BLIP2_PAIR_BATCH_SIZE:-32}"
BLIP2_WARMUP_STEPS="${BLIP2_WARMUP_STEPS:-200}"
BLIP2_GRAD_ACCUM="${BLIP2_GRAD_ACCUM:-8}"

echo "BLIP-2 fine-tuning SciCap settings:"
echo "  epochs=${BLIP2_EPOCHS}"
echo "  min_epochs=${BLIP2_MIN_EPOCHS}"
echo "  early_stopping_patience=${BLIP2_PATIENCE}"
echo "  early_stopping_min_delta=${BLIP2_MIN_DELTA}"
echo "  rerank_top_k=${BLIP2_RERANK_TOP_K}"
echo "  pair_batch_size=${BLIP2_PAIR_BATCH_SIZE}"
echo "  warmup_steps=${BLIP2_WARMUP_STEPS}"
echo "  grad_accum_steps=${BLIP2_GRAD_ACCUM}"
echo "  effective_batch_size=$((8 * BLIP2_GRAD_ACCUM))"

python -m src.train.finetune_blip2_scicap \
  --config configs/experiment_config.yaml \
  --epochs "${BLIP2_EPOCHS}" \
  --min-epochs "${BLIP2_MIN_EPOCHS}" \
  --early-stopping-patience "${BLIP2_PATIENCE}" \
  --early-stopping-min-delta "${BLIP2_MIN_DELTA}" \
  --rerank-top-k "${BLIP2_RERANK_TOP_K}" \
  --pair-batch-size "${BLIP2_PAIR_BATCH_SIZE}" \
  --warmup-steps "${BLIP2_WARMUP_STEPS}" \
  --grad-accum-steps "${BLIP2_GRAD_ACCUM}"
