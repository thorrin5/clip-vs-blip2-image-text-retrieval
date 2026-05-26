#!/bin/bash
#SBATCH --job-name=blip2_scicap_zs
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=96G

# Full BLIP-2 zero-shot evaluation on SciCap with configurable ITM reranking.
# Resources: one GPU, 8 CPU cores, 96 GB RAM, 4 hour limit.
# Environment: activates the configured Conda environment.
# Launches: SciCap split preparation, then src.eval.evaluate_blip2 on SciCap test.
# Outputs: SciCap raw JSON, table rows, and SLURM logs.
# Expensive: ITM reranking should run on a compute node; environment variables
# document alternative rerank settings.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-bakalarka}"
CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/bin/activate}"
cd "$PROJECT_ROOT"
mkdir -p outputs/scicap/logs outputs/scicap/raw_results outputs/scicap/tables

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

BLIP2_RERANK_TOP_K="${BLIP2_RERANK_TOP_K:-128}"
BLIP2_PAIR_BATCH_SIZE="${BLIP2_PAIR_BATCH_SIZE:-32}"

echo "BLIP-2 zero-shot SciCap settings:"
echo "  rerank_top_k=${BLIP2_RERANK_TOP_K}"
echo "  pair_batch_size=${BLIP2_PAIR_BATCH_SIZE}"

python -m src.eval.evaluate_blip2 \
  --config configs/experiment_config.yaml \
  --dataset scicap \
  --split test \
  --rerank-top-k "${BLIP2_RERANK_TOP_K}" \
  --pair-batch-size "${BLIP2_PAIR_BATCH_SIZE}"
