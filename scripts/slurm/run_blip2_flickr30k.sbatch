#!/bin/bash
#SBATCH --job-name=blip2_flickr30k
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G

# Full BLIP-2 zero-shot evaluation on Flickr30K with ITM rerank@128.
# Resources: one GPU, 8 CPU cores, 64 GB RAM, 24 hour limit.
# Environment: activates the configured Conda environment.
# Launches: python -m src.eval.evaluate_blip2 with top-k ITM reranking.
# Outputs: raw JSON metrics under outputs/raw/ and logs under outputs/logs/.
# Expensive: run through SLURM, not on the login node, because ITM scores many pairs.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-bakalarka}"
CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/bin/activate}"
cd "$PROJECT_ROOT"
mkdir -p outputs/logs

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

bash scripts/run_smoke_tests.sh
test -d data/flickr30k/flickr30k_images
test -f data/flickr30k/flickr30k/dataset.json
python -m src.eval.evaluate_blip2 --config configs/experiment_config.yaml --dataset flickr30k --split test --rerank-top-k 128
