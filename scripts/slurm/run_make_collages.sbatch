#!/bin/bash
#SBATCH --job-name=collages
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Generate qualitative retrieval collages from existing checkpoints/results.
# Resources: one GPU, 4 CPU cores, 64 GB RAM, 2 hour limit.
# Environment: activates the configured Conda environment.
# Launches: python -m src.analysis.make_collages.
# Outputs: PNG collages under figures/qualitative_examples/generated and logs under outputs/logs.
# Note: loads full models, so it is a GPU job even though it is not training.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
CONDA_ENV="${CONDA_ENV:-bakalarka}"
CONDA_INIT="${CONDA_INIT:-$HOME/miniconda3/bin/activate}"
cd "$PROJECT_ROOT"
mkdir -p figures/qualitative_examples/generated outputs/logs

echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi || true
if [[ ! -f "$CONDA_INIT" ]]; then
  echo "Set CONDA_INIT to your conda activate script before submitting this job."
  exit 1
fi
source "$CONDA_INIT" "$CONDA_ENV"

python -m src.analysis.make_collages
echo "Done: $(date)"
