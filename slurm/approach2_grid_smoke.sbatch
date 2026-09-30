#!/bin/bash
#SBATCH --job-name=syncat-app2-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=execution/draft/approach2_grid_smoke/logs/syncat-%j.out
#SBATCH --error=execution/draft/approach2_grid_smoke/logs/syncat-%j.err

# Fast 1-epoch smoke test across all 4 cells on small subsets of BH and Suzuki:
# Verifies environment, GPU access, CUDA, and grid runner pipeline in ~2 minutes.
#
# Usage:
#   mkdir -p execution/draft/approach2_grid_smoke/logs
#   sbatch SynCat/slurm/approach2_grid_smoke.sbatch

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Compute node: $(hostname)"
echo "Allocated node: $SLURM_JOB_NODELIST"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-none}"
echo "Started at: $(date)"
echo "========================================"

# Load Conda
module load Miniconda3

# Enable conda activate in the Slurm non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the existing environment
conda activate syncat

# run_approach2_grid.py resolves its default paths (../Data/, ../logs/) from src/
cd "$SLURM_SUBMIT_DIR/SynCat/src"

echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"
echo "Python executable: $(which python)"
python --version

echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU:', torch.cuda.get_device_name(0))"

python -u run_approach2_grid.py \
    --smoke \
    --output_dir ../logs/smoke_approach2/

echo "========================================"
echo "Smoke test completed at: $(date)"
echo "========================================"
