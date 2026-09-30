#!/bin/bash
#SBATCH --job-name=syncat-app2-suzuki
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=execution/syncat_regress/app2/syncat_9/logs/syncat-%j.out
#SBATCH --error=execution/syncat_regress/app2/syncat_9/logs/syncat-%j.err

# Approach 2 grid on the 10 Suzuki splits (split_0 to split_9):
#   4 cells per split:
#     1. current_model: reactant_tokens=ind, head=linear
#     2. approach_3:    reactant_tokens=comb, head=linear
#     3. approach_2:    reactant_tokens=comb, head=mlp
#     4. mlp_control:   reactant_tokens=ind, head=mlp
#
# 10 splits x 4 cells = 40 trainings. Resubmit to continue after a time limit (skips successful runs).
#
# Usage from repo root or submit dir:
#   mkdir -p execution/draft/approach2_grid_suzuki/logs
#   sbatch SynCat/slurm/approach2_grid_suzuki.sbatch

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

# Match CPU-based libraries with the Slurm allocation
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Show the allocated GPU, and stop here rather than silently training on the CPU
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU:', torch.cuda.get_device_name(0))"

python -u run_approach2_grid.py \
    --datasets suzuki \
    --split_ids 0 1 2 3 4 5 6 7 8 9 \
    --output_dir ../logs/suzuki_approach2_grid/ \
    --epochs 100 \
    --patience 10 \
    --num_workers 2 \
    --attention_layer 1 \
    --num_heads 1 \
    --layer 3 \
    --emb_dim 384

echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
