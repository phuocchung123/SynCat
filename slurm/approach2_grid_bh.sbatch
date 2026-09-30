#!/bin/bash
#SBATCH --job-name=syncat-approach2-bh
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=execution/syncat_regress/app2/syncat_6/logs/syncat-%j.out
#SBATCH --error=execution/syncat_regress/app2/syncat_6/logs/syncat-%j.err

# Approach-2 grid on the Buchwald-Hartwig datasets (FullCV 1-10 at split_70 plus
# Test1-4), always with attention_on=reactants:
#   reactant_tokens {ind, comb} x head {linear, mlp}
#   current_model = ind/linear     approach_3 = comb/linear
#   approach_2    = comb/mlp       ind_mlp    = ind/mlp (control)
# 14 datasets x 4 cells = 56 trainings. Resubmit to continue after a time limit:
# a dataset/split/cell row that already succeeded is skipped, and an interrupted
# one is retrained from scratch (its checkpoint is removed first).
#
#   mkdir -p execution/approach2/approach2_grid_bh/logs
#   sbatch SynCat/slurm/approach2_grid_bh.sbatch
#
# Results:  <repo>/logs/approach2_grid_bh/e400/grid_results.csv  (per split)
#           <repo>/logs/approach2_grid_bh/e400/grid_summary.csv  (mean/std)

set -euo pipefail

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Compute node: $(hostname)"
echo "Allocated node: $SLURM_NODELIST"
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

# run_approach2_grid.py resolves its default paths (../Data/, ../logs/) from src/;
# the repository root is found relative to this script, not the submit directory.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR/src"

echo "Repository: $REPO_DIR"
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
    --datasets Buchwald-Hartwig \
    --cv_ids 1 2 3 4 5 6 7 8 9 10 \
    --test_ids 1 2 3 4 \
    --attention_on reactants \
    --output_dir ../logs/approach2_grid_bh/ \
    --run_name e400 \
    --epochs 400 \
    --patience 50 \
    --num_workers 2 \
    --attention_layer 2 \
    --num_heads 16 \
    --layer 2 \
    --emb_dim 384

echo "========================================"
echo "Training completed at: $(date)"
echo "========================================"
