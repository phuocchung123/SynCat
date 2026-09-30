#!/bin/bash
#SBATCH --job-name=syncat-approach2-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/syncat-%j.out
#SBATCH --error=slurm/logs/syncat-%j.err

# One-epoch smoke test of the approach-2 grid: one BH split and one Suzuki split,
# 96/24/24 samples, batch 8, num_workers 0, GIN depth 1, emb_dim 32. Temporary
# subset npz files are written under the run folder; the prepared source data is
# never modified. Use this before approving the full runs.
#
# Run from the repository root:
#   sbatch slurm/approach2_grid_smoke.sbatch
# Slurm resolves the relative --output path against the submit directory, so
# submit from the repo root. slurm/logs/ is tracked, so that folder already
# exists; if you deleted it, recreate it first with:  mkdir -p slurm/logs
#
# Results:  <repo>/logs/approach2_grid_smoke/smoke/grid_results.csv
#           <repo>/logs/approach2_grid_smoke/smoke/grid_summary.csv

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

# Conda: use conda if it is already on PATH, otherwise load the cluster module.
# Override the module or environment name from the submit line, e.g.
#   ENV_NAME=syncat MODULE_NAME=Anaconda3 sbatch slurm/approach2_grid_smoke.sbatch
ENV_NAME="${ENV_NAME:-syncat}"
MODULE_NAME="${MODULE_NAME:-Miniconda3}"

if ! command -v module >/dev/null 2>&1; then
    for _modinit in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /etc/profile.d/module.sh; do
        if [ -r "$_modinit" ]; then . "$_modinit"; break; fi
    done
fi

if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load "$MODULE_NAME" || true
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Load it (e.g. 'module load $MODULE_NAME') or set MODULE_NAME." >&2
    exit 1
fi

# Enable conda activate in the Slurm non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the existing environment
conda activate "$ENV_NAME"

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

# --smoke overrides the epochs/batch/emb_dim/layer and the subset sizes.
python -u run_approach2_grid.py \
    --smoke \
    --attention_on reactants \
    --output_dir ../logs/approach2_grid_smoke/ \
    --run_name smoke

echo "========================================"
echo "Smoke test completed at: $(date)"
echo "========================================"
