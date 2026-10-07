#!/bin/bash
#SBATCH --job-name=syncat-grid-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm/logs/syncat-%j.out
#SBATCH --error=slurm/logs/syncat-%j.err

# Fast preflight for the 16-cell grid:
#   1. pytest on the CPU test suite
#   2. --smoke: 48 tiny runs (16 cells x {BH cv01, BH test1, Suzuki split_0},
#      2 epochs, emb_dim 32, batch 8)
#   3. full-size 1-epoch preflight of the heaviest cells (Suzuki comb: T=105,
#      5460 RN pairs) to read off peak_gpu_memory_mb before the long job
#
# Submit from the repository root:
#   sbatch slurm/grid_smoke.sbatch
#
# Results:  <repo>/logs/grid_smoke/approach1_per_split.csv
#           <repo>/logs/grid_preflight/approach1_per_split.csv

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"

# --- environment (tolerant + overridable) ----------------------------------
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

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# --- locate repo (submit dir, NOT the spooled script) ----------------------
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -d "$REPO_DIR/src" ]; then
    echo "ERROR: no 'src' under $REPO_DIR. Submit from the repository root:" >&2
    echo "       cd <repo> && sbatch slurm/grid_smoke.sbatch" >&2
    exit 1
fi
cd "$REPO_DIR/src"

echo "Repository: $REPO_DIR"
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"
echo "Python executable: $(which python)"
python --version

if command -v git >/dev/null 2>&1; then
    echo "Git branch: $(git branch --show-current)"
    echo "Git commit: $(git rev-parse HEAD)"
fi

# --- thread limits match the allocation ------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- fail loudly if no GPU --------------------------------------------------
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"

# Clean progress logging for cluster output
export TQDM_DISABLE=1

# --- the actual run ---------------------------------------------------------
# pytest is not part of the training environment; install it if missing.
python -m pip install pytest

(cd "$REPO_DIR" && python -m pytest test -q)

python -u run_approach1_grid.py --smoke --datasets all --output_dir ../logs/grid_smoke/

# full-size 1-epoch preflight of the heaviest cells (Suzuki comb: T=105, 5460 RN pairs)
python -u run_approach1_grid.py --datasets all --cv_ids 1 --test_ids --split_ids 0 \
    --cells comb-reactants-rn-mlp comb-none-rn-mlp --epochs 1 --batch_size 128 \
    --emb_dim 384 --layer 3 --num_workers 2 --output_dir ../logs/grid_preflight/

echo "Finished $(date)"
