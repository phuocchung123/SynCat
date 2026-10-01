#!/bin/bash
#SBATCH --job-name=syncat-approach1-bh
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/syncat-%j.out
#SBATCH --error=slurm/logs/syncat-%j.err

# Approach-1 grid on the Buchwald-Hartwig datasets (FullCV 1-10 at split_70 plus Test 1-4):
#   8 cells per split:
#     1. current_model:      reactant_tokens=ind,  reactant_pooling=mean, head=linear
#     2. approach_3:         reactant_tokens=comb, reactant_pooling=mean, head=linear
#     3. approach_2:         reactant_tokens=comb, reactant_pooling=mean, head=mlp
#     4. approach_1_linear:  reactant_tokens=comb, reactant_pooling=rn,   head=linear
#     5. approach_1_mlp:     reactant_tokens=comb, reactant_pooling=rn,   head=mlp
#     6. option_b_linear:    reactant_tokens=ind,  reactant_pooling=rn,   head=linear
#     7. option_b_mlp:       reactant_tokens=ind,  reactant_pooling=rn,   head=mlp
#     8. ind_mean_mlp:       reactant_tokens=ind,  reactant_pooling=mean, head=mlp
# 14 datasets x 8 cells = 112 trainings. Resubmit to continue after a time limit (skips successful runs).
#
# Submit from the repository root:
#   sbatch slurm/approach1_grid_bh.sbatch
#
# Results:  <repo>/logs/bh_approach1_grid/approach1_per_split.csv  (per split)
#           <repo>/logs/bh_approach1_grid/approach1_summary.csv    (summary)

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
    echo "       cd <repo> && sbatch slurm/approach1_grid_bh.sbatch" >&2
    exit 1
fi
cd "$REPO_DIR/src"

echo "Repository: $REPO_DIR"
echo "Working directory: $(pwd)"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Conda prefix: $CONDA_PREFIX"
echo "Python executable: $(which python)"
python --version

echo "Git branch: $(git branch --show-current)"
echo "Git commit: $(git rev-parse HEAD)"

# --- thread limits match the allocation ------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# --- fail loudly if no GPU --------------------------------------------------
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"

# Clean progress logging for cluster output
export TQDM_DISABLE=1

# --- the actual run ---------------------------------------------------------
python -u run_approach1_grid.py \
    --datasets bh \
    --cv_ids 1 2 3 4 5 6 7 8 9 10 \
    --test_ids 1 2 3 4 \
    --split_columns split_70 \
    --attention_on reactants \
    --output_dir ../logs/bh_approach1_grid/ \
    --epochs 100 \
    --patience 10 \
    --num_workers 2 \
    --attention_layer 1 \
    --num_heads 1 \
    --layer 3 \
    --emb_dim 384

echo "Finished $(date)"
