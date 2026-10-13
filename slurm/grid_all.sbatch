#!/bin/bash
#SBATCH --job-name=syncat-grid-sum
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/syncat-%j.out
#SBATCH --error=slurm/logs/syncat-%j.err

# Approach-3 encoder comparison: the same cell
#   approach_3 = comb-reactants-mean-linear
#   (reactant_tokens=comb, attention_on=reactants, pooling=mean, head=linear)
# is trained once per graph encoder, GINE, GAT and DMPNN, so they land in one CSV
# with a `gnn` column for direct comparison:
#     comb-reactants-mean-linear         (GINE, the original)
#     comb-reactants-mean-linear-gat     (GAT)
#     comb-reactants-mean-linear-dmpnn   (DMPNN)
# The encoder is an extra grid axis (`--gnn_types gine gat dmpnn`); all other
# grid cells still default to GINE.
#
# Full ablation grid (see run_approach1_grid.py) for reference:
#     baseline    ind-reactants-mean-linear
#     control     ind-none-mean-{linear,mlp}
#     A           ind-none-rn-{linear,mlp}
#     B           ind-reactants-rn-{linear,mlp}
#     approach_1  comb-reactants-rn-{linear,mlp}
#     approach_2  comb-reactants-mean-mlp
#     approach_3  comb-reactants-mean-linear
#
# This run:      approach_3 x {gine, gat, dmpnn}
#
# 72 trainings = 3 cells x (10 BH FullCV split_70 + 4 BH Test1-4 + 10 Suzuki splits).
# Training runs the full --epochs 500 with early stopping disabled (--patience 0).
# Resubmit to resume (successful rows are skipped).
#
# Order: BH CV, then BH Test, then Suzuki, so the cheaper BH results arrive early.
#
# Outputs: ../logs/grid_approach3_gnn/approach1_per_split.csv
#          ../logs/grid_approach3_gnn/approach1_summary.csv
#
# Submit from the repository root:
#   sbatch slurm/grid_all.sbatch

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
    echo "       cd <repo> && sbatch slurm/grid_all.sbatch" >&2
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
python -u run_approach1_grid.py --datasets all \
    --cv_ids 1 2 3 4 5 6 7 8 9 10 --test_ids 1 2 3 4 --split_columns split_70 \
    --split_ids 0 1 2 3 4 5 6 7 8 9 --cells approach_3 --gnn_types gine\
    --output_dir ../logs/grid_approach3_ginonly_cf/ \
    --epochs 500 --patience 0 --batch_size 128 --lr 1e-3 --weight_decay 1e-4 --dropout 0.1 \
    --emb_dim 384 --layer 2 --attention_layer 2 --num_heads 16 --reaction_combine concat \
    --seed 42 --num_workers 2

echo "Finished $(date)"
