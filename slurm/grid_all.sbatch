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

# no_att_sum_pool encoder comparison: the cell
#   ind-none-sum-mlp
#   (reactant_tokens=ind, attention_on=none, pooling=sum, head=mlp)
# is trained once per graph encoder, GINE, GAT and DMPNN, so they land in one CSV
# with a `gnn` column for direct comparison:
#     ind-none-sum-mlp         (GINE)
#     ind-none-sum-mlp-gat     (GAT)
#     ind-none-sum-mlp-dmpnn   (DMPNN)
# The encoder is an extra grid axis (`--gnn_types gine gat dmpnn`); all other
# grid cells still default to GINE.
#
# attention_on is chosen by the CELL LABEL, not by a command-line flag: the grid
# runner overwrites run_opts.attention_on with the cell's value, so selecting
# `ind-none-sum-mlp` is what turns attention off. The `--attention_layer` /
# `--num_heads` options are ignored for a "none" cell (the model builds zero
# attention layers), so they are omitted here.
#
# Note: reactant_pooling=sum only affects the REACTANT side; products are always
# averaged (`_masked_mean`), see GraphReactionModel.forward.
#
# Full ablation grid (see run_approach1_grid.py) for reference:
#     baseline    ind-reactants-mean-linear
#     control     ind-none-mean-{linear,mlp}
#     A           ind-none-rn-{linear,mlp}
#     B           ind-reactants-rn-{linear,mlp}
#     approach_1  comb-reactants-rn-{linear,mlp}
#     approach_2  comb-reactants-mean-mlp
#     approach_3  comb-reactants-mean-linear
#     (this run)  ind-none-sum-mlp               <- no attention, sum pooling, mlp
#
# This run:      ind-none-sum-mlp x {gine, gat, dmpnn}
#
# 72 trainings = 3 cells x (10 BH FullCV split_70 + 4 BH Test1-4 + 10 Suzuki splits).
# Training runs the full --epochs 500 with early stopping disabled (--patience 0).
# Resubmit to resume (successful rows are skipped).
#
# Order: BH CV, then BH Test, then Suzuki, so the cheaper BH results arrive early.
#
# Outputs: ../logs/grid_no_att_sum_pool_gnn/approach1_per_split.csv
#          ../logs/grid_no_att_sum_pool_gnn/approach1_summary.csv
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
    --split_ids 0 1 2 3 4 5 6 7 8 9 --cells ind-none-sum-mlp --gnn_types gine gat dmpnn \
    --output_dir ../logs/grid_no_att_sum_pool_gnn/ \
    --epochs 500 --patience 0 --batch_size 128 --lr 1e-3 --weight_decay 1e-4 --dropout 0.1 \
    --emb_dim 384 --layer 2 --reaction_combine concat \
    --seed 42 --num_workers 2

echo "Finished $(date)"
