#!/bin/bash
#SBATCH --job-name=syncat-bond-line
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/syncat-%j.out
#SBATCH --error=slurm/logs/syncat-%j.err

# Approach-3 molecular-encoder comparison across bond-line encoders.
# Three variants are trained with identical hyperparameters, changing only the
# molecular encoder, all on the bond-line graph representation:
#   1. bond_line -> gin    (standard GIN, molecular mean pooling)
#   2. bond_line -> gat    (standard GAT, 4 concatenated heads, mean pooling)
#   3. bond_line -> dmpnn  (D-MPNN adapted to the bond-line graph, mean pooling)
# Approach 3 outside the encoder is unchanged:
#   comb-reactants-mean-linear (attention_on=reactants, pooling=mean,
#   reaction_combine=concat, head=linear).
#
# The atom/GINE baseline is deliberately not part of this job; run it separately
# with "--graph_representation atom --gnn_types gine" if it is needed.
#
# The three bond-line variants can also be combined into a single run:
#   ... --graph_representation bond_line --gnn_types gin gat dmpnn
#
# 72 trainings total = 3 variants x (10 BH FullCV split_70 + 4 BH Test1-4 +
# 10 Suzuki splits). Training runs the full --epochs 500 with early stopping
# disabled (--patience 0). Resubmit to resume: a cell is skipped only when a
# result with the same configuration fingerprint (representation, encoder,
# architecture, seed, recipe) already succeeded. Attempted-but-unfinished cells
# are restarted from scratch (their checkpoint is removed first).
#
# Outputs: ../logs/grid_approach3_bond_line/approach1_per_split.csv
#          ../logs/grid_approach3_bond_line/approach1_summary.csv
#
# Submit from the repository root:
#   sbatch slurm/grid_bond_line.sbatch

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
    echo "       cd <repo> && sbatch slurm/grid_bond_line.sbatch" >&2
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

# --- shared experiment recipe (from slurm/grid_all.sbatch) ------------------
COMMON_ARGS=(
  --datasets all
  --cv_ids 1 2 3 4 5 6 7 8 9 10
  --test_ids 1 2 3 4
  --split_columns split_70
  --split_ids 0 1 2 3 4 5 6 7 8 9
  --cells approach_3
  --output_dir ../logs/grid_approach3_bond_line/
  --epochs 500 --patience 0
  --batch_size 128 --lr 1e-3 --weight_decay 1e-4
  --dropout 0.1 --emb_dim 384 --layer 2
  --attention_layer 2 --num_heads 16
  --reaction_combine concat --seed 42 --num_workers 2
)

# --- the three bond-line variants -------------------------------------------
echo "=== Variant 1/3: bond_line / gin ==="
python -u run_approach1_grid.py "${COMMON_ARGS[@]}" \
    --graph_representation bond_line --gnn_types gin

echo "=== Variant 2/3: bond_line / gat ==="
python -u run_approach1_grid.py "${COMMON_ARGS[@]}" \
    --graph_representation bond_line --gnn_types gat --gat_heads 4

echo "=== Variant 3/3: bond_line / dmpnn ==="
python -u run_approach1_grid.py "${COMMON_ARGS[@]}" \
    --graph_representation bond_line --gnn_types dmpnn

# Alternative: run all three bond-line variants in one call.
# python -u run_approach1_grid.py "${COMMON_ARGS[@]}" \
#     --graph_representation bond_line --gnn_types gin gat dmpnn --gat_heads 4

echo "Finished $(date)"
