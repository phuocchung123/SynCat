#!/bin/bash
#SBATCH --job-name=syncat-grid-full
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm/logs/grid-%j.out
#SBATCH --error=slurm/logs/grid-%j.err

# Full factorial ablation grid across Buchwald-Hartwig and Suzuki-Miyaura.
# Runs BH then Suzuki on 1 GPU (about 13-20 h), writing one per-split CSV
# (grid_results.csv) and one summary CSV (grid_summary.csv).
#
# Cells (tok / att / pool / head):
#   0  ind/reactants/mean/linear   baseline
#   1  ind/reactants/mean/mlp      baseline_mlp
#   2  ind/reactants/rn/linear     B_linear
#   3  ind/reactants/rn/mlp        B_mlp
#   4  ind/none/mean/linear        control_linear
#   5  ind/none/mean/mlp           control_mlp
#   6  ind/none/rn/linear          A_linear
#   7  ind/none/rn/mlp             A_mlp
#   8  comb/reactants/mean/linear  approach3
#   9  comb/reactants/mean/mlp     approach2
#   10 comb/reactants/rn/linear    approach1_linear
#   11 comb/reactants/rn/mlp       approach1_mlp
#   12 comb/none/mean/linear       comb_none_mean_linear
#   13 comb/none/mean/mlp          comb_none_mean_mlp
#   14 comb/none/rn/linear         comb_none_rn_linear
#   15 comb/none/rn/mlp            comb_none_rn_mlp
#
# Datasets and splits:
#   - Buchwald-Hartwig (10 CV splits + Test1-4 = 14 splits per cell)
#   - Suzuki-Miyaura   (10 random splits = 10 splits per cell)
#   Total: 16 cells x 24 splits = 384 trainings on 1 GPU.
#
# Resumption:
#   Resubmitting the same command resumes without re-running finished splits.
#   Splits with status "success" in results.csv are skipped automatically.
#
# Optional Array / Task Mode:
#   - To run as a Slurm array: sbatch --array=0-31%4 slurm/grid_full.sbatch
#   - To run a single task:    TASK_ID=17 sbatch slurm/grid_full.sbatch
#
# Results & Collection:
#   Results land in:  <repo>/logs/grid/${RUN_NAME}/
#   Rebuild/collect merged CSVs:
#       python run_grid.py --collect --log_dir ../logs/grid/e400_p100/
#   Merged files:
#       <repo>/logs/grid/${RUN_NAME}/grid_results.csv   (per-split CSV)
#       <repo>/logs/grid/${RUN_NAME}/grid_summary.csv   (summary CSV)

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

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -d "$REPO_DIR/src" ]; then
    echo "ERROR: no 'src' under $REPO_DIR. Submit from the repository root:" >&2
    echo "       cd <repo> && sbatch slurm/grid_full.sbatch" >&2
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU:', torch.cuda.get_device_name(0))"

# Hyperparameters (configurable via environment variables)
EPOCHS="${EPOCHS:-400}"
PATIENCE="${PATIENCE:-100}"
BATCH_SIZE="${BATCH_SIZE:-128}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DROPOUT="${DROPOUT:-0.1}"
EMB_DIM="${EMB_DIM:-384}"
LAYER="${LAYER:-2}"
ATTENTION_LAYER="${ATTENTION_LAYER:-2}"
NUM_HEADS="${NUM_HEADS:-16}"
REACTION_COMBINE="${REACTION_COMBINE:-concat}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-2}"
RUN_NAME="${RUN_NAME:-e400_p100}"
LOG_DIR="../logs/grid/${RUN_NAME}/"

EXTRA_ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    EXTRA_ARGS+=(--task_id "$SLURM_ARRAY_TASK_ID")
elif [ -n "${TASK_ID:-}" ]; then
    EXTRA_ARGS+=(--task_id "$TASK_ID")
fi

# 1. Print the task's plan first with dry run
python -u run_grid.py \
    --profile full \
    "${EXTRA_ARGS[@]}" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    --emb_dim "$EMB_DIM" \
    --layer "$LAYER" \
    --attention_layer "$ATTENTION_LAYER" \
    --num_heads "$NUM_HEADS" \
    --reaction_combine "$REACTION_COMBINE" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --log_dir "$LOG_DIR" \
    --dry_run

# 2. Run the task (BH then Suzuki across all 16 cells)
python -u run_grid.py \
    --profile full \
    "${EXTRA_ARGS[@]}" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    --emb_dim "$EMB_DIM" \
    --layer "$LAYER" \
    --attention_layer "$ATTENTION_LAYER" \
    --num_heads "$NUM_HEADS" \
    --reaction_combine "$REACTION_COMBINE" \
    --seed "$SEED" \
    --num_workers "$NUM_WORKERS" \
    --log_dir "$LOG_DIR"

echo "========================================"
echo "Grid run completed at: $(date)"
echo "========================================"
