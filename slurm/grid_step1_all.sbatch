#!/bin/bash
#SBATCH --job-name=syncat-step1-all
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/syncat-step1-all-%j.out
#SBATCH --error=slurm/logs/syncat-step1-all-%j.err

# Run the COMPLETE step-1 grid (384 runs) sequentially on ONE GPU, like
# slurm/grid_all.sbatch.
#
#   16 cells x (10 BH CV splits + 4 BH Tests + 10 Suzuki splits) = 384 runs
#
# Submit from the repository root:
#   sbatch slurm/grid_step1_all.sbatch
#
# Cost: about 100 GPU-hours (about 1.1 s/epoch on BH, 2.7 s/epoch on Suzuki).
# On one GPU that is roughly 4-5 days. The default wall time is 3 days; Slurm
# will stop the job at the limit. Resubmit the same command to resume:
# successful (dataset, split, cell) rows are skipped.
#
# If your cluster allows a longer wall time, submit with:
#   sbatch --time=5-00:00:00 slurm/grid_step1_all.sbatch
#
# Environment overrides (optional):
#   ENV_NAME=syncat  MODULE_NAME=Miniconda3
#   OUT=../logs/grid_step1_all/  EPOCHS=500  MC_PASSES=30
#   WORKERS=2  BATCH_SIZE=128  DRY_RUN=1
# ---------------------------------------------------------------------------

set -euo pipefail

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) started $(date)"

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
    echo "       cd <repo> && sbatch slurm/grid_step1_all.sbatch" >&2
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

# --- settings ---------------------------------------------------------------
OUT="${OUT:-../logs/grid_step1_all/}"
EPOCHS="${EPOCHS:-500}"
MC_PASSES="${MC_PASSES:-30}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

mkdir -p "$OUT"

# --- the full grid, one GPU, sequential ------------------------------------
CMD=(python -u -W ignore::FutureWarning run_approach1_grid.py
    --datasets all
    --cv_ids 1 2 3 4 5 6 7 8 9 10
    --test_ids 1 2 3 4
    --split_columns split_70
    --split_ids 0 1 2 3 4 5 6 7 8 9
    --cells report16
    --output_dir "$OUT"
    --epochs "$EPOCHS" --patience 0 --batch_size "$BATCH_SIZE"
    --lr 1e-3 --weight_decay 1e-5 --dropout 0.1
    --emb_dim 384 --layer 2 --attention_layer 2 --num_heads 16
    --reaction_combine concat --seed 42 --num_workers "$WORKERS"
    --shuffle_train --target_scaling zscore --loss mse
    --lr_scheduler multistep --lr_milestones 400 450 --lr_gamma 0.1
    --merge_valid_into_train --checkpoint_select last
    --mc_dropout_passes "$MC_PASSES" --clip_predictions
    --device 0)

echo "OUT=$OUT"
echo "EPOCHS=$EPOCHS  MC_PASSES=$MC_PASSES  WORKERS=$WORKERS  BATCH_SIZE=$BATCH_SIZE"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: exiting before the GPU check."
    exit 0
fi

# --- fail loudly if no GPU --------------------------------------------------
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"

# --- run --------------------------------------------------------------------
"${CMD[@]}"

echo "Finished $(date)"
