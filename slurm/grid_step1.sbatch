#!/bin/bash
#SBATCH --job-name=syncat-step1
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/syncat-step1-%j.out
#SBATCH --error=slurm/logs/syncat-step1-%j.err

# Step 1: repeat the report's 16-configuration grid with the reaction_yield_nn
# training recipe (z-scored targets, MSE, shuffling, MultiStepLR, 70% training
# set, last-epoch checkpoint, 30 MC-dropout passes, clipped yields).
#
# ---------------------------------------------------------------------------
# QUICK START (submit from the repository root)
# ---------------------------------------------------------------------------
# 1) Pilot: 8 runs (G1, G6, G10, G12 on BH fullcv01 and Suzuki split_0).
#       MODE=pilot sbatch slurm/grid_step1.sbatch
#
# 2) Full grid, 384 runs = 16 cells x (10 BH CV + 4 BH Tests + 10 Suzuki),
#    as four parallel jobs with separate output folders:
#       MODE=bh_g1_8     sbatch slurm/grid_step1.sbatch
#       MODE=bh_g9_16    sbatch slurm/grid_step1.sbatch
#       MODE=suzuki_g1_8 sbatch slurm/grid_step1.sbatch
#       MODE=suzuki_g9_16 sbatch slurm/grid_step1.sbatch
#
# 3) Plumbing check on a GPU (tiny slices, 4 cells, 2 epochs, no BH test):
#       MODE=smoke_light sbatch slurm/grid_step1.sbatch
#
#    Cost from the report's logs (~1.1 s/epoch on BH, ~2.7 s/epoch on Suzuki):
#    about 100 GPU-hours in total, or about 1 day on 4 GPUs.
#
# ---------------------------------------------------------------------------
# MODE profiles
# ---------------------------------------------------------------------------
#   full          all 16 report cells, all 24 splits (default; 384 runs)
#   pilot         G1/G6/G10/G12 on BH fullcv01 + Suzuki split_0 (8 runs)
#   bh_g1_8       BH CV+Tests, G1-G8 (112 runs)
#   bh_g9_16      BH CV+Tests, G9-G16 (112 runs)
#   suzuki_g1_8   Suzuki split_0..9, G1-G8 (80 runs)
#   suzuki_g9_16  Suzuki split_0..9, G9-G16 (80 runs)
#   smoke_light   tiny slices, 4 cells, 2 epochs (8 runs; plumbing only)
#
# Environment overrides (all optional):
#   DATASETS=all|bh|suzuki   CELLS="label label ..."   OUT=../logs/...
#   EXTRA_ARGS="--bh_splits ..."   EPOCHS=500   MC_PASSES=30
#   WORKERS=2   BATCH_SIZE=128   ENV_NAME=syncat   MODULE_NAME=Miniconda3
#   DRY_RUN=1   print the resolved command and exit before using the GPU
#
# Example with a shorter wall time and a custom output folder:
#   MODE=bh_g1_8 OUT=../logs/step1_try1/ sbatch --time=1-00:00:00 slurm/grid_step1.sbatch
#
# Resubmit the same command to resume: successful (dataset, split, cell) rows
# are skipped. Add --rerun_successful via EXTRA_ARGS to repeat them.
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
    echo "       cd <repo> && MODE=pilot sbatch slurm/grid_step1.sbatch" >&2
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
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

# --- resolve the run profile ------------------------------------------------
MODE="${MODE:-full}"
DATASETS="${DATASETS:-}"
CELLS="${CELLS:-}"
OUT="${OUT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EPOCHS="${EPOCHS:-500}"
MC_PASSES="${MC_PASSES:-30}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"

PILOT_CELLS="ind-reactants-mean-linear ind-none-rn-mlp comb-reactants-mean-mlp comb-reactants-rn-mlp"
BH_G1_8_CELLS="ind-reactants-mean-linear ind-reactants-mean-mlp ind-none-mean-linear ind-none-mean-mlp ind-none-rn-linear ind-none-rn-mlp ind-reactants-rn-linear ind-reactants-rn-mlp"
BH_G9_16_CELLS="comb-reactants-mean-linear comb-reactants-mean-mlp comb-reactants-rn-linear comb-reactants-rn-mlp comb-none-mean-linear comb-none-mean-mlp comb-none-rn-linear comb-none-rn-mlp"
SMOKE_ARGS=()

case "$MODE" in
    pilot)
        DATASETS="${DATASETS:-all}"
        CELLS="${CELLS:-$PILOT_CELLS}"
        OUT="${OUT:-../logs/step1_pilot/}"
        if [ -z "$EXTRA_ARGS" ]; then
            EXTRA_ARGS="--bh_splits fullcv01_split70 --suzuki_splits split_0"
        fi
        ;;
    bh_g1_8)
        DATASETS="${DATASETS:-bh}"
        CELLS="${CELLS:-$BH_G1_8_CELLS}"
        OUT="${OUT:-../logs/step1_bh_g1_8/}"
        ;;
    bh_g9_16)
        DATASETS="${DATASETS:-bh}"
        CELLS="${CELLS:-$BH_G9_16_CELLS}"
        OUT="${OUT:-../logs/step1_bh_g9_16/}"
        ;;
    suzuki_g1_8)
        DATASETS="${DATASETS:-suzuki}"
        CELLS="${CELLS:-$BH_G1_8_CELLS}"
        OUT="${OUT:-../logs/step1_suzuki_g1_8/}"
        ;;
    suzuki_g9_16)
        DATASETS="${DATASETS:-suzuki}"
        CELLS="${CELLS:-$BH_G9_16_CELLS}"
        OUT="${OUT:-../logs/step1_suzuki_g9_16/}"
        ;;
    full)
        DATASETS="${DATASETS:-all}"
        CELLS="${CELLS:-report16}"
        OUT="${OUT:-../logs/grid_step1/}"
        ;;
    smoke_light)
        DATASETS="${DATASETS:-all}"
        CELLS="${CELLS:-report16}"
        OUT="${OUT:-../logs/step1_smoke_light/}"
        # Keep an explicit EPOCHS/MC_PASSES override; otherwise use tiny values.
        if [ "$EPOCHS" = "500" ]; then EPOCHS=2; fi
        if [ "$MC_PASSES" = "30" ]; then MC_PASSES=3; fi
        SMOKE_ARGS=(--smoke --smoke_profile light)
        ;;
    *)
        echo "ERROR: unknown MODE='$MODE'. Use pilot, bh_g1_8, bh_g9_16, suzuki_g1_8, suzuki_g9_16, full, or smoke_light." >&2
        exit 2
        ;;
esac

mkdir -p "$OUT"

# --- build the command (arrays keep spaces in labels/arguments safe) --------
read -r -a CELL_ARGS <<< "$CELLS"
CMD=(python -u -W ignore::FutureWarning run_approach1_grid.py
    --datasets "$DATASETS"
    --cv_ids 1 2 3 4 5 6 7 8 9 10
    --test_ids 1 2 3 4
    --split_columns split_70
    --split_ids 0 1 2 3 4 5 6 7 8 9
    --cells "${CELL_ARGS[@]}"
    --output_dir "$OUT"
    --epochs "$EPOCHS" --patience 0 --batch_size "$BATCH_SIZE"
    --lr 1e-3 --weight_decay 1e-5 --dropout 0.1
    --emb_dim 384 --layer 2 --attention_layer 2 --num_heads 16
    --reaction_combine concat --seed 42 --num_workers "$WORKERS"
    --shuffle_train --target_scaling zscore --loss mse
    --lr_scheduler multistep --lr_milestones 400 450 --lr_gamma 0.1
    --merge_valid_into_train --checkpoint_select last
    --mc_dropout_passes "$MC_PASSES" --clip_predictions)

if [ -n "$EXTRA_ARGS" ]; then
    read -r -a EXTRA_ARRAY <<< "$EXTRA_ARGS"
    CMD+=("${EXTRA_ARRAY[@]}")
fi
if [ ${#SMOKE_ARGS[@]} -gt 0 ]; then
    CMD+=("${SMOKE_ARGS[@]}")
fi

echo "MODE=$MODE"
echo "DATASETS=$DATASETS"
echo "OUT=$OUT"
echo "EPOCHS=$EPOCHS  MC_PASSES=$MC_PASSES  WORKERS=$WORKERS  BATCH_SIZE=$BATCH_SIZE"
echo "CELLS=$CELLS"
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

# --- the actual run ---------------------------------------------------------
"${CMD[@]}"

echo "Finished $(date)"
