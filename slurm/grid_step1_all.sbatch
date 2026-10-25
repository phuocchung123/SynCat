#!/bin/bash
#SBATCH --job-name=syncat-step1-all
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm/logs/syncat-step1-all-%j.out
#SBATCH --error=slurm/logs/syncat-step1-all-%j.err

# Run the COMPLETE step-1 grid (384 runs) in ONE Slurm job on 4 GPUs.
#
# Each GPU runs one independent stream:
#   GPU 0: BH CV + BH Tests, G1-G8      112 runs
#   GPU 1: BH CV + BH Tests, G9-G16     112 runs
#   GPU 2: Suzuki splits 0-9, G1-G8      80 runs
#   GPU 3: Suzuki splits 0-9, G9-G16     80 runs
#   total                               384 runs
#
# Submit from the repository root:
#   sbatch slurm/grid_step1_all.sbatch
#
# Requirements: a node with 4 GPUs (--gres=gpu:4). If your cluster only has
# 1-GPU nodes, submit the four MODE=... jobs from slurm/grid_step1.sbatch.
#
# Cost: about 100 GPU-hours total, roughly 25-30 h wall time with 4 GPUs in
# parallel. The default wall time is 3 days, so the streams fit comfortably.
#
# Resume: resubmit the same command. Successful (dataset, split, cell) rows in
# each stream are skipped, so it continues where it stopped.
#
# Environment overrides (optional):
#   ENV_NAME=syncat  MODULE_NAME=Miniconda3
#   EPOCHS=500  MC_PASSES=30  WORKERS=2  BATCH_SIZE=128
#   OUT_ROOT=../logs  DRY_RUN=1
#
# Example with a different output root and environment:
#   OUT_ROOT=/scratch/me/step1 ENV_NAME=myenv sbatch slurm/grid_step1_all.sbatch
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
OUT_ROOT="${OUT_ROOT:-../logs}"
EPOCHS="${EPOCHS:-500}"
MC_PASSES="${MC_PASSES:-30}"
WORKERS="${WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1

BH_G1_8_CELLS="ind-reactants-mean-linear ind-reactants-mean-mlp ind-none-mean-linear ind-none-mean-mlp ind-none-rn-linear ind-none-rn-mlp ind-reactants-rn-linear ind-reactants-rn-mlp"
BH_G9_16_CELLS="comb-reactants-mean-linear comb-reactants-mean-mlp comb-reactants-rn-linear comb-reactants-rn-mlp comb-none-mean-linear comb-none-mean-mlp comb-none-rn-linear comb-none-rn-mlp"

mkdir -p "$OUT_ROOT"

# --- one grid stream per GPU ------------------------------------------------
run_stream() {
    local gpu="$1"
    local dataset="$2"
    local cells="$3"
    local out="$4"
    local tag="$5"

    echo "[$tag] GPU $gpu -> $out"
    mkdir -p "$out"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -W ignore::FutureWarning \
        run_approach1_grid.py \
        --datasets "$dataset" \
        --cv_ids 1 2 3 4 5 6 7 8 9 10 --test_ids 1 2 3 4 \
        --split_columns split_70 \
        --split_ids 0 1 2 3 4 5 6 7 8 9 \
        --cells $cells --output_dir "$out" \
        --epochs "$EPOCHS" --patience 0 --batch_size "$BATCH_SIZE" \
        --lr 1e-3 --weight_decay 1e-5 --dropout 0.1 \
        --emb_dim 384 --layer 2 --attention_layer 2 --num_heads 16 \
        --reaction_combine concat --seed 42 --num_workers "$WORKERS" \
        --shuffle_train --target_scaling zscore --loss mse \
        --lr_scheduler multistep --lr_milestones 400 450 --lr_gamma 0.1 \
        --merge_valid_into_train --checkpoint_select last \
        --mc_dropout_passes "$MC_PASSES" --clip_predictions \
        --device 0
}

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY_RUN=1: would launch four streams:"
    echo "  GPU 0: bh     G1-G8   -> $OUT_ROOT/step1_bh_g1_8/"
    echo "  GPU 1: bh     G9-G16  -> $OUT_ROOT/step1_bh_g9_16/"
    echo "  GPU 2: suzuki G1-G8   -> $OUT_ROOT/step1_suzuki_g1_8/"
    echo "  GPU 3: suzuki G9-G16  -> $OUT_ROOT/step1_suzuki_g9_16/"
    exit 0
fi

# --- fail loudly if the allocation does not have 4 GPUs ---------------------
nvidia-smi
GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
echo "Visible GPUs: $GPU_COUNT"
if [ "$GPU_COUNT" -lt 4 ]; then
    echo "ERROR: this job needs 4 GPUs, but only $GPU_COUNT are visible." >&2
    echo "       Submit with --gres=gpu:4, or use the four MODE=... jobs." >&2
    exit 1
fi

# --- launch all four streams in parallel ------------------------------------
pids=()
names=()
run_stream 0 bh "$BH_G1_8_CELLS" "$OUT_ROOT/step1_bh_g1_8/" "bh_g1_8" &
pids+=("$!"); names+=("bh_g1_8")
run_stream 1 bh "$BH_G9_16_CELLS" "$OUT_ROOT/step1_bh_g9_16/" "bh_g9_16" &
pids+=("$!"); names+=("bh_g9_16")
run_stream 2 suzuki "$BH_G1_8_CELLS" "$OUT_ROOT/step1_suzuki_g1_8/" "suzuki_g1_8" &
pids+=("$!"); names+=("suzuki_g1_8")
run_stream 3 suzuki "$BH_G9_16_CELLS" "$OUT_ROOT/step1_suzuki_g9_16/" "suzuki_g9_16" &
pids+=("$!"); names+=("suzuki_g9_16")

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[${names[$i]}] finished $(date)"
    else
        echo "[${names[$i]}] FAILED $(date)" >&2
        status=1
    fi
done

echo "All streams finished $(date) with status $status"
exit "$status"
