#!/bin/bash
#SBATCH --job-name=syncat-hier4
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm/logs/syncat-%j.out
#SBATCH --error=slurm/logs/syncat-%j.err

# Hierarchical four-level yield model (--yield_arch hierarchical4) over all datasets.
#
# The architecture: atoms -> molecules -> side groups -> reaction, with
# --hierarchy_passes 2 upward passes of --hierarchy_gine_layers 2 GINE layers
# each and the default feedback between the passes.
#
# Datasets (same coverage as the old Approach-2 grid run):
#   - Buchwald-Hartwig: 10 FullCV <split_70> + 4 Test, trained by train_bh.py
#   - Suzuki:           random splits 0-9, prepared (reusing valid graphs) and
#                       trained split-by-split by main_finetune.py / run_splits_sequential.py
#
# This replaced the legacy run_approach1_grid.py job: that runner is deliberately
# legacy-only (its cells vary legacy-only options), so the hierarchical model is
# trained directly with the shared runners instead.
#
# 24 trainings = 14 BH (10 CV + 4 Test) + 10 Suzuki splits.
# Training runs the full --epochs 500 with early stopping disabled (--patience 0).
# Resubmit to resume: a successful BH dataset or Suzuki split is skipped, and the
# Suzuki prepare stage only regenerates missing/invalid npz files.
#
# Order: BH first (cheaper, results arrive early), then Suzuki.
#
# Outputs:
#   ../logs/bh_hierarchical4/       bh_results.csv, bh_summary.csv, bh_training.log, runs/<dataset>/
#   ../logs/suzuki_hierarchical4/   <name>_results.csv, runs/<name>_<timestamp>/split_<id>/
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

# --- shared settings --------------------------------------------------------
ARCH="hierarchical4"
PASSES=2
GINE_LAYERS=2
HIER_ARGS="--yield_arch $ARCH --hierarchy_passes $PASSES --hierarchy_gine_layers $GINE_LAYERS"
TRAIN_ARGS="--epochs 500 --patience 0 --batch_size 128 --lr 1e-3 --weight_decay 1e-4 --dropout 0.1 --emb_dim 384 --num_workers 2 --seed 42"

# --- the actual run ---------------------------------------------------------

# 1) Buchwald-Hartwig: 10 FullCV split_70 + 4 Test.
#    train_bh.py sets --reaction_column rxn / --y_column Output itself and skips
#    datasets that already succeeded, so the job can be resubmitted.
echo "=== [1/3] BH hierarchical4 ==="
python -u train_bh.py $HIER_ARGS \
    --cv_ids 1 2 3 4 5 6 7 8 9 10 --test_ids 1 2 3 4 --split_columns split_70 \
    --log_dir ../logs/bh_hierarchical4/ \
    $TRAIN_ARGS

# 2) Suzuki: prepare (validate + reuse existing graphs; only regenerate missing
#    or invalid splits) with the strict hierarchical input validation.
echo "=== [2/3] Suzuki prepare ==="
python -u main_finetune.py --stage prepare --split_ids 0 1 2 3 4 5 6 7 8 9 \
    $HIER_ARGS \
    --log_dir ../logs/suzuki_hierarchical4/

# 3) Suzuki: train/evaluate splits one after another, skipping the successful
#    ones, so the job can be resubmitted.
echo "=== [3/3] Suzuki train ==="
python -u run_splits_sequential.py --split_ids 0 1 2 3 4 5 6 7 8 9 \
    $HIER_ARGS \
    --log_dir ../logs/suzuki_hierarchical4/ \
    $TRAIN_ARGS

echo "Finished $(date)"
