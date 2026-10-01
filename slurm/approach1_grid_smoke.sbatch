#!/bin/bash
#SBATCH --job-name=syncat-approach1-smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/syncat-%j.out
#SBATCH --error=slurm/logs/syncat-%j.err

# One-epoch correctness smoke test of the approach-1 grid: one BH split and one
# Suzuki split, 64/32/32 samples, batch 8 (BH) and 4 (Suzuki), emb_dim 32,
# num_workers 0. Temporary subset npz files are written under the run folder;
# the prepared source data is never modified. The eight cells are
#   reactant_tokens {ind, comb} x reactant_pooling {mean, rn} x head {linear, mlp}.
#
# The metrics are a CORRECTNESS CHECK ONLY, never results (the CSV is labelled
# profile=laptop_smoke_correctness_only, note=not a result). Use this before
# submitting the full approach1_grid_bh / approach1_grid_suzuki jobs.
#
# Run from the repository root:
#   sbatch slurm/approach1_grid_smoke.sbatch
# Slurm resolves the relative --output path against the submit directory, so
# submit from the repo root. slurm/logs/ is tracked, so that folder already
# exists; if you deleted it, recreate it first with:  mkdir -p slurm/logs
#
# Results:  <repo>/logs/approach1_grid/grid_results.csv   (per cell/split)
#           <repo>/logs/approach1_grid/grid_summary.csv   (mean/std)
#           <repo>/logs/approach1_grid/server_commands.txt

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
# Override the module or environment name from the submit line, e.g.
#   ENV_NAME=syncat MODULE_NAME=Anaconda3 sbatch slurm/approach1_grid_smoke.sbatch
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

# Enable conda activate in the Slurm non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the existing environment
conda activate "$ENV_NAME"

# Slurm copies this script to /var/spool/slurmd/<jobid>/slurm_script and runs
# that copy, so BASH_SOURCE points into the spool directory, not your checkout.
# Slurm exports SLURM_SUBMIT_DIR = the directory sbatch was run from; submit
# from the repository root (`sbatch slurm/<file>.sbatch`) so it is the repo root.
REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -d "$REPO_DIR/src" ]; then
    echo "ERROR: no 'src' under $REPO_DIR. Submit from the repository root:" >&2
    echo "       cd <repo> && sbatch slurm/approach1_grid_smoke.sbatch" >&2
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

# Match CPU-based libraries with the Slurm allocation
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

# Show the allocated GPU, and stop here rather than silently training on the CPU
nvidia-smi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA is not available'; print('GPU:', torch.cuda.get_device_name(0))"

# run_approach1_grid.py always applies the tiny correctness profile; it writes
# grid_results.csv / grid_summary.csv and server_commands.txt under --log_dir.
python -u run_approach1_grid.py \
    --log_dir ../logs/approach1_grid/

echo "========================================"
echo "Smoke test completed at: $(date)"
echo "========================================"
