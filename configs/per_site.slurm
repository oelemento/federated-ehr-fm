#!/bin/bash
#SBATCH --job-name=per-site
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/per-site-%j.out
#SBATCH --error=logs/per-site-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[slurm] per-site training on $(hostname)"
nvidia-smi || true

PYTHONPATH=src python -u src/train_per_site.py \
    --config configs/poc.yaml \
    --n-sites 5 --alpha 0.5 --num-epochs 3 --seed 20260411

echo "[slurm] done"
