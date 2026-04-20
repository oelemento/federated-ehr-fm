#!/bin/bash
#SBATCH --job-name=fedavg-inrnd-untied
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fedavg-inrnd-untied-%j.out
#SBATCH --error=logs/fedavg-inrnd-untied-%j.err

# Matched counterpart to fedavg_inround_tied.slurm with --untie-embeddings.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[slurm] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

SEED=20260411
CKPT_DIR="checkpoints_fedavg_untied_inrnd_s${SEED}"

PYTHONPATH=src python -u src/train_federated.py \
    --config configs/poc.yaml \
    --n-sites 5 \
    --n-rounds 20 \
    --local-epochs 1 \
    --local-steps 100 \
    --alpha 0.5 \
    --strategy fedavg \
    --mu 0.0 \
    --seed "$SEED" \
    --checkpoint-dir "$CKPT_DIR" \
    --untie-embeddings \
    --save-persite-last-round

echo "[slurm] done"
ls -la "$CKPT_DIR/"
