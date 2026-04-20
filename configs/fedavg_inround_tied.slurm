#!/bin/bash
#SBATCH --job-name=fedavg-inrnd-tied
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/fedavg-inrnd-tied-%j.out
#SBATCH --error=logs/fedavg-inrnd-tied-%j.err

# In-round per-site divergence experiment: run FedAvg with tied embeddings,
# save per-site state BEFORE averaging on the LAST round. This captures
# per-site drift at the ~local_steps scale (what real FedAvg actually averages)
# rather than the 3-epoch drift used in the previous null experiment.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[slurm] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

SEED=20260411
CKPT_DIR="checkpoints_fedavg_tied_inrnd_s${SEED}"

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
    --save-persite-last-round

echo "[slurm] done"
ls -la "$CKPT_DIR/"
