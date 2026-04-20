#!/bin/bash
#SBATCH --job-name=eval-inrnd
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/eval-inrnd-%j.out
#SBATCH --error=logs/eval-inrnd-%j.err

# Evaluate the saved best.pt from both in-round FedAvg runs (tied and untied).
# Needed to confirm the reported paper result (tied FedAvg AUPRC ~0.08) against
# the just-finished run, or whether it reproduces AUPRC ~0.30 (matching the
# post-hoc average of the per-site snapshots).

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/processed

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[eval] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

SEED=20260411

for TAG in fedavg_tied_inrnd fedavg_untied_inrnd; do
    CKPT="checkpoints_${TAG}_s${SEED}/best.pt"
    if [ -f "$CKPT" ]; then
        echo ""
        echo "=== Evaluating $CKPT ==="
        PYTHONPATH=src python -u src/evaluate.py \
            --checkpoint "$CKPT" \
            --out-json "data/processed/eval_${TAG}_s${SEED}.json" \
            --figure-path "/tmp/skip_eval_${TAG}.png"
    else
        echo "[eval] missing $CKPT"
    fi
done

echo "[eval] done"
