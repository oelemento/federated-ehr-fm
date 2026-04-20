#!/bin/bash
#SBATCH --job-name=error-bars
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/error-bars-%j.out
#SBATCH --error=logs/error-bars-%j.err

# Fill in error bars for remaining single-run methods:
# 1. FedProx tied (fullepoch config): 3 seeds × 20 rounds ≈ 3h
# 2. Linear bag-of-DX: 3 seeds × 10 epochs ≈ 30 min
# Marginal and Repeat are deterministic — no error bars needed.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[error-bars] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

SEEDS="20260411 20260412 20260413"

# --- FedProx tied (fullepoch config, the "best" of 6 configs tested) ---
for SEED in $SEEDS; do
    TAG="fedprox_tied_s${SEED}"
    CKPT_DIR="checkpoints_${TAG}"
    EVAL_JSON="data/processed/eval_${TAG}.json"

    echo ""
    echo "================================================================"
    echo "[error-bars] FedProx tied seed=$SEED"
    echo "================================================================"

    PYTHONPATH=src python -u src/train_federated.py \
        --config configs/poc.yaml \
        --n-sites 5 \
        --n-rounds 20 \
        --local-epochs 1 \
        --local-steps 500 \
        --lr-decay 0.95 \
        --alpha 0.5 \
        --strategy fedprox \
        --mu 0.01 \
        --seed "$SEED" \
        --checkpoint-dir "$CKPT_DIR"

    PYTHONPATH=src python -u src/evaluate.py \
        --checkpoint "$CKPT_DIR/best.pt" \
        --out-json "$EVAL_JSON" \
        --figure-path "/tmp/skip_${TAG}.png"

    echo "[error-bars] done FedProx tied seed=$SEED"
done

# --- Linear bag-of-DX ---
for SEED in $SEEDS; do
    TAG="linear_s${SEED}"
    EVAL_JSON="data/processed/eval_${TAG}.json"

    echo ""
    echo "================================================================"
    echo "[error-bars] Linear seed=$SEED"
    echo "================================================================"

    PYTHONPATH=src python -u src/baseline_linear.py \
        --seed "$SEED" \
        --out-json "$EVAL_JSON" \
        --figure-path "/tmp/skip_${TAG}.png"

    echo "[error-bars] done Linear seed=$SEED"
done

echo ""
echo "================================================================"
echo "[error-bars] SUMMARY"
echo "================================================================"

echo "FedProx tied:"
for SEED in $SEEDS; do
    EVAL_JSON="data/processed/eval_fedprox_tied_s${SEED}.json"
    if [ -f "$EVAL_JSON" ]; then
        python -c "
import json
d = json.load(open('$EVAL_JSON'))
f = d['foundation']
print(f'  seed=$SEED: overall={f[\"overall\"][\"auprc\"]:.4f}  new_onset={f[\"new_onset\"][\"auprc\"]:.4f}')
"
    else
        echo "  seed=$SEED: MISSING"
    fi
done

echo "Linear:"
for SEED in $SEEDS; do
    EVAL_JSON="data/processed/eval_linear_s${SEED}.json"
    if [ -f "$EVAL_JSON" ]; then
        python -c "
import json
d = json.load(open('$EVAL_JSON'))
f = d['baseline_linear']
print(f'  seed=$SEED: overall={f[\"overall\"][\"auprc\"]:.4f}  new_onset={f[\"new_onset\"][\"auprc\"]:.4f}')
"
    else
        echo "  seed=$SEED: MISSING"
    fi
done

echo "[error-bars] done"
