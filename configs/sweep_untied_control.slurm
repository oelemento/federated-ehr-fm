#!/bin/bash
#SBATCH --job-name=untied-control
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/untied-control-%j.out
#SBATCH --error=logs/untied-control-%j.err

# Control experiment: untied FedAvg and untied FedProx WITHOUT personalization.
# This isolates the tying mechanism: if untied FedAvg/FedProx also fails,
# the problem is averaging the output projection (even when separate from
# the input embedding). If it succeeds, the problem is specifically the
# dual role of the tied matrix.
#
# 2 strategies × 3 seeds = 6 configs.
# Each: 20 rounds × ~3 min/round ≈ 60 min.
# Total ≈ 6h + eval overhead.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[untied-control] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

SEEDS="20260411 20260412 20260413"
ALPHA=0.5
N_SITES=5

for STRATEGY in fedavg fedprox; do
    for SEED in $SEEDS; do
        TAG="untied_${STRATEGY}_a${ALPHA}_s${SEED}"
        CKPT_DIR="checkpoints_${TAG}"
        EVAL_JSON="data/processed/eval_${TAG}.json"
        COND_JSON="data/processed/eval_cond_${TAG}.json"

        MU=0.0
        if [ "$STRATEGY" = "fedprox" ]; then MU=0.1; fi

        echo ""
        echo "================================================================"
        echo "[untied-control] $TAG (mu=$MU)"
        echo "================================================================"

        PYTHONPATH=src python -u src/train_federated.py \
            --config configs/poc.yaml \
            --n-sites "$N_SITES" \
            --n-rounds 20 \
            --local-epochs 1 \
            --local-steps 100 \
            --lr-decay 0.95 \
            --alpha "$ALPHA" \
            --strategy "$STRATEGY" \
            --mu "$MU" \
            --seed "$SEED" \
            --untie-embeddings \
            --checkpoint-dir "$CKPT_DIR"

        PYTHONPATH=src python -u src/evaluate.py \
            --checkpoint "$CKPT_DIR/best.pt" \
            --out-json "$EVAL_JSON" \
            --figure-path "/tmp/skip_${TAG}.png"

        PYTHONPATH=src python -u src/evaluate_conditions.py \
            --checkpoint "$CKPT_DIR/best.pt" \
            --out-json "$COND_JSON" \
            --figure-path "/tmp/skip_cond_${TAG}.png"

        echo "[untied-control] done $TAG"
    done
done

echo ""
echo "================================================================"
echo "[untied-control] ALL CONFIGS COMPLETE"
echo "================================================================"

for STRATEGY in fedavg fedprox; do
    for SEED in $SEEDS; do
        TAG="untied_${STRATEGY}_a${ALPHA}_s${SEED}"
        EVAL_JSON="data/processed/eval_${TAG}.json"
        if [ -f "$EVAL_JSON" ]; then
            python -c "
import json
d = json.load(open('$EVAL_JSON'))
f = d['foundation']
print(f'  $TAG: overall AUPRC={f[\"overall\"][\"auprc\"]:.4f}  new_onset={f[\"new_onset\"][\"auprc\"]:.4f}')
"
        else
            echo "  $TAG: MISSING $EVAL_JSON"
        fi
    done
done

echo "[untied-control] done"
