#!/bin/bash
#SBATCH --job-name=sweep-fedper
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep-fedper-%j.out
#SBATCH --error=logs/sweep-fedper-%j.err

# Multi-seed FedPer / FedPer+LoRA sweep at alpha=0.5 to give error bars on
# the federated training-time methods (matching the per-site/ensemble bars
# from sweep_splits.slurm).
#
# Per seed, two training jobs:
#   1. FedPer (untied embeddings, 20 rounds, mu=0.1)
#   2. FedPer+LoRA (init from FedPer best.pt, 20 rounds, mu=0.01, rank 8)
# Then evaluate each with overall AUPRC and per-condition AUROC.
#
# Estimated runtime: ~110 min/seed * 3 seeds (both train phases) + eval
# overhead ~= 6-8 hours. 12h walltime gives a safety margin (sister sweep
# at 10h timed out on 12 configs, so we don't want to cut it close).
#
# Note: `set -e` is intentionally NOT used. We want individual seed
# failures (OOM, NaN, transient CUDA fault) to be loud but not abort
# the rest of the sweep — partial results are still useful.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[sweep-fedper] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

SEEDS="20260411 20260412 20260413"
ALPHA=0.5
N_SITES=5

for SEED in $SEEDS; do
    PER_TAG="fedper_a${ALPHA}_s${SEED}"
    PER_CKPT_DIR="checkpoints_${PER_TAG}"
    PER_EVAL_JSON="data/processed/eval_${PER_TAG}.json"
    PER_COND_JSON="data/processed/eval_cond_${PER_TAG}.json"

    LORA_TAG="fedper_lora_a${ALPHA}_s${SEED}"
    LORA_CKPT_DIR="checkpoints_${LORA_TAG}"
    LORA_EVAL_JSON="data/processed/eval_${LORA_TAG}.json"
    LORA_COND_JSON="data/processed/eval_cond_${LORA_TAG}.json"

    echo ""
    echo "================================================================"
    echo "[sweep-fedper] FedPer seed=$SEED tag=$PER_TAG"
    echo "================================================================"

    PYTHONPATH=src python -u src/train_federated.py \
        --config configs/poc.yaml \
        --n-sites "$N_SITES" \
        --n-rounds 20 \
        --local-epochs 1 \
        --local-steps 100 \
        --lr-decay 0.95 \
        --alpha "$ALPHA" \
        --strategy fedper \
        --mu 0.1 \
        --seed "$SEED" \
        --checkpoint-dir "$PER_CKPT_DIR"

    PYTHONPATH=src python -u src/evaluate.py \
        --checkpoint "$PER_CKPT_DIR/best.pt" \
        --out-json "$PER_EVAL_JSON" \
        --figure-path "/tmp/skip_${PER_TAG}.png"

    PYTHONPATH=src python -u src/evaluate_conditions.py \
        --checkpoint "$PER_CKPT_DIR/best.pt" \
        --out-json "$PER_COND_JSON" \
        --figure-path "/tmp/skip_cond_${PER_TAG}.png"

    echo ""
    echo "================================================================"
    echo "[sweep-fedper] FedPer+LoRA seed=$SEED tag=$LORA_TAG"
    echo "================================================================"

    PYTHONPATH=src python -u src/train_federated.py \
        --config configs/poc.yaml \
        --n-sites "$N_SITES" \
        --n-rounds 20 \
        --local-epochs 1 \
        --local-steps 100 \
        --lr-decay 0.95 \
        --alpha "$ALPHA" \
        --strategy fedper_lora \
        --mu 0.01 \
        --lora-rank 8 \
        --seed "$SEED" \
        --init-checkpoint "$PER_CKPT_DIR/best.pt" \
        --checkpoint-dir "$LORA_CKPT_DIR"

    PYTHONPATH=src python -u src/evaluate.py \
        --checkpoint "$LORA_CKPT_DIR/best.pt" \
        --out-json "$LORA_EVAL_JSON" \
        --figure-path "/tmp/skip_${LORA_TAG}.png"

    PYTHONPATH=src python -u src/evaluate_conditions.py \
        --checkpoint "$LORA_CKPT_DIR/best.pt" \
        --out-json "$LORA_COND_JSON" \
        --figure-path "/tmp/skip_cond_${LORA_TAG}.png"

    echo "[sweep-fedper] done seed=$SEED"
done

echo ""
echo "================================================================"
echo "[sweep-fedper] ALL SEEDS COMPLETE"
echo "================================================================"

# Summary
for SEED in $SEEDS; do
    for METH in fedper fedper_lora; do
        TAG="${METH}_a${ALPHA}_s${SEED}"
        EVAL_JSON="data/processed/eval_${TAG}.json"
        COND_JSON="data/processed/eval_cond_${TAG}.json"
        if [ -f "$EVAL_JSON" ]; then
            python -c "
import json
d = json.load(open('$EVAL_JSON'))
f = d['foundation']
print(f'  $METH seed=$SEED: overall AUPRC={f[\"overall\"][\"auprc\"]:.4f}  new_onset={f[\"new_onset\"][\"auprc\"]:.4f}')
"
        else
            echo "  $METH seed=$SEED: MISSING $EVAL_JSON"
        fi
        if [ -f "$COND_JSON" ]; then
            python -c "
import json
d = json.load(open('$COND_JSON'))
print(f'    per-condition AUROC: ' + ' | '.join(f'{r[\"name\"]}={r[\"auroc\"]:.3f}' for r in d['results']))
"
        else
            echo "    $METH seed=$SEED: MISSING $COND_JSON"
        fi
    done
done

echo "[sweep-fedper] done"
