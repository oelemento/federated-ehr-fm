#!/bin/bash
#SBATCH --job-name=sweep-splits
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/sweep-splits-%j.out
#SBATCH --error=logs/sweep-splits-%j.err

# Sweep over 3 Dirichlet seeds × 4 alpha values = 12 configurations.
# Each: train 5 per-site models + ensemble eval.
# ~35 min per config on GPU → ~7h total.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[sweep] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

SEEDS="20260411 20260412 20260413"
ALPHAS="0.5 1.0 2.0 10.0"

for ALPHA in $ALPHAS; do
    for SEED in $SEEDS; do
        TAG="a${ALPHA}_s${SEED}"
        CKPT_DIR="checkpoints_sweep_${TAG}"
        EVAL_JSON="data/processed/eval_sweep_${TAG}.json"

        echo ""
        echo "================================================================"
        echo "[sweep] alpha=$ALPHA seed=$SEED tag=$TAG"
        echo "================================================================"

        # Train 5 per-site models
        PYTHONPATH=src python -u src/train_per_site.py \
            --config configs/poc.yaml \
            --n-sites 5 \
            --alpha "$ALPHA" \
            --num-epochs 3 \
            --seed "$SEED" \
            --checkpoint-dir "$CKPT_DIR"

        # Ensemble eval
        PYTHONPATH=src python -u src/evaluate_ensemble.py \
            --checkpoint-dir "$CKPT_DIR" \
            --n-sites 5 \
            --out-json "$EVAL_JSON"

        echo "[sweep] done: $TAG"
    done
done

echo ""
echo "================================================================"
echo "[sweep] ALL 12 CONFIGURATIONS COMPLETE"
echo "================================================================"

# Print summary
for ALPHA in $ALPHAS; do
    echo ""
    echo "--- alpha=$ALPHA ---"
    for SEED in $SEEDS; do
        TAG="a${ALPHA}_s${SEED}"
        EVAL_JSON="data/processed/eval_sweep_${TAG}.json"
        if [ -f "$EVAL_JSON" ]; then
            python -c "
import json, numpy as np
d = json.load(open('$EVAL_JSON'))
ens = d['ensemble']
print(f'  seed=$SEED: overall AUPRC={ens[\"overall\"][\"auprc\"]:.4f}  new_onset={ens[\"new_onset\"][\"auprc\"]:.4f}')
sites = d['per_site']
auprcs = [s['overall_auprc'] for s in sites]
print(f'    per-site mean={np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}')
if 'per_condition_auroc' in ens:
    aurocs = ens['per_condition_auroc']
    vals = [v['auroc'] for v in aurocs.values()]
    print(f'    condition AUROCs: {\" | \".join(f\"{k}={v[\"auroc\"]:.3f}\" for k,v in aurocs.items())}')
    print(f'    mean AUROC={np.mean(vals):.3f}')
"
        fi
    done
done

echo "[slurm] done"
