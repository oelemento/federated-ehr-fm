#!/bin/bash
#SBATCH --job-name=untied-ensemble
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/untied-ensemble-%j.out
#SBATCH --error=logs/untied-ensemble-%j.err

# Critical control: per-site models with UNTIED embeddings + ensemble.
# Tests whether untied FedAvg's advantage over ensemble is from shared
# training (FedAvg averages backbone across sites) or just from untying.
#
# 3 seeds × 5 per-site models × 3 epochs ≈ 30 min/seed + ensemble eval.
# Total ≈ 2h.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[untied-ensemble] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

# Create config with tie_embeddings: false
UNTIED_CFG="configs/_poc_untied.yaml"
python -c "
import yaml
cfg = yaml.safe_load(open('configs/poc.yaml'))
cfg['model']['tie_embeddings'] = False
with open('$UNTIED_CFG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('wrote $UNTIED_CFG with tie_embeddings:', cfg['model']['tie_embeddings'])
"

SEEDS="20260411 20260412 20260413"

for SEED in $SEEDS; do
    CKPT_DIR="checkpoints_untied_persite_s${SEED}"
    EVAL_JSON="data/processed/eval_untied_ensemble_s${SEED}.json"

    echo ""
    echo "================================================================"
    echo "[untied-ensemble] seed=$SEED"
    echo "================================================================"

    # Train 5 per-site models with untied embeddings
    PYTHONPATH=src python -u src/train_per_site.py \
        --config "$UNTIED_CFG" \
        --n-sites 5 \
        --alpha 0.5 \
        --num-epochs 3 \
        --seed "$SEED" \
        --checkpoint-dir "$CKPT_DIR"

    # Ensemble eval
    PYTHONPATH=src python -u src/evaluate_ensemble.py \
        --checkpoint-dir "$CKPT_DIR" \
        --n-sites 5 \
        --out-json "$EVAL_JSON"

    echo "[untied-ensemble] done seed=$SEED"
done

echo ""
echo "================================================================"
echo "[untied-ensemble] SUMMARY"
echo "================================================================"

for SEED in $SEEDS; do
    EVAL_JSON="data/processed/eval_untied_ensemble_s${SEED}.json"
    if [ -f "$EVAL_JSON" ]; then
        python -c "
import json, numpy as np
d = json.load(open('$EVAL_JSON'))
ens = d['ensemble']
print(f'  seed=$SEED: ensemble overall={ens[\"overall\"][\"auprc\"]:.4f}  new_onset={ens[\"new_onset\"][\"auprc\"]:.4f}')
sites = d['per_site']
auprcs = [s['overall_auprc'] for s in sites]
print(f'    per-site mean={np.mean(auprcs):.4f} +/- {np.std(auprcs):.4f}')
"
    else
        echo "  seed=$SEED: MISSING"
    fi
done

echo "[untied-ensemble] done"
