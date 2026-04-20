#!/bin/bash
#SBATCH --job-name=central-untied
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/central-untied-%j.out
#SBATCH --error=logs/central-untied-%j.err

# Control: centralized training with UNTIED embeddings (tie_embeddings=False).
# Separates "untying helps federation" from "untying helps the model in general."
# 3 seeds × ~17 min each + eval = ~1h total.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[central-untied] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

# Create a temporary config with tie_embeddings: false
UNTIED_CFG="configs/_poc_untied.yaml"
python -c "
import yaml
cfg = yaml.safe_load(open('configs/poc.yaml'))
cfg['model']['tie_embeddings'] = False
with open('$UNTIED_CFG', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('wrote $UNTIED_CFG with tie_embeddings:', cfg['model']['tie_embeddings'])
"

SEEDS="20260410 20260411 20260412"

for SEED in $SEEDS; do
    CKPT_DIR="checkpoints_untied_centralized_s${SEED}"
    EVAL_JSON="data/processed/eval_untied_centralized_s${SEED}.json"
    COND_JSON="data/processed/eval_cond_untied_centralized_s${SEED}.json"

    echo ""
    echo "================================================================"
    echo "[central-untied] seed=$SEED"
    echo "================================================================"

    PYTHONPATH=src python -u src/train.py \
        --config "$UNTIED_CFG" \
        --seed "$SEED" \
        --checkpoint-dir "$CKPT_DIR"

    PYTHONPATH=src python -u src/evaluate.py \
        --checkpoint "$CKPT_DIR/best.pt" \
        --out-json "$EVAL_JSON" \
        --figure-path "/tmp/skip_untied_central_${SEED}.png"

    PYTHONPATH=src python -u src/evaluate_conditions.py \
        --checkpoint "$CKPT_DIR/best.pt" \
        --out-json "$COND_JSON" \
        --figure-path "/tmp/skip_cond_untied_central_${SEED}.png"

    echo "[central-untied] done seed=$SEED"
done

echo ""
echo "================================================================"
echo "[central-untied] SUMMARY"
echo "================================================================"

for SEED in $SEEDS; do
    EVAL_JSON="data/processed/eval_untied_centralized_s${SEED}.json"
    if [ -f "$EVAL_JSON" ]; then
        python -c "
import json
d = json.load(open('$EVAL_JSON'))
f = d['foundation']
print(f'  seed=$SEED: overall AUPRC={f[\"overall\"][\"auprc\"]:.4f}  new_onset={f[\"new_onset\"][\"auprc\"]:.4f}')
"
    else
        echo "  seed=$SEED: MISSING"
    fi
done

echo "[central-untied] done"
