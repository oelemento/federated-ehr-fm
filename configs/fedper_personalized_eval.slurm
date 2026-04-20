#!/bin/bash
#SBATCH --job-name=fedper-personalized
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/fedper-personalized-%j.out
#SBATCH --error=logs/fedper-personalized-%j.err
#SBATCH --dependency=afterany:2807256

# Rerun FedPer / FedPer+LoRA evaluation using the PERSONALIZED procedure:
# restore each site's saved output_proj and report (a) global-head AUPRC
# (reproduces evaluate.py for consistency check), (b) per-site mean
# personalized AUPRC, (c) head-ensemble AUPRC (average predictions across
# site-personalized heads with the shared backbone).
#
# Per-condition AUROCs are computed on the head-ensemble predictions, with
# bootstrap CIs matching evaluate_conditions.py (seed=20260411).
#
# Dependency: runs after sweep 2806532 completes (all 6 checkpoints present).
# Estimated runtime: 6 checkpoints × ~5 min each ≈ 35 min.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[fedper-personalized] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

SEEDS="20260411 20260412 20260413"
ALPHA=0.5

for SEED in $SEEDS; do
    for STRATEGY in fedper fedper_lora; do
        TAG="${STRATEGY}_a${ALPHA}_s${SEED}"
        CKPT="checkpoints_${TAG}/best.pt"
        OUT_JSON="data/processed/eval_${STRATEGY}_personalized_a${ALPHA}_s${SEED}.json"
        COND_JSON="data/processed/eval_cond_${STRATEGY}_personalized_a${ALPHA}_s${SEED}.json"

        if [ ! -f "$CKPT" ]; then
            echo "[fedper-personalized] SKIP $TAG: missing $CKPT"
            continue
        fi
        if [ -f "$OUT_JSON" ] && [ -f "$COND_JSON" ]; then
            echo "[fedper-personalized] SKIP $TAG: outputs already exist"
            continue
        fi

        echo ""
        echo "================================================================"
        echo "[fedper-personalized] $TAG"
        echo "================================================================"

        PYTHONPATH=src python -u src/evaluate_fedper_personalized.py \
            --checkpoint "$CKPT" \
            --out-json "$OUT_JSON" \
            --cond-out-json "$COND_JSON" \
            || echo "[fedper-personalized] FAIL $TAG"
    done
done

echo ""
echo "================================================================"
echo "[fedper-personalized] ALL CHECKPOINTS PROCESSED"
echo "================================================================"

# Summary: head-ensemble numbers per checkpoint
for SEED in $SEEDS; do
    for STRATEGY in fedper fedper_lora; do
        TAG="${STRATEGY}_a${ALPHA}_s${SEED}"
        OUT_JSON="data/processed/eval_${STRATEGY}_personalized_a${ALPHA}_s${SEED}.json"
        if [ -f "$OUT_JSON" ]; then
            python -c "
import json
d = json.load(open('$OUT_JSON'))
g  = d['global_head']
he = d['head_ensemble']
pm = d['per_site_mean_personalized']
print(f'  $TAG:')
print(f'    global-head     overall={g[\"overall\"][\"auprc\"]:.4f}  new_onset={g[\"new_onset\"][\"auprc\"]:.4f}')
print(f'    head-ensemble   overall={he[\"overall\"][\"auprc\"]:.4f}  new_onset={he[\"new_onset\"][\"auprc\"]:.4f}')
print(f'    per-site-mean   overall={pm[\"overall_auprc_mean\"]:.4f} +/- {pm[\"overall_auprc_std\"]:.4f}  new_onset={pm[\"new_onset_auprc_mean\"]:.4f} +/- {pm[\"new_onset_auprc_std\"]:.4f}')
"
        else
            echo "  $TAG: MISSING $OUT_JSON"
        fi
    done
done

echo "[fedper-personalized] done"
