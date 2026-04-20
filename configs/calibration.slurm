#!/bin/bash
#SBATCH --job-name=calibration
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/calibration-%j.out
#SBATCH --error=logs/calibration-%j.err

# Compute per-condition calibration (Brier score, ECE, reliability bins)
# for three strategies: centralized, federated (untied FedAvg), and ensemble.

set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

echo "[calib] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

# Seeds to run (one per method)
CENTRALIZED_SEEDS="s0 s1 s2"
FED_SEEDS="20260411 20260412 20260413"
ENS_SEEDS="20260411 20260412 20260413"

# --- Centralized untied ---
for SEED in 20260410 20260411 20260412; do
    CKPT="checkpoints_untied_centralized_s${SEED}/best.pt"
    OUT="data/processed/calib_centralized_untied_s${SEED}.json"
    if [ -f "$CKPT" ]; then
        echo "[calib] centralized untied seed=$SEED"
        PYTHONPATH=src python -u src/compute_calibration.py \
            --checkpoint "$CKPT" \
            --out-json "$OUT" \
            --label "centralized_untied_s${SEED}"
    fi
done

# --- Federated untied (FedAvg with untied embeddings) ---
for SEED in $FED_SEEDS; do
    CKPT="checkpoints_untied_fedavg_a0.5_s${SEED}/best.pt"
    OUT="data/processed/calib_untied_fedavg_s${SEED}.json"
    if [ -f "$CKPT" ]; then
        echo "[calib] untied FedAvg seed=$SEED"
        PYTHONPATH=src python -u src/compute_calibration.py \
            --checkpoint "$CKPT" \
            --out-json "$OUT" \
            --label "untied_fedavg_s${SEED}"
    fi
done

# --- Ensemble of untied per-site models (average predictions across 5 sites) ---
for SEED in $ENS_SEEDS; do
    OUT="data/processed/calib_untied_ensemble_s${SEED}.json"
    CKPTS=""
    for SITE in 0 1 2 3 4; do
        CKPTS="$CKPTS --checkpoint checkpoints_untied_persite_s${SEED}/site_${SITE}/best.pt"
    done
    # Only run if all 5 checkpoints exist
    MISSING=0
    for SITE in 0 1 2 3 4; do
        [ -f "checkpoints_untied_persite_s${SEED}/site_${SITE}/best.pt" ] || MISSING=1
    done
    if [ $MISSING -eq 0 ]; then
        echo "[calib] untied ensemble seed=$SEED (5 checkpoints)"
        PYTHONPATH=src python -u src/compute_calibration.py \
            $CKPTS --out-json "$OUT" \
            --label "untied_ensemble_s${SEED}"
    fi
done

echo ""
echo "[calib] SUMMARY"
ls -la data/processed/calib_*.json 2>&1 | tail -20
echo "[calib] done"
