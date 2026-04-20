#!/bin/bash
#SBATCH --job-name=gpt-ehr-fed
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/federated-%j.out
#SBATCH --error=logs/federated-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs figures

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1
PY="python -u"

echo "[slurm] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || true

STRATEGY="${STRATEGY:-fedprox}"
ALPHA="${ALPHA:-0.5}"
N_SITES="${N_SITES:-5}"
N_ROUNDS="${N_ROUNDS:-30}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LOCAL_STEPS="${LOCAL_STEPS:-100}"
LR_DECAY="${LR_DECAY:-0.95}"
MU="${MU:-0.1}"
SEED="${SEED:-20260411}"
INIT_CKPT="${INIT_CKPT:-}"
TAG="${TAG:-fed_${STRATEGY}_a${ALPHA}_v3}"
CKPT_DIR="checkpoints_${TAG}"

echo "[slurm] strategy=$STRATEGY alpha=$ALPHA n_sites=$N_SITES n_rounds=$N_ROUNDS local_epochs=$LOCAL_EPOCHS local_steps=$LOCAL_STEPS lr_decay=$LR_DECAY mu=$MU seed=$SEED init=$INIT_CKPT tag=$TAG"

INIT_FLAG=""
if [[ -n "$INIT_CKPT" && -f "$INIT_CKPT" ]]; then
    INIT_FLAG="--init-checkpoint $INIT_CKPT"
fi

PYTHONPATH=src $PY src/train_federated.py \
    --config configs/poc.yaml \
    --n-sites "$N_SITES" \
    --n-rounds "$N_ROUNDS" \
    --local-epochs "$LOCAL_EPOCHS" \
    --local-steps "$LOCAL_STEPS" \
    --lr-decay "$LR_DECAY" \
    --alpha "$ALPHA" \
    --strategy "$STRATEGY" \
    --mu "$MU" \
    --seed "$SEED" \
    --checkpoint-dir "$CKPT_DIR" \
    $INIT_FLAG

PYTHONPATH=src $PY src/evaluate.py \
    --checkpoint "$CKPT_DIR/best.pt" \
    --out-json "data/processed/eval_results_${TAG}.json" \
    --figure-path "figures/eval_curves_${TAG}.png"

echo "[slurm] done"
