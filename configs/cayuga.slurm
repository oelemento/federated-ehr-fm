#!/bin/bash
#SBATCH --job-name=gpt-ehr-poc
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/gpt-ehr-%j.out
#SBATCH --error=logs/gpt-ehr-%j.err

# Cayuga (Cornell CAC / WCM SCU) SLURM submit script for the GPT-EHR POC training run.
# Partitions available: scu-gpu (g0001-g0003, GPU) and scu-cpu (default).
# Submit from ~/EHR with: sbatch configs/cayuga.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs checkpoints figures

echo "[slurm] job $SLURM_JOB_ID on $(hostname)"
echo "[slurm] submit dir: $SLURM_SUBMIT_DIR"
nvidia-smi || true

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1   # otherwise print() block-buffers on the slurm .out file
PY="python -u"

echo "[slurm] python: $($PY --version)"
echo "[slurm] torch: $($PY -c 'import torch; print(torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"

# If processed data is not yet present, (re)tokenize. Requires MIMIC-IV at data/MIMIC-IV/.
if [[ ! -f data/processed/train.pkl ]]; then
    echo "[slurm] processed data missing; running cohort + tokenize"
    $PY src/cohort.py
    $PY src/tokenize_events.py
fi

# Optional per-job overrides (used by multi-seed sweeps):
#   sbatch --export=ALL,SEED=20260411,RUN_TAG=s1 configs/cayuga.slurm
SEED="${SEED:-}"
RUN_TAG="${RUN_TAG:-}"
TRAIN_EXTRA=""
CKPT_DIR="checkpoints"
if [[ -n "$SEED" ]]; then
    TRAIN_EXTRA="$TRAIN_EXTRA --seed $SEED"
fi
if [[ -n "$RUN_TAG" ]]; then
    CKPT_DIR="checkpoints_${RUN_TAG}"
    TRAIN_EXTRA="$TRAIN_EXTRA --checkpoint-dir $CKPT_DIR"
    mkdir -p "$CKPT_DIR"
    echo "[slurm] run tag: $RUN_TAG  ckpt dir: $CKPT_DIR  seed: ${SEED:-default}"
fi

# Train
PYTHONPATH=src $PY src/train.py --config configs/poc.yaml $TRAIN_EXTRA

# Per-run output paths so parallel/sequential seeds don't clobber each other.
EVAL_JSON="data/processed/eval_results${RUN_TAG:+_$RUN_TAG}.json"
EVAL_FIG="figures/eval_curves${RUN_TAG:+_$RUN_TAG}.png"

# Evaluate the best checkpoint
PYTHONPATH=src $PY src/evaluate.py \
    --checkpoint "$CKPT_DIR/best.pt" \
    --out-json "$EVAL_JSON" \
    --figure-path "$EVAL_FIG"
echo "[slurm] done ($EVAL_JSON)"
