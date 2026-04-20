#!/bin/bash
#SBATCH --job-name=gpt-ehr-condeval
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/condeval-%j.out
#SBATCH --error=logs/condeval-%j.err

# Short sbatch wrapper for running evaluate_conditions.py on a GPU node.
# Submit from ~/EHR with:
#   sbatch --export=ALL,CKPT=checkpoints_abnormal_s1/best.pt,TAG=s1 configs/condeval.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs data/processed figures

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ehr
export PYTHONUNBUFFERED=1

CKPT="${CKPT:-checkpoints_abnormal_s1/best.pt}"
TAG="${TAG:-s1}"
OUT_JSON="data/processed/condition_eval_${TAG}.json"
OUT_FIG="figures/condition_eval_${TAG}.png"

echo "[slurm] job $SLURM_JOB_ID on $(hostname)"
echo "[slurm] checkpoint: $CKPT  tag: $TAG"
nvidia-smi || true

PYTHONPATH=src python -u src/evaluate_conditions.py \
    --checkpoint "$CKPT" \
    --test-path data/processed/test.pkl \
    --vocab-path data/processed/vocab.json \
    --out-json "$OUT_JSON" \
    --figure-path "$OUT_FIG"

echo "[slurm] done ($OUT_JSON)"
