"""Compute calibration metrics (Brier score, expected calibration error) for
per-condition zero-shot predictions.

Reuses the scoring pipeline from evaluate_conditions.py:
  * builds eval instances on the held-out test set
  * runs the model once to get per-instance sigmoid probabilities at the cut <sep>
  * for each of the 5 acute conditions, extracts per-patient (y_true, y_score)
  * computes Brier, ECE, and a reliability diagram (bin-wise observed vs predicted)

Optionally supports ensemble mode: pass multiple --checkpoint paths; the script
averages sigmoid probabilities across all models before condition-level scoring.

Usage
-----
  # Single model calibration
  PYTHONPATH=src python src/compute_calibration.py \
      --checkpoint checkpoints/best.pt \
      --out-json data/processed/calib_centralized.json

  # Ensemble calibration (average probs across N checkpoints)
  PYTHONPATH=src python src/compute_calibration.py \
      --checkpoint checkpoints_untied_persite_s20260411/site_0/best.pt \
      --checkpoint checkpoints_untied_persite_s20260411/site_1/best.pt \
      ... \
      --out-json data/processed/calib_ensemble_s20260411.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from evaluate import EvalInstance, build_eval_instances
from evaluate_conditions import (
    CONDITION_CODES,
    resolve_codeset,
    condition_labels_and_scores,
    score_all_instances,
)
from model import build_model


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15,
) -> float:
    """Equal-width bin ECE. Returns mean absolute gap between predicted and observed."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_score >= bin_edges[i]) & (y_score <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_weight = mask.sum() / n
        predicted = y_score[mask].mean()
        observed = y_true[mask].mean()
        ece += bin_weight * abs(predicted - observed)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10,
) -> list[dict]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (y_score >= bin_edges[i]) & (y_score <= bin_edges[i + 1])
        if mask.sum() == 0:
            out.append({
                "bin_low": float(bin_edges[i]),
                "bin_high": float(bin_edges[i + 1]),
                "n": 0, "mean_predicted": None, "observed_rate": None,
            })
        else:
            out.append({
                "bin_low": float(bin_edges[i]),
                "bin_high": float(bin_edges[i + 1]),
                "n": int(mask.sum()),
                "mean_predicted": float(y_score[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            })
    return out


def run_model(ckpt_path: Path, instances, vocab: dict, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    V = len(vocab)
    model = build_model(V, **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    probs = score_all_instances(model, instances, device)
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="Checkpoint path. Repeat the flag for ensemble "
                             "(multiple checkpoints' probs are averaged).")
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.pkl"))
    parser.add_argument("--vocab-path", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--label", type=str, default=None,
                        help="Human-readable label for this calibration run.")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[calib] device={device}  n_checkpoints={len(args.checkpoint)}")

    with args.vocab_path.open() as f:
        vocab: dict[str, int] = json.load(f)
    dx_id_set = set(v for k, v in vocab.items() if k.startswith("DX"))

    with args.test_path.open("rb") as f:
        test_records = pickle.load(f)
    instances = build_eval_instances(test_records, dx_id_set)
    print(f"[calib] test records={len(test_records):,}  instances={len(instances):,}")

    # Accumulate probabilities across checkpoints (ensemble averaging)
    probs_sum = None
    for ckpt_path in args.checkpoint:
        print(f"[calib] scoring {ckpt_path}")
        probs = run_model(Path(ckpt_path), instances, vocab, device)
        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum = probs_sum + probs
    probs = probs_sum / len(args.checkpoint)

    # Compute calibration per condition
    results = []
    for name, prefixes in CONDITION_CODES.items():
        codeset_ids = resolve_codeset(vocab, prefixes)
        if not codeset_ids:
            continue
        y_true, y_score, _ = condition_labels_and_scores(instances, probs, codeset_ids)
        if len(y_true) == 0 or y_true.sum() == 0:
            continue

        # Clip scores to valid probability range (they are sums of sigmoids, may > 1)
        y_score_clip = np.clip(y_score, 0.0, 1.0)

        brier = brier_score(y_true, y_score_clip)
        ece = expected_calibration_error(y_true, y_score_clip, n_bins=15)
        bins = reliability_bins(y_true, y_score_clip, n_bins=10)

        # Calibration-in-the-large: mean predicted vs mean observed
        mean_pred = float(y_score_clip.mean())
        mean_obs = float(y_true.mean())

        results.append({
            "name": name,
            "n": int(len(y_true)),
            "n_positive": int(y_true.sum()),
            "prevalence": float(y_true.mean()),
            "brier": brier,
            "ece": ece,
            "mean_predicted": mean_pred,
            "mean_observed": mean_obs,
            "reliability_bins": bins,
        })
        print(
            f"[calib] {name:25s}  Brier={brier:.4f}  ECE={ece:.4f}  "
            f"mean_pred={mean_pred:.3f}  mean_obs={mean_obs:.3f}"
        )

    summary = {
        "checkpoints": [str(p) for p in args.checkpoint],
        "label": args.label or "",
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"[calib] wrote {args.out_json}")


if __name__ == "__main__":
    main()
