"""Evaluate an ensemble of per-site models.

Instead of averaging weights (which destroys site-specific features), this
loads all N site models, runs inference on each, averages their sigmoid
probabilities, and computes AUPRC on the averaged predictions.

This is the "free win" from the FL literature: each site model is specialized;
the ensemble captures diversity without destructive weight averaging.

Usage
-----
    PYTHONPATH=src python -u src/evaluate_ensemble.py \
        --checkpoint-dir checkpoints_per_site \
        --n-sites 5
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from sklearn.metrics import roc_auc_score

from evaluate import (
    build_eval_instances,
    gather_model_probs,
    compute_metrics,
)
from model import build_model

# Same condition codesets as evaluate_conditions.py
CONDITION_CODES: dict[str, list[str]] = {
    "aki":                 ["DX10:N17", "DX9:584"],
    "sepsis":              ["DX10:A40", "DX10:A41", "DX9:038"],
    "acute_resp_failure":  ["DX10:J96"],
    "heart_failure":       ["DX10:I50", "DX9:428"],
    "acute_mi":            ["DX10:I21", "DX9:410"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_per_site"))
    parser.add_argument("--n-sites", type=int, default=5)
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.pkl"))
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.pkl"))
    parser.add_argument("--vocab-path", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--out-json", type=Path, default=Path("data/processed/eval_results_ensemble.json"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with args.vocab_path.open() as f:
        vocab: dict[str, int] = json.load(f)
    V = len(vocab)
    dx_ids_sorted = np.array(
        sorted(i for tok, i in vocab.items() if tok.startswith("DX")),
        dtype=np.int64,
    )
    dx_id_set = set(int(i) for i in dx_ids_sorted)
    K = len(dx_ids_sorted)
    print(f"[ensemble] vocab={V}  |DX|={K}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    with args.test_path.open("rb") as f:
        test_records = pickle.load(f)
    instances = build_eval_instances(test_records, dx_id_set)
    N = len(instances)
    print(f"[ensemble] {N:,} eval instances")

    # Build targets and history matrices
    dx_id_to_col = {int(i): c for c, i in enumerate(dx_ids_sorted)}
    targets = np.zeros((N, K), dtype=np.uint8)
    history = np.zeros((N, K), dtype=np.uint8)
    for i, inst in enumerate(instances):
        for t in inst.target_dx_ids:
            targets[i, dx_id_to_col[t]] = 1
        for t in inst.history_tokens:
            history[i, dx_id_to_col[t]] = 1

    # Gather probabilities from each site model and average
    ensemble_probs = np.zeros((N, K), dtype=np.float64)
    site_results = []

    for s in range(args.n_sites):
        ckpt_path = args.checkpoint_dir / f"site_{s}" / "best.pt"
        print(f"[ensemble] loading site {s} from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        model = build_model(V, **cfg["model"]).to(device)
        model.load_state_dict(ckpt["model_state"])

        probs = gather_model_probs(model, instances, dx_ids_sorted, device, args.batch_size)

        # Per-site metrics
        site_overall = compute_metrics(probs, targets, history=None)
        site_new = compute_metrics(probs, targets, history=history)
        print(
            f"  site {s} ({ckpt.get('n_patients', '?')} pts): "
            f"overall AUPRC={site_overall['auprc']:.4f}  "
            f"new-onset AUPRC={site_new['auprc']:.4f}"
        )
        site_results.append({
            "site": s,
            "n_patients": ckpt.get("n_patients"),
            "overall_auprc": site_overall["auprc"],
            "new_onset_auprc": site_new["auprc"],
        })

        ensemble_probs += probs
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Average probabilities
    ensemble_probs /= args.n_sites

    # Ensemble metrics
    overall = compute_metrics(ensemble_probs.astype(np.float32), targets, history=None)
    new_onset = compute_metrics(ensemble_probs.astype(np.float32), targets, history=history)

    print(f"\n[ensemble] === ENSEMBLE OF {args.n_sites} SITE MODELS ===")
    print(
        f"  overall AUPRC={overall['auprc']:.4f} F1={overall['best_f1']:.4f}  "
        f"new-onset AUPRC={new_onset['auprc']:.4f} F1={new_onset['best_f1']:.4f}"
    )

    # Marginal baseline for reference
    with args.train_path.open("rb") as f:
        train_records = pickle.load(f)
    from evaluate import marginal_frequencies
    marg_full = marginal_frequencies(train_records, dx_id_set, V)
    marg_probs = np.tile(marg_full[dx_ids_sorted], (N, 1)).astype(np.float32)
    marg_overall = compute_metrics(marg_probs, targets, history=None)
    marg_new = compute_metrics(marg_probs, targets, history=history)
    print(
        f"  marginal: overall AUPRC={marg_overall['auprc']:.4f}  "
        f"new-onset AUPRC={marg_new['auprc']:.4f}"
    )

    # Per-condition AUROC on ensemble predictions
    print(f"\n[ensemble] === PER-CONDITION AUROC ===")
    condition_results = {}
    for cond_name, prefixes in CONDITION_CODES.items():
        cond_ids = sorted(
            i for tok, i in vocab.items()
            if any(tok.startswith(p) for p in prefixes)
        )
        if not cond_ids:
            continue
        cond_cols = [dx_id_to_col[i] for i in cond_ids if i in dx_id_to_col]
        if not cond_cols:
            continue
        # New-onset: condition absent in history, present in target
        cond_target = targets[:, cond_cols].max(axis=1)
        cond_hist = history[:, cond_cols].max(axis=1)
        eligible = cond_hist == 0
        y_true = cond_target[eligible]
        y_score = ensemble_probs[eligible][:, cond_cols].sum(axis=1).astype(np.float32)
        if y_true.sum() == 0:
            continue
        auroc = float(roc_auc_score(y_true, y_score))
        n_pos = int(y_true.sum())
        print(f"  {cond_name:20s}: AUROC={auroc:.3f}  n_pos={n_pos}")
        condition_results[cond_name] = {"auroc": auroc, "n_positive": n_pos}

    print(f"\n[ensemble] === COMPARISON ===")
    site_auprcs = [r["overall_auprc"] for r in site_results]
    site_new_auprcs = [r["new_onset_auprc"] for r in site_results]
    print(f"  Per-site mean:     overall {np.mean(site_auprcs):.4f}  new-onset {np.mean(site_new_auprcs):.4f}")
    print(f"  Ensemble:          overall {overall['auprc']:.4f}  new-onset {new_onset['auprc']:.4f}")
    print(f"  FedPer+LoRA:       overall 0.2419  new-onset 0.0782")
    print(f"  Centralized:       overall 0.3607  new-onset 0.1675")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "ensemble": {
            "overall": {k: v for k, v in overall.items() if k not in ("prec", "rec")},
            "new_onset": {k: v for k, v in new_onset.items() if k not in ("prec", "rec")},
            "per_condition_auroc": condition_results,
        },
        "per_site": site_results,
    }
    args.out_json.write_text(json.dumps(result, indent=2))
    print(f"[ensemble] wrote {args.out_json}")


if __name__ == "__main__":
    main()
