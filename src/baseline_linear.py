"""Linear multi-label baseline for next-admission diagnosis forecasting.

This is the "what if you just use a linear model on bag-of-DX history" baseline.
It sits between the trivial baselines (marginal, repeat) and the foundation model,
and answers the question a reviewer will always ask: "could you have done this with
XGBoost instead of a Transformer?"

Architecture
------------
For each patient with >=2 admissions:
    * Features x  : length-K binary vector (or counts) of DX tokens present in
                    visits 1..N-1 where K = |DX sub-vocab|. Optional 2K mode adds
                    counts of PX + MED tokens from history as extra features.
    * Target y    : length-K binary multi-label indicator of DX tokens in visit N
                    (the held-out admission).
    * Model       : single Linear(K, K) layer trained with BCE-with-logits.

The same micro-AUPRC and new-onset masking used in `evaluate.py` is applied, so
the numbers are directly comparable to the foundation model's PR curves.

Usage
-----
    python3.11 src/baseline_linear.py \
        --train-path data/processed/train.pkl \
        --test-path data/processed/test.pkl \
        --vocab-path data/processed/vocab.json \
        --out-json data/processed/baseline_linear_results.json \
        --figure-path figures/baseline_linear_pr.png

By default trains for 10 epochs with AdamW on CPU.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve

SEP_ID = 2


def build_examples(records: list[dict], dx_id_set: set[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_hist, Y_target, H_hist) arrays where:

    X_hist  : (N, K) float32 — bag-of-DX feature of visits 1..N-1 for each patient (counts).
    Y_target: (N, K) uint8   — multi-label indicator of DX tokens in visit N.
    H_hist  : (N, K) uint8   — binary mask of DX tokens already seen in visits 1..N-1
                              (used for new-onset evaluation, identical to evaluate.py).
    """
    dx_ids_sorted = np.array(sorted(dx_id_set), dtype=np.int64)
    dx_id_to_col = {int(t): c for c, t in enumerate(dx_ids_sorted)}
    K = len(dx_ids_sorted)

    X_rows: list[np.ndarray] = []
    Y_rows: list[np.ndarray] = []
    H_rows: list[np.ndarray] = []

    for rec in records:
        seps = rec["sep_positions"]
        if len(seps) < 2:
            continue
        cut_pos = int(seps[-2])
        tokens = rec["token_ids"]
        blocks = rec["block_ids"]
        last_block_id = int(blocks[seps[-1]])

        hist_mask = np.arange(len(tokens)) <= cut_pos
        target_mask = (blocks == last_block_id) & (tokens != SEP_ID)

        hist_toks = tokens[hist_mask]
        target_toks = tokens[target_mask]

        # Features and history mask
        x = np.zeros(K, dtype=np.float32)
        h = np.zeros(K, dtype=np.uint8)
        for t in hist_toks:
            t_int = int(t)
            if t_int in dx_id_to_col:
                c = dx_id_to_col[t_int]
                x[c] += 1.0
                h[c] = 1

        # Target
        y = np.zeros(K, dtype=np.uint8)
        any_target = False
        for t in target_toks:
            t_int = int(t)
            if t_int in dx_id_to_col:
                y[dx_id_to_col[t_int]] = 1
                any_target = True
        if not any_target:
            continue

        X_rows.append(x)
        Y_rows.append(y)
        H_rows.append(h)

    return np.stack(X_rows), np.stack(Y_rows), np.stack(H_rows)


def compute_metrics(probs: np.ndarray, targets: np.ndarray, history: np.ndarray | None) -> dict:
    """Same micro-AUPRC + best-F1 as `evaluate.compute_metrics`, so baselines are comparable."""
    p, t = probs, targets
    if history is not None:
        keep = history == 0
        p = p[keep]
        t = t[keep]
    else:
        p = p.reshape(-1)
        t = t.reshape(-1)
    if t.sum() == 0:
        return {"auprc": 0.0, "best_f1": 0.0, "n_pos": 0, "n_total": int(len(t))}
    auprc = float(average_precision_score(t, p))
    prec, rec, _ = precision_recall_curve(t, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    return {
        "auprc": auprc,
        "best_f1": float(np.nanmax(f1)),
        "n_pos": int(t.sum()),
        "n_total": int(len(t)),
        "prec": prec.tolist(),
        "rec": rec.tolist(),
    }


def train_linear(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, Y_val: np.ndarray,
    n_epochs: int = 10, lr: float = 1e-3, weight_decay: float = 1e-4,
    batch_size: int = 512, seed: int = 20260410,
) -> nn.Linear:
    torch.manual_seed(seed)
    np.random.seed(seed)
    K = X_train.shape[1]
    model = nn.Linear(K, K)

    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train.astype(np.float32))
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val.astype(np.float32))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    N = X_train_t.shape[0]
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            xb = X_train_t[idx]
            yb = Y_train_t[idx]
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
        train_loss = total_loss / N

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = F.binary_cross_entropy_with_logits(val_logits, Y_val_t).item()
        print(f"  [baseline] epoch {epoch} train={train_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.pkl"))
    parser.add_argument("--val-path", type=Path, default=Path("data/processed/val.pkl"))
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.pkl"))
    parser.add_argument("--vocab-path", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--out-json", type=Path, default=Path("data/processed/baseline_linear_results.json"))
    parser.add_argument("--figure-path", type=Path, default=Path("figures/baseline_linear_pr.png"))
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260410)
    args = parser.parse_args()

    print("[baseline] loading vocab")
    vocab = json.loads(args.vocab_path.read_text())
    dx_ids = [i for tok, i in vocab.items() if tok.startswith("DX")]
    dx_id_set = set(int(i) for i in dx_ids)
    K = len(dx_id_set)
    print(f"[baseline] |DX|={K}  vocab={len(vocab)}")

    print("[baseline] loading splits")
    with args.train_path.open("rb") as f:
        train_records = pickle.load(f)
    with args.val_path.open("rb") as f:
        val_records = pickle.load(f)
    with args.test_path.open("rb") as f:
        test_records = pickle.load(f)

    t0 = time.time()
    X_train, Y_train, _ = build_examples(train_records, dx_id_set)
    X_val, Y_val, _ = build_examples(val_records, dx_id_set)
    X_test, Y_test, H_test = build_examples(test_records, dx_id_set)
    print(
        f"[baseline] built features in {time.time()-t0:.1f}s  "
        f"train={X_train.shape} val={X_val.shape} test={X_test.shape}"
    )

    t0 = time.time()
    model = train_linear(
        X_train, Y_train, X_val, Y_val,
        n_epochs=args.n_epochs, lr=args.lr, seed=args.seed,
    )
    print(f"[baseline] trained in {time.time()-t0:.1f}s")

    # Score test set
    model.eval()
    with torch.no_grad():
        test_logits = model(torch.from_numpy(X_test)).numpy()
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))  # sigmoid

    overall = compute_metrics(test_probs, Y_test, history=None)
    new_onset = compute_metrics(test_probs, Y_test, history=H_test)
    print(
        f"[baseline] overall AUPRC={overall['auprc']:.4f} F1={overall['best_f1']:.4f}  "
        f"new-onset AUPRC={new_onset['auprc']:.4f} F1={new_onset['best_f1']:.4f}"
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    light = {
        "baseline_linear": {
            "overall": {k: v for k, v in overall.items() if k not in ("prec", "rec")},
            "new_onset": {k: v for k, v in new_onset.items() if k not in ("prec", "rec")},
        }
    }
    args.out_json.write_text(json.dumps(light, indent=2))
    print(f"[baseline] wrote {args.out_json}")

    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, res, title in zip(axes, [overall, new_onset], ["Overall diagnoses", "New-onset diagnoses"]):
        ax.plot(res["rec"], res["prec"], label=f"linear (AP={res['auprc']:.3f})", color="C2")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Linear bag-of-DX baseline — next-admission DX forecasting", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.figure_path, dpi=150)
    print(f"[baseline] wrote {args.figure_path}")


if __name__ == "__main__":
    main()
