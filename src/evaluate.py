"""Zero-shot evaluation of the POC GPT-EHR model.

For each test patient with >=2 admissions, cut the sequence at the <sep>
closing admission N-1, run the model, take sigmoid probabilities at that
<sep> position, and score multi-label prediction against the diagnosis
set actually observed in admission N.

Metrics (diagnosis sub-vocabulary only):
- Overall micro-AUPRC and best-F1 (all diagnoses)
- New-onset micro-AUPRC and best-F1 (diagnoses absent from visits 1..N-1)

Two baselines run side by side:
- marginal : predict the training-set global prevalence for every patient
- repeat   : predict exactly the diagnoses observed in admission N-1

Outputs:
- prints a summary table
- writes data/processed/eval_results.json with raw metric values
- writes figures/eval_curves.png with the 2-panel PR figure
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from dataset import SEP_ID
from model import GPTEHR, build_model


@dataclass
class EvalInstance:
    token_ids: np.ndarray
    block_ids: np.ndarray
    position_ids: np.ndarray
    cut_sep_pos: int             # position of <sep> closing visit N-1 (where we predict)
    history_tokens: set[int]     # union over visits 1..N-1 (for new-onset filter)
    target_dx_ids: set[int]      # diagnosis tokens in the held-out visit N
    prior_dx_ids: set[int]       # diagnosis tokens in the most recent prior visit (N-1) -- for repeat baseline


def build_eval_instances(records: list[dict], dx_token_ids: set[int]) -> list[EvalInstance]:
    """Convert tokenized patient records into (history_through_N-1, target_block_N) pairs.

    The input records already encode the full sequence through <sep>_N. We cut at <sep>_{N-1}
    and treat the last block as the held-out target. This matches the design: one evaluation
    instance per test patient.
    """
    out: list[EvalInstance] = []
    for rec in records:
        seps = rec["sep_positions"]
        if len(seps) < 2:
            continue
        cut_pos = int(seps[-2])
        tokens = rec["token_ids"]
        blocks = rec["block_ids"]
        last_block_id = int(blocks[seps[-1]])
        prior_block_id = int(blocks[seps[-2]])
        # History = everything up to and including the cut <sep>
        hist_mask = np.arange(len(tokens)) <= cut_pos
        hist_tokens = set(int(t) for t in tokens[hist_mask] if int(t) in dx_token_ids)
        target_mask = (blocks == last_block_id) & (tokens != SEP_ID)
        target_dx = set(int(t) for t in tokens[target_mask] if int(t) in dx_token_ids)
        if not target_dx:
            continue
        prior_mask = (blocks == prior_block_id) & (tokens != SEP_ID)
        prior_dx = set(int(t) for t in tokens[prior_mask] if int(t) in dx_token_ids)
        out.append(
            EvalInstance(
                token_ids=tokens[: cut_pos + 1],
                block_ids=blocks[: cut_pos + 1],
                position_ids=rec["position_ids"][: cut_pos + 1],
                cut_sep_pos=cut_pos,
                history_tokens=hist_tokens,
                target_dx_ids=target_dx,
                prior_dx_ids=prior_dx,
            )
        )
    return out


def marginal_frequencies(train_records: list[dict], dx_token_ids: set[int], vocab_size: int) -> np.ndarray:
    """Per-token frequency of each diagnosis across training visits (patients * visits)."""
    counts = np.zeros(vocab_size, dtype=np.float64)
    total_visits = 0
    for rec in train_records:
        blocks = rec["block_ids"]
        tokens = rec["token_ids"]
        for block_id in np.unique(blocks):
            if block_id == 0:
                continue
            mask = (blocks == block_id) & (tokens != SEP_ID)
            for t in set(int(x) for x in tokens[mask]):
                if t in dx_token_ids:
                    counts[t] += 1
            total_visits += 1
    counts /= max(1, total_visits)
    return counts


def build_2level_mask(block_ids: np.ndarray) -> torch.Tensor:
    """Same within-block/bidirectional + cross-block/causal mask as the training collator."""
    blk = torch.from_numpy(block_ids).long()
    return (blk[:, None] >= blk[None, :])


def gather_model_probs(
    model: GPTEHR,
    instances: list[EvalInstance],
    dx_ids_sorted: np.ndarray,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Return (N, |DX|) sigmoid probabilities at each instance's cut <sep>."""
    model.eval()
    probs_all = np.zeros((len(instances), len(dx_ids_sorted)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(instances), batch_size):
            batch = instances[start : start + batch_size]
            L = max(len(x.token_ids) for x in batch)
            B = len(batch)
            token_ids = torch.zeros((B, L), dtype=torch.long)
            position_ids = torch.zeros((B, L), dtype=torch.long)
            block_ids = torch.full((B, L), -1, dtype=torch.long)
            attn = torch.zeros((B, L, L), dtype=torch.bool)
            sep_idx = torch.zeros(B, dtype=torch.long)
            for i, inst in enumerate(batch):
                n = len(inst.token_ids)
                token_ids[i, :n] = torch.from_numpy(inst.token_ids.astype(np.int64))
                position_ids[i, :n] = torch.from_numpy(inst.position_ids.astype(np.int64))
                block_ids[i, :n] = torch.from_numpy(inst.block_ids.astype(np.int64))
                attn[i, :n, :n] = build_2level_mask(inst.block_ids)
                sep_idx[i] = inst.cut_sep_pos
            # Padded rows (indices >= n for each sample) would otherwise be all-False,
            # producing NaN softmax in attention. Force the diagonal True everywhere so
            # every query has at least one allowed key. Padded positions' outputs are
            # still discarded because we only read logits[.., sep_idx].
            diag = torch.arange(L)
            attn[:, diag, diag] = True
            token_ids = token_ids.to(device)
            position_ids = position_ids.to(device)
            attn = attn.to(device)
            sep_idx = sep_idx.to(device)

            logits = model(token_ids, position_ids, attn)  # (B, L, V)
            picked = logits[torch.arange(B, device=device), sep_idx]  # (B, V)
            picked = picked[:, dx_ids_sorted]  # restrict to diagnosis sub-vocab
            probs_all[start : start + B] = torch.sigmoid(picked).float().cpu().numpy()
    return probs_all


def compute_metrics(
    probs: np.ndarray,          # (N, K) continuous
    targets: np.ndarray,        # (N, K) 0/1
    history: np.ndarray | None, # (N, K) 0/1 (optional, for new-onset filter)
) -> dict:
    """Micro-averaged AUPRC and best-F1 across all (instance, token) pairs.

    If `history` is given, mask out pairs where the token was present in the patient's history
    so we only score new-onset predictions.
    """
    p = probs
    t = targets
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
    best = float(np.nanmax(f1))
    return {
        "auprc": auprc,
        "best_f1": best,
        "n_pos": int(t.sum()),
        "n_total": int(len(t)),
        "prec": prec.tolist(),
        "rec": rec.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.pkl"))
    parser.add_argument("--train-path", type=Path, default=Path("data/processed/train.pkl"))
    parser.add_argument("--vocab-path", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--out-json", type=Path, default=Path("data/processed/eval_results.json"))
    parser.add_argument("--figure-path", type=Path, default=Path("figures/eval_curves.png"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test patients")
    args = parser.parse_args()

    print("[eval] loading vocab")
    with args.vocab_path.open() as f:
        vocab: dict[str, int] = json.load(f)
    V = len(vocab)
    dx_ids_sorted = np.array(
        sorted(i for tok, i in vocab.items() if tok.startswith("DX")),
        dtype=np.int64,
    )
    dx_id_set = set(int(i) for i in dx_ids_sorted)
    K = len(dx_ids_sorted)
    print(f"[eval] vocab={V}  |DX|={K}")

    print(f"[eval] loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = build_model(V, **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval] device={device}  model params={model.num_parameters()/1e6:.2f}M")

    print("[eval] loading test set")
    with args.test_path.open("rb") as f:
        test_records = pickle.load(f)
    if args.limit:
        test_records = test_records[: args.limit]
    instances = build_eval_instances(test_records, dx_id_set)
    print(f"[eval] test records={len(test_records):,}  eval instances={len(instances):,}")

    # targets matrix (N, K) and history matrix (N, K) restricted to DX tokens.
    N = len(instances)
    targets = np.zeros((N, K), dtype=np.uint8)
    history = np.zeros((N, K), dtype=np.uint8)
    dx_id_to_col = {int(i): c for c, i in enumerate(dx_ids_sorted)}
    for i, inst in enumerate(instances):
        for t in inst.target_dx_ids:
            targets[i, dx_id_to_col[t]] = 1
        for t in inst.history_tokens:
            history[i, dx_id_to_col[t]] = 1

    print("[eval] scoring model")
    model_probs = gather_model_probs(model, instances, dx_ids_sorted, device, args.batch_size)

    print("[eval] computing marginal baseline")
    with args.train_path.open("rb") as f:
        train_records = pickle.load(f)
    marg_full = marginal_frequencies(train_records, dx_id_set, V)
    marg_probs = np.tile(marg_full[dx_ids_sorted], (N, 1)).astype(np.float32)

    print("[eval] computing repeat-history baseline")
    # Alignment: each EvalInstance already carries its prior-visit diagnosis set, so row i of
    # repeat_probs lines up with instances[i] by construction. No filter duplication.
    repeat_probs = np.zeros_like(model_probs)
    for i, inst in enumerate(instances):
        for t in inst.prior_dx_ids:
            repeat_probs[i, dx_id_to_col[t]] = 1.0

    print("[eval] computing metrics")
    results = {}
    for name, probs in [("foundation", model_probs), ("marginal", marg_probs), ("repeat", repeat_probs)]:
        overall = compute_metrics(probs, targets, history=None)
        new_onset = compute_metrics(probs, targets, history=history)
        results[name] = {"overall": overall, "new_onset": new_onset}
        print(
            f"  {name:10s}  overall AUPRC={overall['auprc']:.4f} F1={overall['best_f1']:.4f}  "
            f"new-onset AUPRC={new_onset['auprc']:.4f} F1={new_onset['best_f1']:.4f}"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    light = {
        name: {
            k: {kk: vv for kk, vv in v.items() if kk not in ("prec", "rec")}
            for k, v in r.items()
        }
        for name, r in results.items()
    }
    args.out_json.write_text(json.dumps(light, indent=2))
    print(f"[eval] wrote {args.out_json}")

    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"foundation": "C0", "marginal": "C7", "repeat": "C3"}
    for name in ("marginal", "repeat", "foundation"):
        r_over = results[name]["overall"]
        r_new = results[name]["new_onset"]
        axes[0].plot(r_over["rec"], r_over["prec"], label=f"{name} (AP={r_over['auprc']:.3f})", color=colors[name])
        axes[1].plot(r_new["rec"], r_new["prec"], label=f"{name} (AP={r_new['auprc']:.3f})", color=colors[name])
    for ax, title in zip(axes, ["Overall diagnoses", "New-onset diagnoses"]):
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("GPT-EHR on MIMIC-IV: next-admission diagnosis forecasting", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.figure_path, dpi=150)
    print(f"[eval] wrote {args.figure_path}")


if __name__ == "__main__":
    main()
