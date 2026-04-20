"""Personalized evaluation of FedPer / FedPer+LoRA checkpoints.

The standard `evaluate.py` path loads only `model_state`, which contains the
(weighted-averaged) `output_proj` placeholder. That does NOT match the FedPer
deployment story — FedPer keeps `output_proj` per-site, so eval should use
each site's personalized head.

This script restores each site's `output_proj` from `ckpt["site_output_proj"]`
and reports three numbers:

    mean_personalized : mean AUPRC across the N site-personalized
                        evaluations (one per site head)
    head_ensemble     : AUPRC of the average of the per-site-head predictions
                        (shared backbone, ensemble over heads)
    global_head       : the AUPRC produced by the averaged output_proj
                        placeholder (what evaluate.py currently reports).

Per-condition AUROCs are computed for the head_ensemble deployment, which is
the natural apples-to-apples comparison to our inference-time ensemble.

Usage
-----
    PYTHONPATH=src python3.11 src/evaluate_fedper_personalized.py \\
        --checkpoint checkpoints_fedper_a0.5_s20260411/best.pt \\
        --out-json data/processed/eval_fedper_personalized_s20260411.json \\
        --cond-out-json data/processed/cond_fedper_personalized_s20260411.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate import (
    build_eval_instances,
    compute_metrics,
    gather_model_probs,
)
from evaluate_conditions import CONDITION_CODES
from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.pkl"))
    parser.add_argument("--vocab-path", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--cond-out-json", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))

    with args.vocab_path.open() as f:
        vocab: dict[str, int] = json.load(f)
    V = len(vocab)
    dx_ids_sorted = np.array(
        sorted(i for tok, i in vocab.items() if tok.startswith("DX")),
        dtype=np.int64,
    )
    dx_id_set = set(int(i) for i in dx_ids_sorted)
    K = len(dx_ids_sorted)
    print(f"[fedper-eval] vocab={V}  |DX|={K}")

    print(f"[fedper-eval] loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "site_output_proj" not in ckpt:
        raise RuntimeError(
            f"Checkpoint {args.checkpoint} does not contain 'site_output_proj'. "
            "Is this really a FedPer/FedPer+LoRA checkpoint?"
        )
    site_heads = ckpt["site_output_proj"]
    n_sites = len(site_heads)
    print(f"[fedper-eval] found {n_sites} per-site output_proj heads")

    cfg = ckpt["config"]
    model = build_model(V, **cfg["model"]).to(device)
    # Load the aggregated state (backbone + averaged head)
    model.load_state_dict(ckpt["model_state"])

    print("[fedper-eval] loading test set")
    with args.test_path.open("rb") as f:
        test_records = pickle.load(f)
    instances = build_eval_instances(test_records, dx_id_set)
    N = len(instances)
    print(f"[fedper-eval] test instances={N:,}")

    # Build (N, K) targets and history matrices
    targets = np.zeros((N, K), dtype=np.uint8)
    history = np.zeros((N, K), dtype=np.uint8)
    dx_id_to_col = {int(i): c for c, i in enumerate(dx_ids_sorted)}
    for i, inst in enumerate(instances):
        for t in inst.target_dx_ids:
            targets[i, dx_id_to_col[t]] = 1
        for t in inst.history_tokens:
            history[i, dx_id_to_col[t]] = 1

    # 1) Global-head (what evaluate.py would produce on this ckpt)
    print("[fedper-eval] scoring global (averaged) head")
    probs_global = gather_model_probs(model, instances, dx_ids_sorted, device, args.batch_size)
    global_overall = compute_metrics(probs_global, targets, history=None)
    global_new = compute_metrics(probs_global, targets, history=history)

    # 2) Per-site personalized heads
    site_probs = []
    per_site_results = []
    for s, head_w in enumerate(site_heads):
        print(f"[fedper-eval] scoring site {s}'s personalized head")
        assert head_w.dtype == model.output_proj.weight.dtype, (
            f"site head dtype {head_w.dtype} != model output_proj dtype "
            f"{model.output_proj.weight.dtype}"
        )
        with torch.no_grad():
            model.output_proj.weight.data.copy_(head_w.to(device))
        probs_s = gather_model_probs(model, instances, dx_ids_sorted, device, args.batch_size)
        site_probs.append(probs_s)
        o = compute_metrics(probs_s, targets, history=None)
        n = compute_metrics(probs_s, targets, history=history)
        # Strip prec/rec arrays — they are large and not needed in the summary JSON.
        o_light = {k: v for k, v in o.items() if k not in ("prec", "rec")}
        n_light = {k: v for k, v in n.items() if k not in ("prec", "rec")}
        per_site_results.append({"site": s, "overall": o_light, "new_onset": n_light})
        print(
            f"  site {s}: overall AUPRC={o['auprc']:.4f}  new-onset AUPRC={n['auprc']:.4f}"
        )

    # 3) Head-ensemble: average predictions across site-personalized heads
    probs_head_ens = np.mean(np.stack(site_probs, axis=0), axis=0)
    he_overall = compute_metrics(probs_head_ens, targets, history=None)
    he_new = compute_metrics(probs_head_ens, targets, history=history)

    # Aggregate per-site mean ± SD
    per_site_ov = [r["overall"]["auprc"] for r in per_site_results]
    per_site_no = [r["new_onset"]["auprc"] for r in per_site_results]

    summary = {
        "checkpoint": str(args.checkpoint),
        "strategy": ckpt.get("strategy", "?"),
        "n_sites": n_sites,
        "alpha": ckpt.get("alpha"),
        "global_head": {
            "overall": {k: v for k, v in global_overall.items() if k not in ("prec", "rec")},
            "new_onset": {k: v for k, v in global_new.items() if k not in ("prec", "rec")},
        },
        "per_site_mean_personalized": {
            "overall_auprc_mean": float(np.mean(per_site_ov)),
            "overall_auprc_std":  float(np.std(per_site_ov, ddof=1)),
            "new_onset_auprc_mean": float(np.mean(per_site_no)),
            "new_onset_auprc_std":  float(np.std(per_site_no, ddof=1)),
            "per_site": per_site_results,
        },
        "head_ensemble": {
            "overall": {k: v for k, v in he_overall.items() if k not in ("prec", "rec")},
            "new_onset": {k: v for k, v in he_new.items() if k not in ("prec", "rec")},
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"[fedper-eval] wrote {args.out_json}")

    print(
        f"\n[fedper-eval] summary:\n"
        f"  global (averaged) head: overall={global_overall['auprc']:.4f}  new-onset={global_new['auprc']:.4f}\n"
        f"  per-site-mean personalized: overall={np.mean(per_site_ov):.4f} +/- {np.std(per_site_ov, ddof=1):.4f}  "
        f"new-onset={np.mean(per_site_no):.4f} +/- {np.std(per_site_no, ddof=1):.4f}\n"
        f"  head-ensemble: overall={he_overall['auprc']:.4f}  new-onset={he_new['auprc']:.4f}"
    )

    # Per-condition eval on head-ensemble predictions
    if args.cond_out_json:
        print("\n[fedper-eval] per-condition AUROC (head-ensemble)")
        # Per-condition scores are sum-of-sigmoid-probs over each condition's
        # codeset. Since every CONDITION_CODES prefix resolves to a DX token
        # and sigmoid is elementwise, summing DX-restricted probs is exactly
        # equivalent to the full-vocab approach in evaluate_conditions.py.
        cond_results = []
        for cname, prefixes in CONDITION_CODES.items():
            # Token ids in the condition codeset that are also DX tokens
            cond_dx_ids = [vocab[tok] for tok in vocab
                           if any(tok.startswith(p) for p in prefixes)]
            cond_dx_ids = [i for i in cond_dx_ids if i in dx_id_set]
            if not cond_dx_ids:
                print(f"  skip {cname}: no matching DX tokens")
                continue
            cols = [dx_id_to_col[i] for i in cond_dx_ids]
            # Build labels = 1 iff any codeset token appears in target AND no
            # codeset token appeared in history (new-onset filter)
            in_history = history[:, cols].any(axis=1)
            in_target  = targets[:, cols].any(axis=1)
            eligible = ~in_history
            if eligible.sum() == 0:
                print(f"  skip {cname}: no eligible patients")
                continue
            y = in_target[eligible].astype(np.int64)
            # score: sum of predicted probs across codeset tokens
            s = probs_head_ens[eligible][:, cols].sum(axis=1)
            if y.sum() == 0 or y.sum() == len(y):
                print(f"  skip {cname}: degenerate labels (n_pos={int(y.sum())})")
                continue
            auroc = float(roc_auc_score(y, s))
            auprc = float(average_precision_score(y, s))
            # Bootstrap — seed matches evaluate_conditions.py for reproducibility
            rng = np.random.default_rng(20260411)
            boot_auroc, boot_auprc = [], []
            for _ in range(args.n_boot):
                idx = rng.integers(0, len(y), size=len(y))
                if y[idx].sum() == 0 or y[idx].sum() == len(idx):
                    continue
                boot_auroc.append(roc_auc_score(y[idx], s[idx]))
                boot_auprc.append(average_precision_score(y[idx], s[idx]))
            if boot_auroc:
                auroc_ci = [float(np.percentile(boot_auroc, 2.5)),
                            float(np.percentile(boot_auroc, 97.5))]
                auprc_ci = [float(np.percentile(boot_auprc, 2.5)),
                            float(np.percentile(boot_auprc, 97.5))]
            else:
                auroc_ci = [float("nan"), float("nan")]
                auprc_ci = [float("nan"), float("nan")]
            cond_results.append({
                "name": cname,
                "n_eligible": int(eligible.sum()),
                "n_positive": int(y.sum()),
                "prevalence": float(y.sum() / len(y)),
                "auroc": auroc, "auroc_ci": auroc_ci,
                "auprc": auprc, "auprc_ci": auprc_ci,
            })
            print(
                f"  {cname:25s}  n={int(eligible.sum()):>5}  pos={int(y.sum()):>4}  "
                f"AUROC={auroc:.3f} [{auroc_ci[0]:.3f},{auroc_ci[1]:.3f}]  "
                f"AUPRC={auprc:.3f}"
            )

        cond_summary = {"checkpoint": str(args.checkpoint), "results": cond_results}
        args.cond_out_json.parent.mkdir(parents=True, exist_ok=True)
        args.cond_out_json.write_text(json.dumps(cond_summary, indent=2))
        print(f"[fedper-eval] wrote {args.cond_out_json}")


if __name__ == "__main__":
    main()
