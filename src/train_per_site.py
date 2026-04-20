"""Train independent per-site models to establish the baseline for federation value.

For each of the N Dirichlet-allocated sites, trains a standard centralized model
on ONLY that site's patients and evaluates on the shared test set. This answers:
"Is the federated model better than what each site could achieve alone?"

Usage
-----
    PYTHONPATH=src python -u src/train_per_site.py \
        --config configs/poc.yaml --n-sites 5 --alpha 0.5 \
        --num-epochs 3 --seed 20260411

Outputs per-site eval results to data/processed/eval_results_site_N.json
and a summary comparison to stdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import PatientSequenceDataset, CollateFn, CollatorConfig, make_loader
from model import build_model
from train import seed_everything, pick_device, pick_amp_dtype, run_epoch, make_worker_init_fn, EpochStats
from train_federated import assign_sites_dirichlet, print_site_stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--n-sites", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_per_site"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    seed_everything(args.seed)
    device = pick_device()
    amp_dtype = pick_amp_dtype(cfg["training"]["amp_dtype"], device)

    with open(cfg["data"]["vocab_path"]) as f:
        vocab = json.load(f)
    V = len(vocab)

    excluded_target_ids: np.ndarray | None = None
    exclude_prefixes = cfg.get("loss", {}).get("exclude_prefixes", [])
    if exclude_prefixes:
        ids = sorted(i for tok, i in vocab.items() if any(tok.startswith(p) for p in exclude_prefixes))
        excluded_target_ids = np.asarray(ids, dtype=np.int64)

    # Load all training data and split into sites
    ds = PatientSequenceDataset(Path(cfg["data"]["train_path"]))
    all_records = ds.records
    site_indices = assign_sites_dirichlet(all_records, vocab, args.n_sites, args.alpha, args.seed)
    print_site_stats(site_indices, all_records, vocab)

    # Shared val loader for comparable evaluation
    collate_cfg = CollatorConfig(
        vocab_size=V, max_len=cfg["data"]["max_len"],
        delta=cfg["loss"]["delta"], excluded_target_ids=excluded_target_ids,
    )

    results_summary = []

    for s in range(args.n_sites):
        print(f"\n{'='*60}")
        print(f"  SITE {s}: {len(site_indices[s]):,} patients")
        print(f"{'='*60}")

        seed_everything(args.seed + s)

        # Per-site dataset
        site_ds = PatientSequenceDataset.__new__(PatientSequenceDataset)
        site_ds.records = [all_records[i] for i in site_indices[s]]
        site_loader = torch.utils.data.DataLoader(
            site_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=0,
            collate_fn=CollateFn(collate_cfg),
        )

        val_loader = make_loader(
            Path(cfg["data"]["val_path"]),
            vocab_size=V,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=0,
            delta=cfg["loss"]["delta"],
            max_len=cfg["data"]["max_len"],
            excluded_target_ids=excluded_target_ids,
        )

        model = build_model(V, **cfg["model"]).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["optim"]["lr"],
            betas=tuple(cfg["optim"]["betas"]),
            weight_decay=cfg["optim"]["weight_decay"],
        )

        best_val = float("inf")
        ckpt_dir = args.checkpoint_dir / f"site_{s}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.num_epochs + 1):
            train_stats = run_epoch(
                model, site_loader, device, amp_dtype,
                optimizer=optimizer,
                grad_clip=cfg["optim"]["grad_clip"],
                epoch_label=f"site{s}-e{epoch}",
            )
            with torch.no_grad():
                val_stats = run_epoch(
                    model, val_loader, device, amp_dtype,
                    epoch_label=f"site{s}-val{epoch}",
                )
            print(
                f"  [site {s} epoch {epoch}] train={train_stats.loss:.4f} "
                f"val={val_stats.loss:.4f} wall={train_stats.wall:.0f}s"
            )
            if val_stats.loss < best_val:
                best_val = val_stats.loss
                torch.save({
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "vocab_size": V,
                    "val_loss": val_stats.loss,
                    "site": s,
                    "n_patients": len(site_indices[s]),
                }, ckpt_dir / "best.pt")

        print(f"  [site {s}] best val loss = {best_val:.4f}")
        results_summary.append({
            "site": s,
            "n_patients": len(site_indices[s]),
            "best_val_loss": best_val,
        })

    # Now evaluate each site's best model
    print(f"\n{'='*60}")
    print("  EVALUATING ALL SITE MODELS")
    print(f"{'='*60}")

    import pickle
    from evaluate import build_eval_instances, gather_model_probs, marginal_frequencies, compute_metrics
    from dataset import SEP_ID

    with open(cfg["data"]["test_path"], "rb") as f:
        test_records = pickle.load(f)
    with open(cfg["data"]["train_path"], "rb") as f:
        train_records = pickle.load(f)

    dx_ids_sorted = np.array(
        sorted(i for tok, i in vocab.items() if tok.startswith("DX")),
        dtype=np.int64,
    )
    dx_id_set = set(int(i) for i in dx_ids_sorted)
    K = len(dx_ids_sorted)

    instances = build_eval_instances(test_records, dx_id_set)
    N = len(instances)

    # Build targets and history matrices
    dx_id_to_col = {int(i): c for c, i in enumerate(dx_ids_sorted)}
    targets = np.zeros((N, K), dtype=np.uint8)
    history = np.zeros((N, K), dtype=np.uint8)
    for i, inst in enumerate(instances):
        for t in inst.target_dx_ids:
            targets[i, dx_id_to_col[t]] = 1
        for t in inst.history_tokens:
            history[i, dx_id_to_col[t]] = 1

    for s in range(args.n_sites):
        ckpt_path = args.checkpoint_dir / f"site_{s}" / "best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(V, **cfg["model"]).to(device)
        model.load_state_dict(ckpt["model_state"])

        probs = gather_model_probs(model, instances, dx_ids_sorted, device)
        overall = compute_metrics(probs, targets, history=None)
        new_onset = compute_metrics(probs, targets, history=history)

        out_path = args.checkpoint_dir / f"eval_results_site_{s}.json"
        out_path.write_text(json.dumps({
            "site": s,
            "n_patients": ckpt["n_patients"],
            "overall": {k: v for k, v in overall.items() if k not in ("prec", "rec")},
            "new_onset": {k: v for k, v in new_onset.items() if k not in ("prec", "rec")},
        }, indent=2))

        print(
            f"  site {s} ({ckpt['n_patients']:,} pts): "
            f"overall AUPRC={overall['auprc']:.4f}  "
            f"new-onset AUPRC={new_onset['auprc']:.4f}"
        )
        results_summary[s]["overall_auprc"] = overall["auprc"]
        results_summary[s]["new_onset_auprc"] = new_onset["auprc"]

    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    auprcs = [r["overall_auprc"] for r in results_summary]
    print(f"  Per-site mean AUPRC:    {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"  Per-site range:         {min(auprcs):.4f} – {max(auprcs):.4f}")
    print(f"  FedPer:                 0.2200")
    print(f"  FedPer+LoRA:            0.2419")
    print(f"  Centralized:            0.3607")


if __name__ == "__main__":
    main()
