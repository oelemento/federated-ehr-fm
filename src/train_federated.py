"""Federated training POC for GPT-EHR.

Simulates multi-site federation on a single dataset (MIMIC-IV) by splitting
the training patients into N sites with non-IID allocation, training local
models, and averaging weights.

Strategies:
    fedavg  — FedAvg (McMahan et al. 2017): average model weights after each
              local training round.
    fedprox — FedProx (Li et al. 2020): adds a proximal regularization term
              (mu/2)||w - w_global||^2 to each site's local loss, penalizing
              drift from the global model.

Non-IID split:
    Patients are grouped by their dominant ICD chapter (first character of
    their most-frequent DX token). A Dirichlet(alpha) allocation assigns
    patients of each chapter to sites. Low alpha (e.g. 0.5) creates strong
    non-IID; high alpha (e.g. 100) approximates IID.

Usage
-----
    PYTHONPATH=src python -u src/train_federated.py \\
        --config configs/poc.yaml \\
        --n-sites 5 --n-rounds 15 --local-epochs 1 \\
        --alpha 0.5 --strategy fedprox --mu 0.01 \\
        --checkpoint-dir checkpoints_federated

    # Or via sbatch:
    sbatch configs/federated.slurm
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset import PatientSequenceDataset, CollateFn, CollatorConfig
from model import build_model, apply_lora, merge_lora


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_amp_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if requested == "bf16":
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    return None


# ---------------------------------------------------------------------------
# Non-IID site assignment
# ---------------------------------------------------------------------------

def assign_sites_dirichlet(
    records: list[dict],
    vocab: dict[str, int],
    n_sites: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """Assign patient indices to sites using Dirichlet-based non-IID allocation.

    For each ICD chapter, we sample a proportion vector p ~ Dir(alpha) over
    n_sites and assign patients of that chapter to sites according to p.
    Returns a list of n_sites lists of patient indices.
    """
    rng = np.random.default_rng(seed)
    id_to_tok = {v: k for k, v in vocab.items()}

    def dx_chapter(tid: int) -> str | None:
        tok = id_to_tok.get(tid)
        if tok is None:
            return None
        if tok.startswith("DX10:"):
            return tok[5]
        if tok.startswith("DX9:"):
            return "9" + tok[4]
        return None

    # Find dominant chapter per patient
    patient_chapters: list[str] = []
    for rec in records:
        chs = []
        for t in rec["token_ids"]:
            ch = dx_chapter(int(t))
            if ch:
                chs.append(ch)
        if chs:
            patient_chapters.append(Counter(chs).most_common(1)[0][0])
        else:
            patient_chapters.append("X")

    # Group patient indices by chapter
    chapter_to_patients: dict[str, list[int]] = {}
    for i, ch in enumerate(patient_chapters):
        chapter_to_patients.setdefault(ch, []).append(i)

    # Dirichlet allocation: for each chapter, draw proportions over sites
    site_indices: list[list[int]] = [[] for _ in range(n_sites)]
    for ch, patient_ids in sorted(chapter_to_patients.items()):
        proportions = rng.dirichlet([alpha] * n_sites)
        rng.shuffle(patient_ids)
        # Use cumulative-sum partitioning to avoid negative-split rounding bugs
        cum = np.round(np.cumsum(proportions) * len(patient_ids)).astype(int)
        cum[-1] = len(patient_ids)  # ensure exact coverage
        prev = 0
        for site_id in range(n_sites):
            site_indices[site_id].extend(patient_ids[prev:cum[site_id]])
            prev = cum[site_id]

    for s in range(n_sites):
        rng.shuffle(site_indices[s])

    return site_indices


def print_site_stats(
    site_indices: list[list[int]],
    records: list[dict],
    vocab: dict[str, int],
) -> None:
    """Print per-site patient count and top-3 dominant ICD chapters."""
    id_to_tok = {v: k for k, v in vocab.items()}

    def dx_chapter(tid: int) -> str | None:
        tok = id_to_tok.get(tid)
        if tok is None:
            return None
        if tok.startswith("DX10:"):
            return tok[5]
        if tok.startswith("DX9:"):
            return "9" + tok[4]
        return None

    for s, idxs in enumerate(site_indices):
        chs: list[str] = []
        for i in idxs:
            for t in records[i]["token_ids"]:
                ch = dx_chapter(int(t))
                if ch:
                    chs.append(ch)
        top3 = Counter(chs).most_common(3)
        top3_str = ", ".join(f"{ch}={n}" for ch, n in top3)
        print(f"  site {s}: {len(idxs):,} patients  top chapters: {top3_str}")


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def train_local(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    grad_clip: float,
    n_epochs: int,
    global_state: dict | None = None,
    mu: float = 0.0,
    max_steps: int | None = None,
    optimizer_state: dict | None = None,
    prox_skip_keys: set[str] | None = None,
) -> dict:
    """Train the model on one site's data for n_epochs (or max_steps, whichever is first).

    If global_state and mu > 0, adds a FedProx proximal term:
        (mu/2) * sum((param - global_param)^2)

    If optimizer_state is provided, the optimizer is initialized from it (preserving
    Adam momentum across rounds). The updated optimizer state is returned in the result
    dict so the caller can persist it.
    """
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=lr, betas=betas, weight_decay=weight_decay,
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        # Update LR in case it changed between rounds (server-side decay)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

    global_params = None
    if global_state is not None and mu > 0:
        global_params = {
            name: param.detach().clone().to(device)
            for name, param in global_state.items()
        }

    total_loss = 0.0
    total_seps = 0
    total_active = 0  # sum of active (non-zero weight) positions — used for FedAvg weighting
    step = 0
    for _epoch in range(n_epochs):
        for batch in loader:
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            if batch["sep_batch_index"].numel() == 0:
                continue

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype,
                enabled=amp_dtype is not None,
            ):
                logits = model(
                    batch["token_ids"],
                    batch["position_ids"],
                    batch["attn_mask_2d"],
                )
                loss, info = model.sep_loss(
                    logits,
                    batch["sep_batch_index"],
                    batch["sep_flat_positions"],
                    batch["targets"],
                    batch["weights"],
                )

            # FedProx proximal term (excludes personal layers in FedPer so they
            # can specialize per-site without being penalized for drifting)
            if global_params is not None and mu > 0:
                prox = sum(
                    ((p - global_params[name]) ** 2).sum()
                    for name, p in model.named_parameters()
                    if p.requires_grad and (prox_skip_keys is None or name not in prox_skip_keys)
                )
                loss = loss + (mu / 2.0) * prox

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item() * info["n_seps"]
            total_seps += info["n_seps"]
            total_active += int(batch["weights"].gt(0).sum().item())
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    avg_loss = total_loss / max(1, total_seps)
    return {
        "loss": avg_loss,
        "n_seps": total_seps,
        "total_active": total_active,
        "steps": step,
        "optimizer_state": optimizer.state_dict(),
    }


# ---------------------------------------------------------------------------
# Weight averaging
# ---------------------------------------------------------------------------

def fedavg(
    local_states: list[dict[str, torch.Tensor]],
    weights: list[float],
    skip_keys: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted average of model state dicts.

    Keys in `skip_keys` are excluded from averaging — each site keeps its own
    version (used by FedPer to keep per-site embeddings). For skipped keys,
    the first site's values are used as a placeholder in the returned dict.
    """
    total = sum(weights)
    w = [x / total for x in weights]
    avg = {}
    for key in local_states[0]:
        if skip_keys and key in skip_keys:
            # Placeholder — caller handles per-site values separately.
            # Use the weighted average anyway so eval has something reasonable.
            avg[key] = sum(
                w[i] * local_states[i][key].float()
                for i in range(len(local_states))
            ).to(local_states[0][key].dtype)
        else:
            avg[key] = sum(
                w[i] * local_states[i][key].float()
                for i in range(len(local_states))
            ).to(local_states[0][key].dtype)
    return avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--n-sites", type=int, default=5)
    parser.add_argument("--n-rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dirichlet concentration. Lower = more non-IID.")
    parser.add_argument("--strategy", choices=["fedavg", "fedprox", "fedper", "fedlora", "fedper_lora"], default="fedavg")
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA adapter rank (only used with --strategy fedlora).")
    parser.add_argument("--mu", type=float, default=0.1,
                        help="FedProx proximal term weight (ignored for fedavg).")
    parser.add_argument("--local-steps", type=int, default=None,
                        help="Cap local training at N steps per site per round (default: full epoch).")
    parser.add_argument("--lr-decay", type=float, default=0.95,
                        help="Multiply server-side LR by this factor each round.")
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_federated"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device (cpu/cuda/mps). Default: auto-detect.")
    parser.add_argument("--init-checkpoint", type=Path, default=None,
                        help="Initialize global model from a pretrained checkpoint (e.g. centralized best.pt).")
    parser.add_argument("--untie-embeddings", action="store_true",
                        help="Use separate output_proj even for fedavg/fedprox (still averaged, "
                             "not personalized). Tests whether the failure is from weight tying "
                             "vs from averaging the output projection.")
    parser.add_argument("--save-persite-last-round", action="store_true",
                        help="On the final round, save each site's state_dict BEFORE averaging "
                             "to <checkpoint_dir>/persite_site{s}_preavg.pt. Used for in-round "
                             "divergence analysis (does the tied matrix diverge across sites "
                             "more than untied E / W_out at the ~100-local-step drift scale?).")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else pick_device()
    amp_dtype = pick_amp_dtype(cfg["training"]["amp_dtype"], device)
    print(f"[fed] device={device} amp_dtype={amp_dtype} strategy={args.strategy}")

    with open(cfg["data"]["vocab_path"]) as f:
        vocab = json.load(f)
    V = len(vocab)

    excluded_target_ids: np.ndarray | None = None
    exclude_prefixes = cfg.get("loss", {}).get("exclude_prefixes", [])
    if exclude_prefixes:
        ids = sorted(i for tok, i in vocab.items() if any(tok.startswith(p) for p in exclude_prefixes))
        excluded_target_ids = np.asarray(ids, dtype=np.int64)
        print(f"[fed] excluding {len(ids)} token ids from loss (prefixes={exclude_prefixes})")

    print(f"[fed] loading training data from {cfg['data']['train_path']}")
    ds = PatientSequenceDataset(Path(cfg["data"]["train_path"]))
    all_records = ds.records
    print(f"[fed] {len(all_records):,} training patients")

    # Non-IID site assignment
    print(f"[fed] splitting into {args.n_sites} sites (alpha={args.alpha})")
    site_indices = assign_sites_dirichlet(
        all_records, vocab, args.n_sites, args.alpha, args.seed,
    )
    print_site_stats(site_indices, all_records, vocab)

    # Build per-site data loaders
    collate_cfg = CollatorConfig(
        vocab_size=V,
        max_len=cfg["data"]["max_len"],
        delta=cfg["loss"]["delta"],
        excluded_target_ids=excluded_target_ids,
    )
    site_loaders: list[torch.utils.data.DataLoader] = []
    for s, idxs in enumerate(site_indices):
        site_ds = PatientSequenceDataset.__new__(PatientSequenceDataset)
        site_ds.records = [all_records[i] for i in idxs]
        loader = torch.utils.data.DataLoader(
            site_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=0,
            collate_fn=CollateFn(collate_cfg),
            pin_memory=False,
        )
        site_loaders.append(loader)

    # Validation loader
    val_ds = PatientSequenceDataset(Path(cfg["data"]["val_path"]))
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=CollateFn(collate_cfg),
        pin_memory=False,
    )

    # Strategy flags (defined here so model init can use them)
    is_fedper = args.strategy in ("fedper", "fedper_lora")
    is_fedlora = args.strategy in ("fedlora", "fedper_lora")
    mu = args.mu if args.strategy in ("fedprox", "fedper", "fedlora", "fedper_lora") else 0.0

    # Initialize global model. FedPer needs untied embeddings so the output projection
    # can be kept per-site while the input embedding and transformer blocks are shared.
    # --untie-embeddings allows untied embeddings for non-FedPer strategies too (control
    # experiment: test whether the failure is from weight tying vs averaging output_proj).
    model_overrides = dict(cfg["model"])
    if is_fedper or args.untie_embeddings:
        model_overrides["tie_embeddings"] = False
        cfg["model"]["tie_embeddings"] = False  # persist in cfg so checkpoint is self-contained
    model = build_model(V, **model_overrides).to(device)
    if args.init_checkpoint:
        # Load pretrained weights BEFORE applying LoRA so the base weights get the
        # pretrained values (LoRA wrapping changes state_dict keys from 'qkv.weight'
        # to 'qkv.base.weight', so loading after wrapping would silently drop them).
        print(f"[fed] loading pretrained init from {args.init_checkpoint}")
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"], strict=False)
        print(f"[fed] loaded checkpoint (val_loss={ckpt.get('val_loss', '?')})")
    if is_fedlora:
        apply_lora(model, rank=args.lora_rank, alpha=1.0)
        # FedPer+LoRA: apply_lora froze everything including output_proj.
        # Unfreeze output_proj so it stays trainable and per-site.
        if is_fedper and hasattr(model, "output_proj"):
            model.output_proj.weight.requires_grad_(True)
            n_unfrozen = model.output_proj.weight.numel()
            print(f"[fed] unfroze output_proj ({n_unfrozen:,} params) for FedPer+LoRA")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fed] model params: {model.num_parameters()/1e6:.2f}M ({n_trainable:,} trainable)")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    current_lr = cfg["optim"]["lr"]

    # Per-site optimizer state persistence (avoids resetting Adam momentum each round)
    site_optimizer_states: list[dict | None] = [None] * args.n_sites
    # FedPer: per-site output projection weights (personal layers, not averaged).
    # With untied embeddings, tok_emb is shared (averaged) and output_proj is personal.
    site_output_proj: list[torch.Tensor | None] = [None] * args.n_sites
    PERSONAL_KEYS = {"output_proj.weight"} if is_fedper else set()

    print(
        f"[fed] starting: {args.n_rounds} rounds, {args.local_epochs} local epoch(s), "
        f"max_local_steps={args.local_steps}, lr={current_lr}, lr_decay={args.lr_decay}, "
        f"mu={mu}, strategy={args.strategy}, "
        f"personal_keys={PERSONAL_KEYS if PERSONAL_KEYS else 'none'}, "
        f"init={'pretrained' if args.init_checkpoint else 'random'}"
    )

    for rnd in range(1, args.n_rounds + 1):
        t0 = time.time()
        global_state = copy.deepcopy(model.state_dict())
        local_states: list[dict[str, torch.Tensor]] = []
        site_weights: list[float] = []
        site_losses: list[float] = []

        for s, loader in enumerate(site_loaders):
            model.load_state_dict(global_state)
            # FedPer: restore per-site personal layers before local training
            if is_fedper and site_output_proj[s] is not None:
                model.output_proj.weight.data.copy_(site_output_proj[s])
            info = train_local(
                model, loader, device, amp_dtype,
                lr=current_lr,
                weight_decay=cfg["optim"]["weight_decay"],
                betas=tuple(cfg["optim"]["betas"]),
                grad_clip=cfg["optim"]["grad_clip"],
                n_epochs=args.local_epochs,
                global_state=global_state if mu > 0 else None,
                mu=mu,
                max_steps=args.local_steps,
                optimizer_state=site_optimizer_states[s],
                prox_skip_keys=PERSONAL_KEYS if PERSONAL_KEYS else None,
            )
            local_states.append(copy.deepcopy(model.state_dict()))
            site_weights.append(float(info["total_active"]))
            site_losses.append(info["loss"])
            site_optimizer_states[s] = info["optimizer_state"]
            # FedPer: save per-site personal layers
            if is_fedper:
                site_output_proj[s] = model.output_proj.weight.data.clone()

        # On the last round, optionally snapshot each site's state BEFORE averaging.
        # Lets us measure cross-site matrix drift at the real FedAvg drift scale
        # (local_steps of local training, not full-epoch independent training).
        if args.save_persite_last_round and rnd == args.n_rounds:
            for s, sd in enumerate(local_states):
                out = args.checkpoint_dir / f"persite_site{s}_preavg.pt"
                torch.save(
                    {"round": rnd, "site": s, "model_state": sd, "config": cfg,
                     "vocab_size": V, "strategy": args.strategy, "alpha": args.alpha},
                    out,
                )
            print(f"[fed] saved pre-averaging per-site state for round {rnd}")

        # Aggregate (FedPer: skip personal keys during averaging, but include
        # their weighted average in the global model for eval purposes)
        aggregated = fedavg(local_states, site_weights, skip_keys=PERSONAL_KEYS)
        model.load_state_dict(aggregated)

        # Validation
        model.eval()
        val_loss = 0.0
        val_seps = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                    for k, v in batch.items()
                }
                if batch["sep_batch_index"].numel() == 0:
                    continue
                with torch.autocast(
                    device_type=device.type, dtype=amp_dtype,
                    enabled=amp_dtype is not None,
                ):
                    logits = model(
                        batch["token_ids"],
                        batch["position_ids"],
                        batch["attn_mask_2d"],
                    )
                    loss, info = model.sep_loss(
                        logits,
                        batch["sep_batch_index"],
                        batch["sep_flat_positions"],
                        batch["targets"],
                        batch["weights"],
                    )
                val_loss += loss.item() * info["n_seps"]
                val_seps += info["n_seps"]
        val_loss /= max(1, val_seps)
        wall = time.time() - t0

        avg_site_loss = sum(site_losses) / len(site_losses)
        print(
            f"[fed] round {rnd}/{args.n_rounds}  "
            f"site_loss_avg={avg_site_loss:.4f}  val_loss={val_loss:.4f}  "
            f"lr={current_lr:.2e}  wall={wall:.0f}s"
        )

        # Server-side LR decay
        current_lr *= args.lr_decay

        ckpt = {
            "round": rnd,
            "model_state": model.state_dict(),
            "config": cfg,
            "vocab_size": V,
            "val_loss": val_loss,
            "strategy": args.strategy,
            "n_sites": args.n_sites,
            "alpha": args.alpha,
        }
        # FedPer: also save per-site personal weights so we can evaluate
        # personalized models (shared transformer + per-site head) later.
        if is_fedper and site_output_proj[0] is not None:
            ckpt["site_output_proj"] = [w.cpu() for w in site_output_proj]
        torch.save(ckpt, args.checkpoint_dir / f"round_{rnd:02d}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.checkpoint_dir / "best.pt")
            print(f"[fed] new best val loss at round {rnd}")

    # For LoRA: merge adapters into base weights so the saved checkpoint is a
    # standard GPTEHR model that evaluate.py can load without LoRA awareness.
    if is_fedlora:
        print("[fed] merging LoRA adapters into base weights for eval checkpoint")
        best_ckpt = torch.load(args.checkpoint_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt["model_state"])
        merge_lora(model)
        best_ckpt["model_state"] = model.state_dict()
        torch.save(best_ckpt, args.checkpoint_dir / "best.pt")

    print(f"[fed] done. best val loss = {best_val:.4f}")


if __name__ == "__main__":
    main()
