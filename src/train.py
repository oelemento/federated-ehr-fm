"""POC training loop for GPT-EHR.

Usage:
    python3.11 src/train.py --config configs/poc.yaml

Designed to run on a Cayuga A100 (BF16, one GPU). Falls back gracefully on
CPU/MPS for dry runs on a laptop.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import LambdaLR

from dataset import make_loader
from model import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int, base_seed: int = 0) -> None:
    """Module-level worker init fn (must be picklable for spawn start method)."""
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_worker_init_fn(base_seed: int):
    return functools.partial(_seed_worker, base_seed=base_seed)


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


def cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


@dataclass
class EpochStats:
    loss: float
    n_seps: int
    wall: float


def run_epoch(
    model,
    loader,
    device,
    amp_dtype,
    optimizer=None,
    scheduler=None,
    grad_clip: float | None = None,
    log_every: int = 50,
    epoch_label: str = "train",
    global_step_ref: list | None = None,
    max_steps: int | None = None,
) -> EpochStats:
    """Run one epoch. When `global_step_ref` is given (a 1-element list), increments it per
    optimizer step and stops as soon as it reaches `max_steps` (useful for smoke tests)."""
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_seps = 0
    t0 = time.time()
    step = 0
    for batch in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        if batch["sep_batch_index"].numel() == 0:
            continue

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(batch["token_ids"], batch["position_ids"], batch["attn_mask_2d"])
            loss, info = model.sep_loss(
                logits,
                batch["sep_batch_index"],
                batch["sep_flat_positions"],
                batch["targets"],
                batch["weights"],
            )

        last_grad_norm: float | None = None
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip is not None:
                # Returns the PRE-clip gradient norm, so we can see how aggressively
                # the optimizer is saturated at the clip threshold.
                last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if global_step_ref is not None:
                global_step_ref[0] += 1

        n_seps = info["n_seps"]
        total_loss += loss.item() * n_seps
        total_seps += n_seps
        step += 1

        if is_train and step % log_every == 0:
            avg = total_loss / max(1, total_seps)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            gn_str = f" grad_norm={last_grad_norm:.2f}" if last_grad_norm is not None else ""
            print(
                f"  [{epoch_label}] step={step} loss={avg:.4f} lr={lr:.2e}{gn_str} elapsed={elapsed:.1f}s"
            )

        if is_train and max_steps is not None and global_step_ref is not None and global_step_ref[0] >= max_steps:
            break

    wall = time.time() - t0
    return EpochStats(
        loss=total_loss / max(1, total_seps),
        n_seps=total_seps,
        wall=wall,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/poc.yaml"))
    parser.add_argument("--limit-train-patients", type=int, default=None,
                        help="For smoke tests: subsample training patients.")
    parser.add_argument("--limit-val-patients", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="For smoke tests: stop after N optimizer steps total.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override cfg.training.batch_size.")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Override cfg.training.num_epochs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override cfg.training.seed. Used for multi-seed variance runs.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Override cfg.training.checkpoint_dir (for per-seed isolation).")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.num_epochs is not None:
        cfg["training"]["num_epochs"] = args.num_epochs
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed
    if args.checkpoint_dir is not None:
        cfg["training"]["checkpoint_dir"] = str(args.checkpoint_dir)
    seed_everything(cfg["training"]["seed"])

    device = pick_device()
    amp_dtype = pick_amp_dtype(cfg["training"]["amp_dtype"], device)
    print(f"[train] device={device} amp_dtype={amp_dtype}")

    with open(cfg["data"]["vocab_path"]) as f:
        vocab = json.load(f)
    V = len(vocab)
    print(f"[train] vocab_size={V}")

    # Optional: tokens that should appear in the input sequence but be masked out of
    # the loss (input-only context). Configured via cfg.loss.exclude_prefixes, e.g.
    # ["LAB:"] to use labs as context without having the model predict them.
    excluded_target_ids: np.ndarray | None = None
    exclude_prefixes = cfg.get("loss", {}).get("exclude_prefixes", [])
    if exclude_prefixes:
        ids = sorted(i for tok, i in vocab.items() if any(tok.startswith(p) for p in exclude_prefixes))
        excluded_target_ids = np.asarray(ids, dtype=np.int64)
        print(f"[train] excluding {len(ids)} token ids from loss (prefixes={exclude_prefixes})")

    worker_init_fn = make_worker_init_fn(cfg["training"]["seed"])
    train_loader = make_loader(
        Path(cfg["data"]["train_path"]),
        vocab_size=V,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        delta=cfg["loss"]["delta"],
        max_len=cfg["data"]["max_len"],
        worker_init_fn=worker_init_fn,
        excluded_target_ids=excluded_target_ids,
    )
    val_loader = make_loader(
        Path(cfg["data"]["val_path"]),
        vocab_size=V,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        delta=cfg["loss"]["delta"],
        max_len=cfg["data"]["max_len"],
        worker_init_fn=worker_init_fn,
        excluded_target_ids=excluded_target_ids,
    )

    if args.limit_train_patients:
        train_loader.dataset.records = train_loader.dataset.records[: args.limit_train_patients]
    if args.limit_val_patients:
        val_loader.dataset.records = val_loader.dataset.records[: args.limit_val_patients]
    print(f"[train] train records={len(train_loader.dataset):,} val records={len(val_loader.dataset):,}")

    model = build_model(V, **cfg["model"]).to(device)
    print(f"[train] model params: {model.num_parameters()/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        betas=tuple(cfg["optim"]["betas"]),
        weight_decay=cfg["optim"]["weight_decay"],
    )
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * cfg["training"]["num_epochs"]
    if args.max_steps:
        total_steps = args.max_steps
    scheduler = cosine_schedule_with_warmup(
        optimizer, cfg["optim"]["warmup_steps"], total_steps
    )
    print(f"[train] total steps: {total_steps}  steps/epoch: {steps_per_epoch}")

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step_ref = [0]
    for epoch in range(1, cfg["training"]["num_epochs"] + 1):
        print(f"\n=== epoch {epoch}/{cfg['training']['num_epochs']} ===")
        train_stats = run_epoch(
            model, train_loader, device, amp_dtype,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=cfg["optim"]["grad_clip"],
            log_every=cfg["training"]["log_every"],
            epoch_label=f"train-e{epoch}",
            global_step_ref=global_step_ref,
            max_steps=args.max_steps,
        )
        with torch.no_grad():
            val_stats = run_epoch(
                model, val_loader, device, amp_dtype,
                epoch_label=f"val-e{epoch}",
            )
        print(
            f"[epoch {epoch}] train loss={train_stats.loss:.4f} ({train_stats.n_seps} seps, {train_stats.wall:.0f}s)"
            f"  val loss={val_stats.loss:.4f} ({val_stats.n_seps} seps, {val_stats.wall:.0f}s)"
        )

        ckpt_path = ckpt_dir / f"epoch_{epoch:02d}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": cfg,
                "vocab_size": V,
                "train_loss": train_stats.loss,
                "val_loss": val_stats.loss,
            },
            ckpt_path,
        )
        print(f"[epoch {epoch}] saved {ckpt_path}")

        if val_stats.loss < best_val:
            best_val = val_stats.loss
            best_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "vocab_size": V,
                    "val_loss": val_stats.loss,
                },
                best_path,
            )
            print(f"[epoch {epoch}] new best val loss, saved {best_path}")

        if args.max_steps is not None and global_step_ref[0] >= args.max_steps:
            print(f"[train] hit --max-steps {args.max_steps} after epoch {epoch}; stopping")
            break


if __name__ == "__main__":
    main()
