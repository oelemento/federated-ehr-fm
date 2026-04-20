"""GPT-EHR: decoder-only Transformer with two-level attention and RoPE on day indices.

Tailored for the POC:
- Custom attention mask passed from the collator (bidirectional within visit,
  causal across visits).
- Rotary positional embeddings indexed by `position_ids` (day index, where
  the <sep>_i token carries day(i+1)).
- Multi-label BCE loss evaluated only at <sep> positions (via gather).
- Tied input/output embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTEHRConfig:
    vocab_size: int
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    max_len: int = 1024
    rope_base: float = 10_000.0
    tie_embeddings: bool = True  # False = separate output projection (needed for FedPer)


class LoRALinear(nn.Module):
    """Low-rank adapter wrapper for a frozen Linear layer.

    Output = frozen_linear(x) + (x @ A @ B) * (alpha / rank)
    Only A and B are trained; the base weight is frozen.
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)
        # Create on same device as the base weight (model may already be on GPU)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.randn(base.in_features, rank, device=dev) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, base.out_features, device=dev))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.lora_A @ self.lora_B) * self.scale


def merge_lora(model: "GPTEHR") -> None:
    """Merge LoRA adapters back into the base Linear weights and remove the wrappers.

    After this call the model has no LoRA modules and can be saved/loaded as a
    standard GPTEHR checkpoint. Call before saving the best checkpoint for eval.
    """
    for block in model.blocks:
        attn = block.attn
        for attr in ("qkv", "out"):
            layer = getattr(attn, attr)
            if isinstance(layer, LoRALinear):
                # Merge: W += scale * (A @ B).T  (Linear stores weight as out×in)
                with torch.no_grad():
                    layer.base.weight.add_(
                        (layer.lora_A @ layer.lora_B).t() * layer.scale
                    )
                    layer.base.weight.requires_grad_(True)
                setattr(attn, attr, layer.base)


def apply_lora(model: "GPTEHR", rank: int = 4, alpha: float = 1.0) -> None:
    """Wrap attention qkv and out projections with LoRA adapters and freeze everything else.

    After this call, only LoRA A/B parameters have requires_grad=True.
    """
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # Add LoRA to each attention layer's qkv and out
    for block in model.blocks:
        attn = block.attn
        attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)
        attn.out = LoRALinear(attn.out, rank=rank, alpha=alpha)

    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[lora] applied rank={rank} alpha={alpha}: {n_lora:,} trainable / {n_total:,} total ({100*n_lora/n_total:.1f}%)")


def _apply_rope(x: torch.Tensor, position_ids: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x of shape (B, n_heads, L, d_head) using per-position integer indices."""
    B, H, L, D = x.shape
    # (B, L) -> (B, L, D/2)
    freqs = position_ids.to(inv_freq.dtype).unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    cos = freqs.cos()  # (B, L, D/2)
    sin = freqs.sin()  # (B, L, D/2)
    cos = cos.unsqueeze(1)  # (B, 1, L, D/2)
    sin = sin.unsqueeze(1)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rotated_even = x1 * cos - x2 * sin
    rotated_odd = x1 * sin + x2 * cos
    out = torch.stack((rotated_even, rotated_odd), dim=-1).reshape(B, H, L, D)
    return out


class RoPEMultiheadSelfAttention(nn.Module):
    def __init__(self, cfg: GPTEHRConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout_p = cfg.dropout
        # Registered buffer so inv_freq moves with .to(device) and isn't reallocated per forward.
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, self.d_head, 2, dtype=torch.float32) / self.d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x: torch.Tensor,                     # (B, L, D)
        position_ids: torch.Tensor,          # (B, L) int64
        attn_mask_2d: torch.Tensor,          # (B, L, L) bool, True = attend
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, d_head)

        q = _apply_rope(q.float(), position_ids, self.inv_freq).to(x.dtype)
        k = _apply_rope(k.float(), position_ids, self.inv_freq).to(x.dtype)

        # sdpa expects (B, H, L, L) bool mask; broadcast (B, L, L) -> (B, 1, L, L)
        mask = attn_mask_2d.unsqueeze(1)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, cfg: GPTEHRConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: GPTEHRConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = RoPEMultiheadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x, position_ids, attn_mask_2d):
        x = x + self.attn(self.ln1(x), position_ids, attn_mask_2d)
        x = x + self.ffn(self.ln2(x))
        return x


class GPTEHR(nn.Module):
    def __init__(self, cfg: GPTEHRConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # Output projection: tied to tok_emb by default, separate when tie_embeddings=False
        # (FedPer needs separate output_proj so it can be kept per-site while averaging
        # the shared transformer blocks + input embedding)
        if not cfg.tie_embeddings:
            self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        token_ids: torch.Tensor,         # (B, L)
        position_ids: torch.Tensor,      # (B, L)
        attn_mask_2d: torch.Tensor,      # (B, L, L) bool
    ) -> torch.Tensor:
        x = self.tok_emb(token_ids)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, position_ids, attn_mask_2d)
        x = self.ln_f(x)
        if self.cfg.tie_embeddings:
            logits = x @ self.tok_emb.weight.t()
        else:
            logits = self.output_proj(x)
        return logits

    def sep_loss(
        self,
        logits: torch.Tensor,               # (B, L, V)
        sep_batch_index: torch.Tensor,      # (S,)
        sep_flat_positions: torch.Tensor,   # (S,)
        targets: torch.Tensor,              # (S, V)
        weights: torch.Tensor,              # (S, V)
    ) -> tuple[torch.Tensor, dict]:
        if sep_batch_index.numel() == 0:
            zero = logits.new_zeros(())
            return zero, {"n_seps": 0}

        sep_logits = logits[sep_batch_index, sep_flat_positions]  # (S, V)
        loss_per = F.binary_cross_entropy_with_logits(
            sep_logits, targets, reduction="none"
        )
        weighted = loss_per * weights
        # Normalize by the count of ACTIVE (sep, token) pairs — i.e. the number of
        # positions with non-zero weight. Previous versions divided by the full (S * V),
        # which over-counts the denominator whenever tokens are explicitly masked out of
        # the loss (e.g. LAB tokens via `loss.exclude_prefixes`). Dividing by the full
        # V in that case dilutes the effective per-DX loss proportionally to the number
        # of masked positions and shrinks gradients by the same factor, making the
        # LR/warmup schedule no longer comparable across runs with different vocabularies.
        # Using `weights > 0` as the active-position count keeps per-active loss invariant
        # under exclusion, so runs with and without masked tokens are directly comparable.
        active = weights.gt(0).sum()
        denom = active.clamp(min=1)
        loss = weighted.sum() / denom
        return loss, {"n_seps": int(sep_batch_index.numel())}


def build_model(vocab_size: int, **overrides) -> GPTEHR:
    cfg = GPTEHRConfig(vocab_size=vocab_size, **overrides)
    return GPTEHR(cfg)
