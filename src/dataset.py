"""Dataset and collator for the POC foundation model.

Each sample is one patient sequence produced by `tokenize_events.py`. The
collator builds per-batch:

- `token_ids`         (B, L)     padded int64
- `position_ids`      (B, L)     padded int64; <sep>_i carries day(i+1)
- `attention_mask_2d` (B, L, L)  bool mask with within-block bidirectional
                                 attention and cross-block causal attention
- `key_padding_mask`  (B, L)     bool, True where pad
- `sep_positions`     list[Tensor[int64]]  per-sample positions of <sep>s
                                           contributing to the loss (all
                                           except the final <sep>, since it
                                           has no i+1 target).
- `sep_batch_index`   (S,)       which batch element each flat sep belongs to
- `sep_flat_positions`(S,)       flat sequence position of each scored <sep>
- `targets`           (S, V)     float32 multi-hot of tokens in the target
                                 block (excluding <sep>)
- `weights`           (S, V)     float32 per-token loss weight exp(-δ·r)
                                 where r is repeat count of token k in
                                 visits 1..i for the sep closing visit i.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PAD_ID, BOS_ID, SEP_ID, UNK_ID = 0, 1, 2, 3
NON_EVENT_IDS = {PAD_ID, BOS_ID, SEP_ID, UNK_ID}


class PatientSequenceDataset(Dataset):
    def __init__(self, pkl_path: Path):
        with Path(pkl_path).open("rb") as f:
            self.records: list[dict] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


@dataclass
class CollatorConfig:
    vocab_size: int
    max_len: int = 1024
    delta: float = 0.5
    pad_id: int = PAD_ID
    # Token ids that should appear in the input sequence but NOT be predicted by the
    # loss (e.g. LAB tokens in the input-only-context experiment). The forward pass
    # is unchanged; these positions are masked out of both target and weight so the
    # BCE contribution and its gradient are exactly zero.
    excluded_target_ids: np.ndarray | None = None


class CollateFn:
    def __init__(self, cfg: CollatorConfig):
        self.cfg = cfg

    def __call__(self, batch: list[dict]) -> dict:
        B = len(batch)
        longest = max(len(r["token_ids"]) for r in batch)
        if longest > self.cfg.max_len:
            raise ValueError(
                f"Record length {longest} exceeds collator max_len {self.cfg.max_len}. "
                f"Re-run tokenize_events.py with --max-len <= {self.cfg.max_len}, "
                f"or raise the collator's max_len to match the tokenized data."
            )
        L = longest

        token_ids = np.full((B, L), self.cfg.pad_id, dtype=np.int64)
        position_ids = np.zeros((B, L), dtype=np.int64)
        block_ids = np.full((B, L), -1, dtype=np.int64)
        key_padding_mask = np.ones((B, L), dtype=bool)  # True = padded

        for i, rec in enumerate(batch):
            n = len(rec["token_ids"])
            token_ids[i, :n] = rec["token_ids"]
            position_ids[i, :n] = rec["position_ids"]
            block_ids[i, :n] = rec["block_ids"]
            key_padding_mask[i, :n] = False

        attn_mask = self._build_attention_mask(block_ids, key_padding_mask)

        sep_batch_index: list[int] = []
        sep_flat_positions: list[int] = []
        target_rows: list[np.ndarray] = []
        weight_rows: list[np.ndarray] = []

        V = self.cfg.vocab_size
        delta = self.cfg.delta

        for i, rec in enumerate(batch):
            seps = [p for p in rec["sep_positions"].tolist() if p < L]
            if len(seps) < 2:
                continue

            tokens = rec["token_ids"][:L]
            blocks = rec["block_ids"][:L]
            # Score every <sep> whose target block still exists in the truncated sequence,
            # i.e. all but the final <sep> (since there's no next block after it).
            n_scored = len(seps) - 1
            # Running per-token repeat count across visits. We increment AFTER processing
            # sep_i so weights at sep_i reflect history in visits 1..i (not including i+1).
            repeat_count = np.zeros(V, dtype=np.float32)

            for j in range(n_scored):
                sep_pos = seps[j]
                target_block_id = int(blocks[sep_pos]) + 1
                # Update repeat_count with tokens from block (j+1)-th-in-sequence = current block
                # We do the update first so that weights at sep_j use history through block j,
                # which includes the current block (i.e. "visits 1..i"). This matches the paper's
                # H_{t_i} formulation (history up to and including visit i).
                current_block_id = int(blocks[sep_pos])
                current_mask = (blocks == current_block_id) & (tokens != SEP_ID)
                for tok in np.unique(tokens[current_mask]):
                    tok = int(tok)
                    if tok in NON_EVENT_IDS:
                        continue
                    repeat_count[tok] += 1  # count once per visit containing it

                # Target = tokens present in the NEXT block, excluding non-event ids
                target_mask = (blocks == target_block_id) & (tokens != SEP_ID)
                target_tokens = [int(t) for t in tokens[target_mask] if int(t) not in NON_EVENT_IDS]
                target_vec = np.zeros(V, dtype=np.float32)
                if target_tokens:
                    target_vec[np.array(target_tokens, dtype=np.int64)] = 1.0

                weight_vec = np.exp(-delta * repeat_count).astype(np.float32)

                # Mask out tokens we don't want to predict (e.g. labs used as
                # input-only context). Both target and weight must be zero so the
                # BCE loss and its gradient collapse at those positions.
                if self.cfg.excluded_target_ids is not None and len(self.cfg.excluded_target_ids) > 0:
                    target_vec[self.cfg.excluded_target_ids] = 0.0
                    weight_vec[self.cfg.excluded_target_ids] = 0.0

                sep_batch_index.append(i)
                sep_flat_positions.append(sep_pos)
                target_rows.append(target_vec)
                weight_rows.append(weight_vec)

        if target_rows:
            targets = np.stack(target_rows, axis=0)
            weights = np.stack(weight_rows, axis=0)
        else:
            targets = np.zeros((0, V), dtype=np.float32)
            weights = np.zeros((0, V), dtype=np.float32)

        return {
            "token_ids": torch.from_numpy(token_ids),
            "position_ids": torch.from_numpy(position_ids),
            "attn_mask_2d": torch.from_numpy(attn_mask),
            "key_padding_mask": torch.from_numpy(key_padding_mask),
            "sep_batch_index": torch.tensor(sep_batch_index, dtype=torch.long),
            "sep_flat_positions": torch.tensor(sep_flat_positions, dtype=torch.long),
            "targets": torch.from_numpy(targets),
            "weights": torch.from_numpy(weights),
        }

    @staticmethod
    def _build_attention_mask(block_ids: np.ndarray, key_padding_mask: np.ndarray) -> np.ndarray:
        """Return a (B, L, L) bool mask. True means ALLOWED to attend.

        Within the same non-padded block: bidirectional.
        Across blocks in the same sequence: causal (query block >= key block).
        Padded positions on either side are disallowed everywhere.
        """
        B, L = block_ids.shape
        mask = np.zeros((B, L, L), dtype=bool)
        for b in range(B):
            blk = block_ids[b]
            valid = ~key_padding_mask[b]
            # Query q can attend to key k iff both valid and blk[q] >= blk[k].
            # Inside same block it's allowed in both directions. Since blk is monotone
            # non-decreasing along the sequence and <sep> of block i carries block_id=i,
            # blk[q] >= blk[k] correctly gives within-block bidirectional + cross-block causal.
            allowed = (blk[:, None] >= blk[None, :]) & valid[:, None] & valid[None, :]
            # Padded query rows would otherwise have zero valid keys, which makes SDPA
            # softmax produce NaN for that row. The NaN then propagates through shared
            # weights on backward, corrupting training even though the loss is computed
            # only at non-padded <sep> positions. Force self-attention on the diagonal
            # so every row has at least one allowed key. Non-padded rows are unaffected
            # since they already attend to themselves via blk[q] >= blk[q].
            np.fill_diagonal(allowed, True)
            mask[b] = allowed
        return mask


def make_loader(
    pkl_path: Path,
    vocab_size: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    delta: float = 0.5,
    max_len: int = 1024,
    worker_init_fn=None,
    excluded_target_ids: np.ndarray | None = None,
) -> torch.utils.data.DataLoader:
    ds = PatientSequenceDataset(pkl_path)
    collate = CollateFn(
        CollatorConfig(
            vocab_size=vocab_size,
            max_len=max_len,
            delta=delta,
            excluded_target_ids=excluded_target_ids,
        )
    )
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
