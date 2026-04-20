"""Per-condition zero-shot endpoint evaluation.

Matches the Rajamohan et al. 2025 framing: pick specific disease endpoints,
define a codeset of ICD tokens, and measure whether the foundation model can
predict NEW-ONSET of the condition at the next admission. This shape of
evaluation maps onto real clinical decisions ("will this patient develop X?")
rather than "predict all tokens in the next visit."

For each condition:
    label    : 1 iff any token in the condition codeset appears in visit N
               AND no token in the codeset appeared in visits 1..N-1
    score    : sum of sigmoid probabilities across the condition's token ids
               at the cut <sep> position (NYU-style aggregation)
    metrics  : AUROC, AUPRC, and a 1000-sample bootstrap 95% CI for each

Patients where the condition was already present in their history are excluded
(so we only score patients at risk, matching the NYU protocol).

Usage
-----
    PYTHONPATH=src python3.11 src/evaluate_conditions.py \
        --checkpoint checkpoints/best_labs_ctx_h1.pt \
        --test-path data/processed/test.pkl \
        --vocab-path data/processed/vocab.json \
        --out-json data/processed/condition_eval.json \
        --figure-path figures/condition_eval.png
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
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate import EvalInstance, build_eval_instances, build_2level_mask
from model import GPTEHR, build_model


# Codeset definitions targeting MIMIC-IV-appropriate acute endpoints. Each
# condition maps to a list of vocab token string prefixes. Tokens whose string
# starts with any of the prefixes are members of the codeset. The training
# pipeline collapses ICD codes to 3-char, so a prefix like "DX10:N17" matches
# exactly one vocab token representing the entire N17.x family.
#
# Why these five (and not dementia / type-2 diabetes / CKD): MIMIC-IV is a
# critical-care inpatient dataset, so chronic outpatient-detected conditions
# aren't a natural fit. The conditions below are acute, common in hospitalized
# populations, and strongly tied to lab trajectories — which is where we'd
# expect a foundation model with lab context to pull ahead of trivial baselines.
CONDITION_CODES: dict[str, list[str]] = {
    "aki":                 ["DX10:N17", "DX9:584"],                 # acute kidney injury
    "sepsis":              ["DX10:A40", "DX10:A41", "DX9:038"],     # bacterial sepsis
    "acute_resp_failure":  ["DX10:J96"],                            # acute respiratory failure
    "heart_failure":       ["DX10:I50", "DX9:428"],                 # heart failure (decompensation)
    "acute_mi":            ["DX10:I21", "DX9:410"],                 # acute myocardial infarction
}


@dataclass
class ConditionResult:
    name: str
    n_eligible: int
    n_positive: int
    auroc: float
    auroc_ci: tuple[float, float]
    auprc: float
    auprc_ci: tuple[float, float]
    prevalence: float


def resolve_codeset(vocab: dict[str, int], prefixes: list[str]) -> list[int]:
    """Return the sorted list of vocab token ids whose string starts with any prefix."""
    hits = []
    for tok, tid in vocab.items():
        for p in prefixes:
            if tok.startswith(p):
                hits.append(tid)
                break
    return sorted(hits)


def bootstrap_ci(
    y_true: np.ndarray, y_score: np.ndarray, metric_fn, n_boot: int = 1000,
    seed: int = 20260411,
) -> tuple[float, float]:
    """Bootstrap 95% CI for a binary metric (AUROC or AUPRC)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if y_true.sum() == 0 or y_true.sum() == n:
        return (float("nan"), float("nan"))
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        samples.append(metric_fn(yt, y_score[idx]))
    if not samples:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return (float(lo), float(hi))


def score_all_instances(
    model: GPTEHR,
    instances: list[EvalInstance],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run the model ONCE across all instances, return (N, V) sigmoid probabilities
    at each instance's cut <sep>. Per-condition aggregation is done in numpy after
    this pass — avoids doing 5× forward passes for 5 conditions."""
    N = len(instances)
    V = int(model.tok_emb.weight.shape[0])
    all_probs = np.zeros((N, V), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            batch = instances[start : start + batch_size]
            L = max(len(x.token_ids) for x in batch)
            B = len(batch)
            token_ids = torch.zeros((B, L), dtype=torch.long)
            position_ids = torch.zeros((B, L), dtype=torch.long)
            attn = torch.zeros((B, L, L), dtype=torch.bool)
            sep_idx = torch.zeros(B, dtype=torch.long)
            for i, inst in enumerate(batch):
                n = len(inst.token_ids)
                token_ids[i, :n] = torch.from_numpy(inst.token_ids.astype(np.int64))
                position_ids[i, :n] = torch.from_numpy(inst.position_ids.astype(np.int64))
                attn[i, :n, :n] = build_2level_mask(inst.block_ids)
                sep_idx[i] = inst.cut_sep_pos
            diag = torch.arange(L)
            attn[:, diag, diag] = True
            token_ids = token_ids.to(device)
            position_ids = position_ids.to(device)
            attn = attn.to(device)
            sep_idx = sep_idx.to(device)
            logits = model(token_ids, position_ids, attn)  # (B, L, V)
            picked = logits[torch.arange(B, device=device), sep_idx]  # (B, V)
            probs = torch.sigmoid(picked).float().cpu().numpy()       # (B, V)
            all_probs[start : start + B] = probs
            if (start // batch_size) % 20 == 0:
                print(f"  [condeval] scored {min(start + B, N):,}/{N:,}", flush=True)
    return all_probs


def condition_labels_and_scores(
    instances: list[EvalInstance],
    probs: np.ndarray,
    codeset_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Given pre-computed per-instance probabilities and a condition codeset, return
    (labels, scores, n_excluded) over eligible instances (those without the condition
    in history). The score is the sum of sigmoid probabilities across codeset tokens."""
    codeset = set(codeset_ids)
    codeset_arr = np.asarray(codeset_ids, dtype=np.int64)
    labels: list[int] = []
    scores: list[float] = []
    n_excluded = 0
    for i, inst in enumerate(instances):
        if codeset & inst.history_tokens:
            n_excluded += 1
            continue
        label = int(bool(codeset & inst.target_dx_ids))
        score = float(probs[i, codeset_arr].sum())
        labels.append(label)
        scores.append(score)
    return (
        np.array(labels, dtype=np.uint8),
        np.array(scores, dtype=np.float32),
        n_excluded,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-path", type=Path, default=Path("data/processed/test.pkl"))
    parser.add_argument("--vocab-path", type=Path, default=Path("data/processed/vocab.json"))
    parser.add_argument("--out-json", type=Path, default=Path("data/processed/condition_eval.json"))
    parser.add_argument("--figure-path", type=Path, default=Path("figures/condition_eval.png"))
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    print(f"[condeval] loading vocab from {args.vocab_path}")
    vocab: dict[str, int] = json.loads(args.vocab_path.read_text())
    V = len(vocab)
    dx_id_set = {int(i) for tok, i in vocab.items() if tok.startswith("DX")}

    print(f"[condeval] loading checkpoint {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else
                          ("mps" if torch.backends.mps.is_available() else "cpu"))
    model = build_model(V, **cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[condeval] device={device}  model params={model.num_parameters()/1e6:.2f}M")

    print(f"[condeval] loading test set from {args.test_path}")
    with args.test_path.open("rb") as f:
        test_records = pickle.load(f)
    instances = build_eval_instances(test_records, dx_id_set)
    print(f"[condeval] {len(instances):,} eval instances (patients with >= 2 admissions and DX in target visit)")

    print(f"[condeval] running forward pass across all {len(instances):,} instances (single pass, reused across conditions)")
    probs = score_all_instances(model, instances, device)
    print(f"[condeval] logits shape: {probs.shape}")

    results: list[ConditionResult] = []
    for name, prefixes in CONDITION_CODES.items():
        codeset_ids = resolve_codeset(vocab, prefixes)
        if not codeset_ids:
            print(f"[condeval] {name:20s}: NO tokens matching {prefixes} in vocab — skipping")
            continue
        cs_tokens = [k for k, v in vocab.items() if v in set(codeset_ids)]
        print(f"[condeval] {name:20s}: codeset has {len(codeset_ids)} vocab tokens {cs_tokens}")
        y_true, y_score, n_excluded = condition_labels_and_scores(instances, probs, codeset_ids)
        n_eligible = len(y_true)
        n_positive = int(y_true.sum())
        prevalence = n_positive / max(1, n_eligible)
        if n_positive == 0:
            print(f"[condeval] {name:20s}: no new-onset cases among {n_eligible} eligible — skipping")
            continue

        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
        auroc_ci = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot=args.n_boot)
        auprc_ci = bootstrap_ci(y_true, y_score, average_precision_score, n_boot=args.n_boot)

        r = ConditionResult(
            name=name,
            n_eligible=n_eligible,
            n_positive=n_positive,
            auroc=auroc,
            auroc_ci=auroc_ci,
            auprc=auprc,
            auprc_ci=auprc_ci,
            prevalence=prevalence,
        )
        results.append(r)

        print(
            f"[condeval] {name:13s}: n_eligible={n_eligible:,}  n_positive={n_positive} "
            f"({100*prevalence:.2f}%)  AUROC={auroc:.3f} [{auroc_ci[0]:.3f},{auroc_ci[1]:.3f}]  "
            f"AUPRC={auprc:.3f} [{auprc_ci[0]:.3f},{auprc_ci[1]:.3f}]"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(args.checkpoint),
        "results": [
            {
                "name": r.name,
                "n_eligible": r.n_eligible,
                "n_positive": r.n_positive,
                "prevalence": r.prevalence,
                "auroc": r.auroc,
                "auroc_ci": list(r.auroc_ci),
                "auprc": r.auprc,
                "auprc_ci": list(r.auprc_ci),
            }
            for r in results
        ],
    }
    args.out_json.write_text(json.dumps(summary, indent=2))
    print(f"[condeval] wrote {args.out_json}")

    # Figure: AUROC and AUPRC with CIs across conditions
    if results:
        args.figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        names = [r.name for r in results]
        x = np.arange(len(names))
        aurocs = [r.auroc for r in results]
        auroc_err = [
            [r.auroc - r.auroc_ci[0] for r in results],
            [r.auroc_ci[1] - r.auroc for r in results],
        ]
        auprcs = [r.auprc for r in results]
        auprc_err = [
            [r.auprc - r.auprc_ci[0] for r in results],
            [r.auprc_ci[1] - r.auprc for r in results],
        ]

        axes[0].bar(x, aurocs, yerr=auroc_err, capsize=4, color="C0")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(names, rotation=30, ha="right")
        axes[0].set_ylabel("AUROC")
        axes[0].set_ylim(0.5, 1.0)
        axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_title("AUROC (95% bootstrap CI)")
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(x, auprcs, yerr=auprc_err, capsize=4, color="C2")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=30, ha="right")
        axes[1].set_ylabel("AUPRC")
        axes[1].set_ylim(0, 1.0)
        axes[1].set_title("AUPRC (95% bootstrap CI)")
        axes[1].grid(alpha=0.3, axis="y")

        fig.suptitle("GPT-EHR zero-shot new-onset prediction — MIMIC-IV", fontsize=12)
        fig.tight_layout()
        fig.savefig(args.figure_path, dpi=150)
        print(f"[condeval] wrote {args.figure_path}")


if __name__ == "__main__":
    main()
