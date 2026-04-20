"""Generate the final POC summary figure for the GPT-EHR MIMIC-IV result.

Three panels:
    A. Multi-label next-admission DX forecasting — overall + new-onset AUPRC,
       comparing the foundation model (mean ± std across 3 seeds) against
       marginal, repeat, and linear bag-of-DX baselines.
    B. Per-condition zero-shot prediction — AUROC with 95% bootstrap CIs for
       AKI, sepsis, acute respiratory failure, heart failure, acute MI.
    C. Per-condition AUPRC with 95% bootstrap CIs.

Outputs `figures/poc_summary.png` (PNG, 12 in × 8 in, 150 dpi).

Inputs:
    data/processed/eval_results_abnormal_s{0,1,2}.json   3 seeds of foundation eval
    data/processed/baseline_linear_results.json          linear bag-of-DX baseline
    data/processed/condition_eval_s1.json                per-condition eval
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "processed"
OUT = REPO / "figures" / "poc_summary.png"


def load_seed_runs(paths: list[Path]) -> dict[str, dict[str, np.ndarray]]:
    """Stack per-seed AUPRCs into arrays so we can compute mean / std."""
    overall_auprc, new_onset_auprc = [], []
    overall_f1, new_onset_f1 = [], []
    for p in paths:
        d = json.loads(p.read_text())
        overall_auprc.append(d["foundation"]["overall"]["auprc"])
        new_onset_auprc.append(d["foundation"]["new_onset"]["auprc"])
        overall_f1.append(d["foundation"]["overall"]["best_f1"])
        new_onset_f1.append(d["foundation"]["new_onset"]["best_f1"])
    return {
        "overall_auprc": np.asarray(overall_auprc),
        "new_onset_auprc": np.asarray(new_onset_auprc),
        "overall_f1": np.asarray(overall_f1),
        "new_onset_f1": np.asarray(new_onset_f1),
    }


def main() -> None:
    seed_paths = [
        DATA / "eval_results_abnormal_s0.json",
        DATA / "eval_results_abnormal_s1.json",
        DATA / "eval_results_abnormal_s2.json",
    ]
    seed_data = load_seed_runs(seed_paths)
    foundation_overall = seed_data["overall_auprc"]
    foundation_new = seed_data["new_onset_auprc"]
    print(
        f"foundation overall AUPRC = {foundation_overall.mean():.4f} ± {foundation_overall.std(ddof=1):.4f}  "
        f"new-onset AUPRC = {foundation_new.mean():.4f} ± {foundation_new.std(ddof=1):.4f}"
    )

    baseline_linear = json.loads((DATA / "baseline_linear_results.json").read_text())["baseline_linear"]
    # Marginal/repeat numbers are identical across seeds; pull from any one
    abnormal_run = json.loads((DATA / "eval_results_abnormal_s0.json").read_text())

    cond = json.loads((DATA / "condition_eval_s1.json").read_text())["results"]

    # Build comparison data: (label, mean_overall, err_overall, mean_new, err_new, color)
    methods = [
        (
            "Foundation\n(3 seeds, mean±std)",
            float(foundation_overall.mean()),
            float(foundation_overall.std(ddof=1)),
            float(foundation_new.mean()),
            float(foundation_new.std(ddof=1)),
            "C0",
        ),
        (
            "Linear bag-of-DX\n(BCE, 10 epochs)",
            baseline_linear["overall"]["auprc"],
            0.0,
            baseline_linear["new_onset"]["auprc"],
            0.0,
            "C2",
        ),
        (
            "Repeat-history\n(prior visit DX)",
            abnormal_run["repeat"]["overall"]["auprc"],
            0.0,
            abnormal_run["repeat"]["new_onset"]["auprc"],
            0.0,
            "C3",
        ),
        (
            "Marginal\n(global frequency)",
            abnormal_run["marginal"]["overall"]["auprc"],
            0.0,
            abnormal_run["marginal"]["new_onset"]["auprc"],
            0.0,
            "C7",
        ),
    ]

    fig = plt.figure(figsize=(12, 8.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_overall = fig.add_subplot(gs[0, 0:2])
    ax_cond_roc = fig.add_subplot(gs[1, 0])
    ax_cond_pr = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[0:2, 2])
    ax_text.axis("off")

    # Panel A — multi-label next-admission AUPRC
    n_methods = len(methods)
    bar_width = 0.36
    x = np.arange(n_methods)
    overall_vals = [m[1] for m in methods]
    overall_errs = [m[2] for m in methods]
    new_vals = [m[3] for m in methods]
    new_errs = [m[4] for m in methods]
    colors = [m[5] for m in methods]
    labels = [m[0] for m in methods]

    bars1 = ax_overall.bar(x - bar_width/2, overall_vals, bar_width, yerr=overall_errs,
                            capsize=4, color=colors, label="Overall AUPRC", alpha=0.95)
    bars2 = ax_overall.bar(x + bar_width/2, new_vals, bar_width, yerr=new_errs,
                            capsize=4, color=colors, hatch="//", edgecolor="black",
                            linewidth=0.4, alpha=0.6, label="New-onset AUPRC")
    ax_overall.set_xticks(x)
    ax_overall.set_xticklabels(labels, fontsize=8)
    ax_overall.set_ylabel("AUPRC")
    ax_overall.set_title(
        "A. Next-admission diagnosis forecasting\n"
        "MIMIC-IV, 15,012 test patients, 2,590 DX tokens",
        fontsize=11,
    )
    ax_overall.set_ylim(0, 0.45)
    ax_overall.grid(alpha=0.3, axis="y")
    # Annotate the foundation new-onset bar with the gap to linear
    fnd_new = methods[0][3]
    lin_new = methods[1][3]
    ratio = fnd_new / max(lin_new, 1e-9)
    ax_overall.annotate(
        f"{ratio:.1f}× linear",
        xy=(x[0] + bar_width/2, fnd_new),
        xytext=(x[0] + bar_width/2 + 0.05, fnd_new + 0.04),
        fontsize=8, color="C0",
        arrowprops={"arrowstyle": "->", "color": "C0", "lw": 0.7},
    )
    ax_overall.legend(loc="upper right", fontsize=8)

    # Panel B — per-condition AUROC
    cond_names = [c["name"].replace("_", " ").title() for c in cond]
    aurocs = [c["auroc"] for c in cond]
    auroc_lo = [c["auroc"] - c["auroc_ci"][0] for c in cond]
    auroc_hi = [c["auroc_ci"][1] - c["auroc"] for c in cond]
    cond_x = np.arange(len(cond))
    ax_cond_roc.bar(cond_x, aurocs, yerr=[auroc_lo, auroc_hi], capsize=4,
                    color="C0", alpha=0.85)
    ax_cond_roc.axhline(0.5, color="gray", linestyle="--", alpha=0.5,
                        label="random")
    ax_cond_roc.set_xticks(cond_x)
    ax_cond_roc.set_xticklabels(cond_names, rotation=30, ha="right", fontsize=8)
    ax_cond_roc.set_ylabel("AUROC")
    ax_cond_roc.set_ylim(0.5, 1.0)
    ax_cond_roc.set_title(
        "B. Per-condition AUROC (95% bootstrap CI)",
        fontsize=11,
    )
    ax_cond_roc.grid(alpha=0.3, axis="y")
    ax_cond_roc.legend(loc="upper right", fontsize=8)

    # Panel C — per-condition AUPRC
    auprcs = [c["auprc"] for c in cond]
    auprc_lo = [c["auprc"] - c["auprc_ci"][0] for c in cond]
    auprc_hi = [c["auprc_ci"][1] - c["auprc"] for c in cond]
    ax_cond_pr.bar(cond_x, auprcs, yerr=[auprc_lo, auprc_hi], capsize=4,
                   color="C2", alpha=0.85)
    # Show prevalence as a horizontal dashed line per bar
    prevs = [c["prevalence"] for c in cond]
    for xi, p in zip(cond_x, prevs):
        ax_cond_pr.plot([xi - 0.4, xi + 0.4], [p, p],
                        color="gray", linestyle="--", linewidth=1.0)
    ax_cond_pr.set_xticks(cond_x)
    ax_cond_pr.set_xticklabels(cond_names, rotation=30, ha="right", fontsize=8)
    ax_cond_pr.set_ylabel("AUPRC")
    ax_cond_pr.set_ylim(0, max(0.35, max(auprcs) * 1.2))
    ax_cond_pr.set_title(
        "C. Per-condition AUPRC (gray = prevalence)",
        fontsize=11,
    )
    ax_cond_pr.grid(alpha=0.3, axis="y")

    # Side panel: text summary
    bullets = (
        "GPT-EHR on MIMIC-IV — POC summary\n\n"
        f"• 100,163 patients with ≥2 admissions\n"
        f"  (70k train / 15k val / 15k test)\n\n"
        f"• 28M parameter decoder\n"
        f"  (8 layers, 8 heads, d=512)\n"
        f"  RoPE on inter-visit days,\n"
        f"  bidirectional within visit,\n"
        f"  repeat-token decay BCE\n\n"
        f"• Vocab 6,256 tokens\n"
        f"  (DX 2,590 + PX 1,328 +\n"
        f"   MED 809 + LAB 1,525)\n\n"
        f"• Cluster: 1× NVIDIA A40\n"
        f"  3 epochs, ~17 min wall\n\n"
        f"• Foundation model is\n"
        f"  reproducible across seeds\n"
        f"  (overall AUPRC σ = 0.003)\n\n"
        f"• Beats linear bag-of-DX by\n"
        f"  {ratio:.1f}× on new-onset\n"
        f"  diagnoses ({lin_new:.3f} → {fnd_new:.3f})\n\n"
        f"• Per-condition AUROC 0.77–0.84\n"
        f"  zero-shot (no fine-tuning)\n"
        f"  matches NYU 1.29M-patient\n"
        f"  GPT-EHR (Rajamohan 2025)\n"
        f"  on AUROC headline numbers"
    )
    ax_text.text(0.02, 0.98, bullets, va="top", ha="left",
                  fontsize=9, family="monospace",
                  bbox={"boxstyle": "round,pad=0.6",
                        "facecolor": "#f5f5f5",
                        "edgecolor": "#cccccc"})

    fig.suptitle(
        "GPT-EHR foundation model on MIMIC-IV — proof of concept",
        fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
