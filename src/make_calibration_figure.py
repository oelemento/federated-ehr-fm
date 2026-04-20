"""Generate Figure 4: calibration (reliability diagrams) for the 3 main strategies
across 5 acute conditions.

Reads data/processed/calib_{strategy}_{seed}.json produced by compute_calibration.py,
averages reliability bins across seeds, and plots one reliability diagram per
condition with 3 methods overlaid.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PROC = REPO / "data" / "processed"
OUT = REPO / "figures"

CONDS = ["aki", "sepsis", "acute_resp_failure", "heart_failure", "acute_mi"]
COND_LABELS = {
    "aki": "AKI",
    "sepsis": "Sepsis",
    "acute_resp_failure": "Acute resp.\nfailure",
    "heart_failure": "Heart\nfailure",
    "acute_mi": "Acute MI",
}

GROUPS = [
    ("Centralized", [PROC / f"calib_centralized_untied_s2026041{s}.json" for s in [0, 1, 2]], "#1f77b4"),
    ("FedAvg (untied)", [PROC / f"calib_untied_fedavg_s2026041{s}.json" for s in [1, 2, 3]], "#2ca02c"),
    ("Ensemble",   [PROC / f"calib_untied_ensemble_s2026041{s}.json" for s in [1, 2, 3]], "#ff7f0e"),
]


def aggregate_bins(paths, cond_name):
    """Average reliability bins across seeds for a given condition."""
    mean_pred_by_bin = [[] for _ in range(10)]
    obs_rate_by_bin = [[] for _ in range(10)]
    total_n_by_bin = np.zeros(10)
    for p in paths:
        d = json.loads(p.read_text())
        for r in d["results"]:
            if r["name"] != cond_name:
                continue
            for i, b in enumerate(r["reliability_bins"]):
                if b["n"] == 0:
                    continue
                mean_pred_by_bin[i].append(b["mean_predicted"])
                obs_rate_by_bin[i].append(b["observed_rate"])
                total_n_by_bin[i] += b["n"]
    points = []
    for i in range(10):
        if len(mean_pred_by_bin[i]) == 0:
            continue
        mp = float(np.mean(mean_pred_by_bin[i]))
        obs = float(np.mean(obs_rate_by_bin[i]))
        points.append((mp, obs, int(total_n_by_bin[i])))
    return points


def compute_summary(paths, cond_name):
    briers, eces = [], []
    for p in paths:
        d = json.loads(p.read_text())
        for r in d["results"]:
            if r["name"] == cond_name:
                briers.append(r["brier"])
                eces.append(r["ece"])
    return float(np.mean(briers)), float(np.mean(eces))


def main():
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    })
    fig, axes = plt.subplots(1, 5, figsize=(22, 5.2))

    # Find per-condition max for consistent x-range
    for ax, cond in zip(axes, CONDS):
        # Diagonal = perfect calibration
        ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, linewidth=1.5,
                label="Perfect calibration" if cond == CONDS[0] else None)

        x_max = 0
        for label, paths, color in GROUPS:
            points = aggregate_bins(paths, cond)
            if not points:
                continue
            xs, ys, ns = zip(*points)
            # Filter bins with very few samples
            xs = np.array(xs); ys = np.array(ys); ns = np.array(ns)
            keep = ns >= 5
            xs, ys, ns = xs[keep], ys[keep], ns[keep]
            ax.plot(xs, ys, marker="o", color=color, lw=2,
                    markersize=8, label=label if cond == CONDS[0] else None)
            x_max = max(x_max, xs.max() if len(xs) else 0)

        # Axis config: zoom to where the data actually lives (acute conditions are rare)
        upper = max(0.1, min(1.0, x_max * 1.2)) if x_max > 0 else 0.3
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("Predicted probability", fontsize=16)
        if cond == CONDS[0]:
            ax.set_ylabel("Observed frequency", fontsize=16)
        ax.set_title(COND_LABELS[cond].replace("\n", " "), fontsize=17, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2, linestyle="--")
        ax.set_aspect("equal", adjustable="box")

        # Show Brier/ECE in panel corner
        text_lines = []
        for label, paths, color in GROUPS:
            brier, ece = compute_summary(paths, cond)
            text_lines.append(f"{label[:10]}: B={brier:.3f}, ECE={ece:.3f}")
        ax.text(0.98, 0.02, "\n".join(text_lines),
                transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
                family="monospace",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9, pad=4))

    # Shared legend
    axes[0].legend(loc="upper left", fontsize=13, frameon=True, framealpha=0.95)

    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.16, wspace=0.28)
    out_path = OUT / "paper_fig4_calibration.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    print(f"wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
