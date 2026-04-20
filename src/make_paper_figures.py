"""Generate publication figures and tables for the federated EHR paper.

All reported numbers are loaded from canonical evaluation JSONs in
`data/processed/`. No AUPRC/AUROC values are hard-coded — modify the
artifact registry below to change which JSON produces which bar.

Figures:
    2. Main AUPRC comparison across all methods (with error bars where
       multi-seed data is available).
    3. Per-condition AUROC (centralized vs ensemble vs FedPer).
    4. Ensemble robustness vs Dirichlet alpha.

Tables (printed to stdout as markdown):
    1. Full methods comparison.
    2. Per-condition AUROC.
    3. Alpha sweep summary.

Error bars:
    - Centralized: std across 3 independent seeds.
    - Ensemble, per-site mean: std across 3 per-split means at alpha=0.5
      (computed per-split first, then reduced). This matches the caption.
      NOT the std across all 15 site×seed values (which would conflate
      site-size variability with split variability).
    - FedPer, FedPer+LoRA, FedProx: single-run (pending multi-seed).
    - Linear, repeat, marginal: deterministic.

Run: python3.11 src/make_paper_figures.py
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
OUT.mkdir(exist_ok=True)

ALPHAS = [0.5, 1.0, 2.0]
SEEDS = [20260411, 20260412, 20260413]
HEADLINE_ALPHA = 0.5
CENTRALIZED_SEEDS = ["s0", "s1", "s2"]

CONDITIONS_KEY = ["aki", "sepsis", "acute_resp_failure", "heart_failure", "acute_mi"]
CONDITIONS_LABEL = ["AKI", "Sepsis", "Acute\nresp fail", "Heart\nfailure", "Acute MI"]
CONDITIONS_FULL = [
    ("Acute kidney injury", "9.3%"),
    ("Sepsis", "4.9%"),
    ("Acute respiratory failure", "4.6%"),
    ("Heart failure", "5.1%"),
    ("Acute myocardial infarction", "2.2%"),
]


# ---------------------------------------------------------------------------
# Artifact registry — all paths relative to data/processed/
# ---------------------------------------------------------------------------

ART = {
    "centralized":        [PROC / f"eval_results_abnormal_{s}.json" for s in CENTRALIZED_SEEDS],
    "centralized_tied":   [PROC / f"eval_results_abnormal_{s}.json" for s in CENTRALIZED_SEEDS],
    "centralized_untied": [PROC / f"eval_untied_centralized_s{s}.json" for s in ["20260410", "20260411", "20260412"]],
    "linear":        PROC / "baseline_linear_results.json",
    "marginal":      PROC / "eval_results_abnormal_s1.json",  # deterministic — any seed OK
    "repeat":        PROC / "eval_results_abnormal_s1.json",
    # FedProx (best of 6 hyperparameter configurations tested — full_epoch variant).
    # Other configs ranged 0.0812–0.0856, all within marginal-baseline noise.
    "fedprox":       PROC / "eval_results_fed_fullepoch.json",
    # FedPer / FedPer+LoRA: multi-seed reruns from sweep 2806532. Loader uses
    # whatever subset of seeds is available; falls back to legacy single-seed
    # artifact if none of the sweep seeds are present yet.
    "fedper":             [PROC / f"eval_fedper_a0.5_s{s}.json" for s in SEEDS],
    "fedper_lora":        [PROC / f"eval_fedper_lora_a0.5_s{s}.json" for s in SEEDS],
    "fedper_legacy":      PROC / "eval_results_fed_fedper_v2.json",
    "fedper_lora_legacy": PROC / "eval_results_fed_fedper_lora.json",
    "sweep":         {(a, s): PROC / f"eval_sweep_a{a}_s{s}.json"
                      for a in ALPHAS for s in SEEDS},
    "cond_centralized":   [PROC / f"condition_eval_{s}.json" for s in CENTRALIZED_SEEDS],
    "cond_fedper":        [PROC / f"eval_cond_fedper_a0.5_s{s}.json" for s in SEEDS],
    "cond_fedper_legacy": PROC / "condition_eval_fedper.json",
    # Personalized eval (head-ensemble deployment)
    "fedper_pers":        [PROC / f"eval_fedper_personalized_a0.5_s{s}.json" for s in SEEDS],
    "fedper_lora_pers":   [PROC / f"eval_fedper_lora_personalized_a0.5_s{s}.json" for s in SEEDS],
    "cond_fedper_pers":   [PROC / f"eval_cond_fedper_personalized_a0.5_s{s}.json" for s in SEEDS],
    "cond_fedper_lora_pers": [PROC / f"eval_cond_fedper_lora_personalized_a0.5_s{s}.json" for s in SEEDS],
    # Untied control experiment results
    "untied_fedavg":      [PROC / f"eval_untied_fedavg_a0.5_s{s}.json" for s in SEEDS],
    "untied_fedprox":     [PROC / f"eval_untied_fedprox_a0.5_s{s}.json" for s in SEEDS],
    "untied_ensemble":    [PROC / f"eval_untied_ensemble_s{s}.json" for s in SEEDS],
    # Matched-config FedAvg tied (5 sites, alpha=0.5, 100 local steps, 20 rounds,
    # seed 20260411). The previous entry here pointed at a FedProx file, which
    # produced the incorrect "FedAvg (tied) = 0.082" failure number used in
    # earlier drafts. At matched hyperparameters FedAvg tied does NOT fail.
    "tied_fedavg":        PROC / "eval_fedavg_tied_inrnd_s20260411.json",
    "cond_untied_fedavg": [PROC / f"eval_cond_untied_fedavg_a0.5_s{s}.json" for s in SEEDS],
}


# ---------------------------------------------------------------------------
# Helpers: load JSONs and enforce denominator consistency
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical artifact: {path}")
    return json.loads(path.read_text())


def _check_denominators(records: dict[str, dict]) -> None:
    """Verify that all methods share the same (n_pos, n_total) for overall
    and new_onset separately — guards against mixing evaluation runs done
    at different times with different cohort filters."""
    ref: dict[str, tuple[int, int]] = {}
    for name, payload in records.items():
        for bucket in ("overall", "new_onset"):
            key = (payload[bucket]["n_pos"], payload[bucket]["n_total"])
            if bucket not in ref:
                ref[bucket] = key
            elif ref[bucket] != key:
                raise AssertionError(
                    f"Denominator mismatch for '{name}' [{bucket}]: "
                    f"expected {ref[bucket]}, got {key}. "
                    "Mixing incompatible evaluation runs is forbidden."
                )


def _centralized_stats() -> dict:
    vals_overall, vals_new = [], []
    for path in ART["centralized"]:
        d = _load_json(path)["foundation"]
        vals_overall.append(d["overall"]["auprc"])
        vals_new.append(d["new_onset"]["auprc"])
    return {
        "overall": (float(np.mean(vals_overall)), float(np.std(vals_overall, ddof=1))),
        "new_onset": (float(np.mean(vals_new)), float(np.std(vals_new, ddof=1))),
    }


def _sweep_stats(alpha: float) -> dict:
    """Aggregate sweep at a given alpha: per-split mean first, then std
    across the 3 split-level means. This is the honest per-split SD."""
    ens_overall, ens_new = [], []
    persite_split_overall, persite_split_new = [], []
    cond = {c: [] for c in CONDITIONS_KEY}
    for s in SEEDS:
        d = _load_json(ART["sweep"][(alpha, s)])
        ens_overall.append(d["ensemble"]["overall"]["auprc"])
        ens_new.append(d["ensemble"]["new_onset"]["auprc"])
        site_ov = [x["overall_auprc"] for x in d["per_site"]]
        site_no = [x["new_onset_auprc"] for x in d["per_site"]]
        persite_split_overall.append(float(np.mean(site_ov)))
        persite_split_new.append(float(np.mean(site_no)))
        for c in CONDITIONS_KEY:
            cond[c].append(d["ensemble"]["per_condition_auroc"][c]["auroc"])
    return {
        "ens_overall": (float(np.mean(ens_overall)), float(np.std(ens_overall, ddof=1))),
        "ens_new":     (float(np.mean(ens_new)),     float(np.std(ens_new, ddof=1))),
        "persite_overall": (float(np.mean(persite_split_overall)),
                            float(np.std(persite_split_overall, ddof=1))),
        "persite_new":     (float(np.mean(persite_split_new)),
                            float(np.std(persite_split_new, ddof=1))),
        "cond": {c: (float(np.mean(v)), float(np.std(v, ddof=1)))
                 for c, v in cond.items()},
    }


def _single_run(path: Path, key: str = "foundation") -> dict:
    d = _load_json(path)[key]
    return {
        "overall": (d["overall"]["auprc"], 0.0),
        "new_onset": (d["new_onset"]["auprc"], 0.0),
    }


def _multi_seed_or_legacy(paths: list[Path], legacy: Path, key: str = "foundation") -> dict:
    """Aggregate across available multi-seed JSONs. Falls back to a single legacy
    JSON if none of the paths exist. Returns (mean, SD) tuples per bucket.
    n_seeds is stored under key "_n_seeds" for caption labeling."""
    present = [p for p in paths if p.exists()]
    if not present:
        d = _load_json(legacy)[key]
        return {
            "overall": (d["overall"]["auprc"], 0.0),
            "new_onset": (d["new_onset"]["auprc"], 0.0),
            "_n_seeds": 1,
            "_legacy": True,
        }
    overall, new_onset = [], []
    for p in present:
        d = _load_json(p)[key]
        overall.append(d["overall"]["auprc"])
        new_onset.append(d["new_onset"]["auprc"])
    n = len(present)
    sd_ov = float(np.std(overall, ddof=1)) if n >= 2 else 0.0
    sd_no = float(np.std(new_onset, ddof=1)) if n >= 2 else 0.0
    return {
        "overall": (float(np.mean(overall)), sd_ov),
        "new_onset": (float(np.mean(new_onset)), sd_no),
        "_n_seeds": n,
        "_legacy": False,
    }


def _cond_multi_seed_or_legacy(paths: list[Path], legacy: Path) -> dict:
    """Per-condition AUROC: multi-seed mean across available JSONs, falls back
    to legacy single-seed if none of the paths exist."""
    present = [p for p in paths if p.exists()]
    if not present:
        d = _load_json(legacy)
        return {r["name"]: (r["auroc"], 0.0, 1) for r in d["results"]}
    rows = {c: [] for c in CONDITIONS_KEY}
    for p in present:
        d = _load_json(p)
        for r in d["results"]:
            rows[r["name"]].append(r["auroc"])
    n = len(present)
    return {
        c: (float(np.mean(v)), float(np.std(v, ddof=1)) if n >= 2 else 0.0, n)
        for c, v in rows.items() if v
    }


def _linear() -> dict:
    d = _load_json(ART["linear"])["baseline_linear"]
    return {
        "overall": (d["overall"]["auprc"], 0.0),
        "new_onset": (d["new_onset"]["auprc"], 0.0),
    }


def _cond_centralized() -> dict:
    """Per-condition AUROC for centralized — mean across available seeds."""
    per_seed = []
    for path in ART["cond_centralized"]:
        if not path.exists():
            continue
        d = _load_json(path)
        per_seed.append({r["name"]: r["auroc"] for r in d["results"]})
    if not per_seed:
        raise RuntimeError("No centralized condition_eval_*.json artifacts found")
    n_seeds = len(per_seed)
    agg = {}
    for c in CONDITIONS_KEY:
        vals = [s[c] for s in per_seed if c in s]
        if not vals:
            continue
        if len(vals) > 1:
            agg[c] = (float(np.mean(vals)), float(np.std(vals, ddof=1)), n_seeds)
        else:
            agg[c] = (float(vals[0]), 0.0, 1)
    return agg


def _personalized_stats(paths: list[Path], key: str = "head_ensemble") -> dict:
    """Aggregate head-ensemble (or per_site_mean_personalized) stats from
    personalized eval JSONs. Returns (mean, SD) tuples per bucket."""
    present = [p for p in paths if p.exists()]
    if not present:
        return None
    overall, new_onset = [], []
    for p in present:
        d = _load_json(p)
        he = d[key]
        if key == "head_ensemble":
            overall.append(he["overall"]["auprc"])
            new_onset.append(he["new_onset"]["auprc"])
        else:
            overall.append(he["overall_auprc_mean"])
            new_onset.append(he["new_onset_auprc_mean"])
    n = len(present)
    return {
        "overall": (float(np.mean(overall)),
                    float(np.std(overall, ddof=1)) if n >= 2 else 0.0),
        "new_onset": (float(np.mean(new_onset)),
                      float(np.std(new_onset, ddof=1)) if n >= 2 else 0.0),
        "_n_seeds": n,
    }


def _cond_personalized(paths: list[Path]) -> dict | None:
    """Per-condition AUROC from personalized (head-ensemble) eval JSONs."""
    present = [p for p in paths if p.exists()]
    if not present:
        return None
    rows = {c: [] for c in CONDITIONS_KEY}
    for p in present:
        d = _load_json(p)
        for r in d["results"]:
            rows[r["name"]].append(r["auroc"])
    n = len(present)
    return {
        c: (float(np.mean(v)), float(np.std(v, ddof=1)) if n >= 2 else 0.0, n)
        for c, v in rows.items() if v
    }


def _cond_fedper() -> dict:
    """FedPer per-condition AUROC: multi-seed if sweep has produced them,
    otherwise fall back to the legacy single-seed artifact."""
    return _cond_multi_seed_or_legacy(ART["cond_fedper"], ART["cond_fedper_legacy"])


def _cond_fedavg() -> dict:
    """Untied FedAvg per-condition AUROC across available seeds."""
    paths = [p for p in ART["cond_untied_fedavg"] if p.exists()]
    if not paths:
        raise FileNotFoundError("No cond_untied_fedavg seeds present")
    rows = {c: [] for c in CONDITIONS_KEY}
    for p in paths:
        d = _load_json(p)
        for r in d["results"]:
            rows[r["name"]].append(r["auroc"])
    n = len(paths)
    return {
        c: (float(np.mean(v)), float(np.std(v, ddof=1)) if n >= 2 else 0.0, n)
        for c, v in rows.items() if v
    }


# ---------------------------------------------------------------------------
# Style constants — 2-panel layout (effective scale ~0.35x on letter page)
# ---------------------------------------------------------------------------

PANEL_STYLE = {
    "font.size": 24,
    "axes.labelsize": 28,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 20,
    "axes.linewidth": 2.0,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
}

# Strategy colors — organized by collaboration type
COLOR_CENTRALIZED = "#1f77b4"   # blue — data pooling
COLOR_FEDERATED   = "#2ca02c"   # green — weight sharing
COLOR_ENSEMBLE    = "#ff7f0e"   # orange — prediction sharing
COLOR_FAILURE     = "#d62728"   # red — methods that fail
COLOR_BASELINE    = "#7f7f7f"   # gray — simple baselines


def _clean_ax(ax):
    """Remove top/right spines and add subtle gridlines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2, axis="x", linestyle="--")


def _panel_label(ax, letter, fontsize=26):
    """Bold lowercase panel label at top-left."""
    ax.text(-0.04, 1.04, letter, transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold", va="bottom")


# ---------------------------------------------------------------------------
# Figure 2: Multi-hospital strategies — AUROC headline + horizontal AUPRC bars
# ---------------------------------------------------------------------------

def figure2_main_results(records: dict, sweep_a05: dict,
                         cond_central: dict, cond_fedavg: dict):
    """(a) Per-condition AUROC, (b) overall AUPRC (horizontal), (c) new-onset AUPRC (horizontal)."""
    import matplotlib.gridspec as gridspec

    with plt.rc_context(PANEL_STYLE):
        fig = plt.figure(figsize=(22, 20))

        gs_top = gridspec.GridSpec(1, 1, top=0.98, bottom=0.60, left=0.07, right=0.97)
        gs_bot = gridspec.GridSpec(1, 2, top=0.55, bottom=0.14, left=0.22, right=0.97, wspace=0.70)

        # Color legend BELOW panels (b) and (c), linked to those panels
        legend_ax = fig.add_axes([0.22, 0.02, 0.75, 0.06])
        legend_ax.axis("off")
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=COLOR_CENTRALIZED, label="Centralized training"),
            Patch(facecolor=COLOR_FEDERATED,   label="Federated training"),
            Patch(facecolor="#9467bd",         label="Personalized federated training (FedPer)"),
            Patch(facecolor=COLOR_ENSEMBLE,    label="Inference-time ensemble"),
            Patch(facecolor=COLOR_BASELINE,    label="Baseline / per-site"),
        ]
        legend_ax.legend(
            handles=legend_handles, loc="center", ncol=3,
            frameon=True, framealpha=0.95, fontsize=18,
            handletextpad=0.6, columnspacing=1.5,
        )

        # ── Panel (a): Per-condition AUROC ──
        ax_a = fig.add_subplot(gs_top[0])
        _panel_label(ax_a, "a")

        central_mu = np.array([cond_central[c][0] for c in CONDITIONS_KEY])
        central_sd = np.array([cond_central[c][1] for c in CONDITIONS_KEY])
        fedavg_mu = np.array([cond_fedavg[c][0] for c in CONDITIONS_KEY])
        fedavg_sd = np.array([cond_fedavg[c][1] for c in CONDITIONS_KEY])
        ensemble_mu = np.array([sweep_a05["cond"][c][0] for c in CONDITIONS_KEY])
        ensemble_sd = np.array([sweep_a05["cond"][c][1] for c in CONDITIONS_KEY])

        x = np.arange(len(CONDITIONS_LABEL))
        width = 0.25
        err_kw = dict(ecolor="black", capsize=5, lw=1.3)

        ax_a.bar(x - width, central_mu, width, yerr=central_sd,
                 label="Centralized (data pooling)", color=COLOR_CENTRALIZED,
                 alpha=0.9, error_kw=err_kw)
        ax_a.bar(x, fedavg_mu, width, yerr=fedavg_sd,
                 label="FedAvg (weight sharing)", color=COLOR_FEDERATED,
                 alpha=0.9, error_kw=err_kw)
        ax_a.bar(x + width, ensemble_mu, width, yerr=ensemble_sd,
                 label="Ensemble (prediction sharing)", color=COLOR_ENSEMBLE,
                 alpha=0.9, error_kw=err_kw)

        ax_a.set_xticks(x)
        ax_a.set_xticklabels(
            ["AKI", "Sepsis", "Acute resp.\nfailure", "Heart\nfailure", "Acute MI"],
            fontsize=24,
        )
        ax_a.set_ylabel("AUROC", fontsize=28)
        ax_a.set_ylim(0, 1.0)
        ax_a.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1.0)
        ax_a.text(len(CONDITIONS_LABEL) - 0.5, 0.51, "random",
                  fontsize=18, color="gray", ha="right")
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)
        ax_a.grid(alpha=0.2, axis="y", linestyle="--")
        ax_a.legend(loc="lower right", fontsize=20, framealpha=0.95)

        # Reduction-from-centralized % labels above each strategy's own bar
        for i in range(len(CONDITIONS_LABEL)):
            fedavg_red = (1 - fedavg_mu[i] / central_mu[i]) * 100
            ensemble_red = (1 - ensemble_mu[i] / central_mu[i]) * 100
            # FedAvg label above the FedAvg (middle) bar
            ax_a.text(x[i], fedavg_mu[i] + fedavg_sd[i] + 0.018,
                      f"-{fedavg_red:.1f}%",
                      ha="center", fontsize=18, color=COLOR_FEDERATED,
                      fontweight="bold")
            # Ensemble label above the Ensemble (right) bar
            ax_a.text(x[i] + width, ensemble_mu[i] + ensemble_sd[i] + 0.018,
                      f"-{ensemble_red:.1f}%",
                      ha="center", fontsize=18, color=COLOR_ENSEMBLE,
                      fontweight="bold")

        # ── Panels (b) and (c): Horizontal AUPRC bars ──
        # Ordered from best to worst (by overall AUPRC) for readability.
        # Main-text figure shows only the untied configuration throughout;
        # tied variants are described briefly in Methods.
        methods = [
            ("Centralized",                   records["centralized_untied"], COLOR_CENTRALIZED, False),
            ("FedAvg",                        records["untied_fedavg"],  COLOR_FEDERATED, False),
            ("Ensemble",                      records["untied_ensemble"], COLOR_ENSEMBLE, False),
            ("Per-site mean",                 records["untied_persite"], COLOR_BASELINE, False),
            ("FedPer + LoRA",                 records["fedper_lora_pers"], "#9467bd", False),
            ("Linear bag-of-DX",              records["linear"],         COLOR_BASELINE, True),
            ("FedPer",                        records["fedper_pers"],    "#9467bd", False),
            ("Repeat history",                records["repeat"],         COLOR_BASELINE, True),
            ("FedProx",                       records["untied_fedprox"], COLOR_FEDERATED, False),
            ("Marginal frequency",            records["marginal"],       COLOR_BASELINE, True),
        ]

        names      = [m[0] for m in methods]
        overall_mu = np.array([m[1]["overall"][0] for m in methods])
        overall_sd = np.array([m[1]["overall"][1] for m in methods])
        new_mu     = np.array([m[1]["new_onset"][0] for m in methods])
        new_sd     = np.array([m[1]["new_onset"][1] for m in methods])
        colors     = [m[2] for m in methods]

        y_pos = np.arange(len(names))[::-1]  # top-to-bottom order

        marginal_overall = records["marginal"]["overall"][0]
        marginal_new     = records["marginal"]["new_onset"][0]

        for panel_idx, (mu, sd, xlabel, xmax, baseline_x, label) in enumerate([
            (overall_mu, overall_sd, "Overall AUPRC", 0.42, marginal_overall, "b"),
            (new_mu,     new_sd,     "New-onset AUPRC", 0.20, marginal_new,   "c"),
        ]):
            ax = fig.add_subplot(gs_bot[0, panel_idx])
            # Place panel label at the far top-left of the axes (inside), clear of the legend strip above
            ax.text(-0.28, 1.01, label, transform=ax.transAxes,
                    fontsize=30, fontweight="bold", va="top", ha="left")

            ax.barh(y_pos, mu, xerr=sd, color=colors, alpha=0.9,
                    edgecolor="black", linewidth=0.8,
                    error_kw=dict(ecolor="black", capsize=4, lw=1.0))
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=22)
            ax.set_xlabel(xlabel, fontsize=28)
            ax.set_xlim(0, xmax)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, axis="x", linestyle="--")
            ax.axvline(baseline_x, color="gray", linestyle="--",
                       alpha=0.5, linewidth=1.0)
            ax.text(baseline_x + xmax * 0.008, -0.5, "marginal",
                    fontsize=17, color="gray", ha="left", va="top",
                    rotation=0)

            # Value labels: inside bar (right-aligned white text) when bar is long enough,
            # otherwise outside. "Long enough" means the value reaches at least 30% of xmax.
            for i, (m, s) in enumerate(zip(mu, sd)):
                # Check bar width relative to xmax
                if m >= xmax * 0.30:
                    # Inside the bar, right-aligned, in white
                    ax.text(m - xmax * 0.01, y_pos[i], f"{m:.3f}",
                            va="center", ha="right", fontsize=18,
                            color="white", fontweight="bold")
                else:
                    # Outside the bar, to the right of the error bar
                    ax.text(m + sd[i] + xmax * 0.015, y_pos[i], f"{m:.3f}",
                            va="center", ha="left", fontsize=18,
                            color="black")

        path = OUT / "paper_fig2_main_results.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        print(f"wrote {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Robustness to heterogeneity
# ---------------------------------------------------------------------------

def figure3_alpha_robustness(sweep_by_alpha: dict, centralized: dict):
    """(a) Overall AUPRC vs alpha, (b) New-onset AUPRC vs alpha."""
    with plt.rc_context(PANEL_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
        xpos = np.arange(len(ALPHAS))

        for ax_i, (metric_key, central, ylabel_suffix, ymax, label) in enumerate([
            ("overall",  centralized["overall"][0],   "Overall", 0.42, "a"),
            ("new",      centralized["new_onset"][0], "New-onset", 0.20, "b"),
        ]):
            ens_mu = [sweep_by_alpha[a][f"ens_{metric_key}"][0] for a in ALPHAS]
            ens_sd = [sweep_by_alpha[a][f"ens_{metric_key}"][1] for a in ALPHAS]
            ps_mu  = [sweep_by_alpha[a][f"persite_{metric_key}"][0] for a in ALPHAS]
            ps_sd  = [sweep_by_alpha[a][f"persite_{metric_key}"][1] for a in ALPHAS]
            ax = axes[ax_i]
            _panel_label(ax, label, fontsize=22)

            ax.errorbar(xpos, ens_mu, yerr=ens_sd, marker="o", lw=3,
                        markersize=12, capsize=7, color=COLOR_ENSEMBLE,
                        label="Ensemble (5 sites)")
            ax.errorbar(xpos, ps_mu, yerr=ps_sd, marker="s", lw=3,
                        markersize=12, capsize=7, color=COLOR_BASELINE,
                        label="Per-site mean")
            ax.axhline(central, color=COLOR_CENTRALIZED, linestyle="--", lw=2.5,
                       label=f"Centralized (70K) = {central:.3f}")
            ax.set_xticks(xpos)
            ax.set_xticklabels([f"α={a}" for a in ALPHAS], fontsize=24)
            ax.set_xlabel("Site heterogeneity (lower α = more non-IID)",
                          fontsize=24)
            ax.set_ylabel(f"{ylabel_suffix} AUPRC", fontsize=26)
            ax.set_ylim(0, ymax)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.2, axis="y", linestyle="--")
            ax.legend(loc="lower right", fontsize=19, framealpha=0.95)

        fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.14,
                            wspace=0.28)
        path = OUT / "paper_fig3_alpha_robustness.png"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
        print(f"wrote {path}")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fmt(mu: float, sd: float) -> str:
    return f"{mu:.3f} ± {sd:.3f}" if sd > 0 else f"{mu:.3f}"


def table1_methods_comparison(records: dict, sweep_a05: dict):
    print("\n## Table 1. Performance comparison across federated strategies")
    print(f"(Federated/per-site values: Dirichlet α={HEADLINE_ALPHA}; "
          "mean ± SD over 3 splits where multi-seed available)\n")
    print("| Method | Training data | Communication | Overall AUPRC | New-onset AUPRC |")
    print("|---|---|---|---|---|")
    rows = [
        ("Centralized", "70K (pooled)", "N/A",
         _fmt(*records["centralized"]["overall"]),
         _fmt(*records["centralized"]["new_onset"])),
        ("**Ensemble (5 sites)**", "**~14K each**", "**Predictions at inference**",
         f"**{_fmt(*sweep_a05['ens_overall'])}**",
         f"**{_fmt(*sweep_a05['ens_new'])}**"),
        ("Per-site mean", "~14K each", "None",
         _fmt(*sweep_a05['persite_overall']),
         _fmt(*sweep_a05['persite_new'])),
        ("FedPer + LoRA (head ens.)", "70K (federated)", "Weights (rounds)",
         _fmt(*records['fedper_lora_pers']['overall']),
         _fmt(*records['fedper_lora_pers']['new_onset'])),
        ("FedPer (head ens.)", "70K (federated)", "Weights (rounds)",
         _fmt(*records['fedper_pers']['overall']),
         _fmt(*records['fedper_pers']['new_onset'])),
        ("Linear bag-of-DX", "70K", "N/A",
         _fmt(*records['linear']['overall']),
         _fmt(*records['linear']['new_onset'])),
        ("Repeat history", "N/A", "N/A",
         _fmt(*records['repeat']['overall']),
         _fmt(*records['repeat']['new_onset'])),
        ("FedProx (tied, best)", "70K (federated)", "Weights (rounds)",
         _fmt(*records['fedprox']['overall']),
         _fmt(*records['fedprox']['new_onset'])),
        ("Marginal frequency", "N/A", "N/A",
         _fmt(*records['marginal']['overall']),
         _fmt(*records['marginal']['new_onset'])),
    ]
    for r in rows:
        print(f"| {' | '.join(r)} |")


def table2_condition_auroc(sweep_a05: dict, cond_central: dict, cond_fedper: dict):
    print("\n## Table 2. Per-condition zero-shot AUROC for new-onset prediction")
    central_n = max(cond_central[c][2] for c in CONDITIONS_KEY)
    fedper_n  = max(cond_fedper[c][2] for c in CONDITIONS_KEY)
    print(f"(Ensemble: Dirichlet α={HEADLINE_ALPHA}, mean ± SD over 3 splits. "
          f"Centralized: mean ± SD over {central_n} seed{'s' if central_n > 1 else ''}. "
          f"FedPer (global head): mean ± SD over {fedper_n} seed{'s' if fedper_n > 1 else ''}.)\n")
    print("| Condition | Prevalence | Centralized | Ensemble | FedPer (global head) | Recovery (Ensemble) |")
    print("|---|---|---|---|---|---|")
    for i, (name, prev) in enumerate(CONDITIONS_FULL):
        c = CONDITIONS_KEY[i]
        central_mu, central_sd, _n1 = cond_central[c]
        ens_mu, ens_sd = sweep_a05["cond"][c]
        fp_mu, fp_sd, _n2 = cond_fedper[c]
        recovery = ens_mu / central_mu * 100
        print(
            f"| {name} | {prev} | {_fmt(central_mu, central_sd)} | "
            f"{_fmt(ens_mu, ens_sd)} | {_fmt(fp_mu, fp_sd)} | {recovery:.0f}% |"
        )


def table3_alpha_sweep(sweep_by_alpha: dict):
    print("\n## Table 3. Robustness across heterogeneity (α sweep, 3 splits each)\n")
    print("| α | Ensemble overall AUPRC | Ensemble new-onset AUPRC | "
          "Per-site overall AUPRC | Per-site new-onset AUPRC |")
    print("|---|---|---|---|---|")
    for a in ALPHAS:
        s = sweep_by_alpha[a]
        print(
            f"| {a} | "
            f"{_fmt(*s['ens_overall'])} | {_fmt(*s['ens_new'])} | "
            f"{_fmt(*s['persite_overall'])} | {_fmt(*s['persite_new'])} |"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load every AUPRC from canonical JSONs
    records = {
        "centralized": _centralized_stats(),
        "linear":      _linear(),
        "marginal":    _single_run(ART["marginal"], "marginal"),
        "repeat":      _single_run(ART["repeat"], "repeat"),
        "fedprox":     _single_run(ART["fedprox"]),
        "fedper":      _multi_seed_or_legacy(ART["fedper"], ART["fedper_legacy"]),
        "fedper_lora": _multi_seed_or_legacy(ART["fedper_lora"], ART["fedper_lora_legacy"]),
    }
    # Personalized (head-ensemble) stats
    fp_pers = _personalized_stats(ART["fedper_pers"])
    fl_pers = _personalized_stats(ART["fedper_lora_pers"])
    records["fedper_pers"]      = fp_pers if fp_pers else records["fedper"]
    records["fedper_lora_pers"] = fl_pers if fl_pers else records["fedper_lora"]

    # Untied control experiment records (3-seed means)
    def _load_multi(paths):
        overall, new_onset = [], []
        for p in paths:
            if p.exists():
                d = _load_json(p)["foundation"]
                overall.append(d["overall"]["auprc"])
                new_onset.append(d["new_onset"]["auprc"])
        n = len(overall)
        return {
            "overall": (float(np.mean(overall)), float(np.std(overall, ddof=1)) if n >= 2 else 0.0),
            "new_onset": (float(np.mean(new_onset)), float(np.std(new_onset, ddof=1)) if n >= 2 else 0.0),
            "_n_seeds": n,
        }

    records["untied_fedavg"]   = _load_multi(ART["untied_fedavg"])
    records["untied_fedprox"]  = _load_multi(ART["untied_fedprox"])
    records["centralized_tied"]   = _load_multi(ART["centralized_tied"])
    records["centralized_untied"] = _load_multi(ART["centralized_untied"])
    # Untied ensemble — nested "ensemble" key, not "foundation"
    ens_ov, ens_no = [], []
    ps_ov, ps_no = [], []
    for p in ART["untied_ensemble"]:
        if p.exists():
            d = _load_json(p)
            ens_ov.append(d["ensemble"]["overall"]["auprc"])
            ens_no.append(d["ensemble"]["new_onset"]["auprc"])
            ps_ov.append(float(np.mean([s["overall_auprc"] for s in d["per_site"]])))
            ps_no.append(float(np.mean([s["new_onset_auprc"] for s in d["per_site"]])))
    records["untied_ensemble"] = {
        "overall": (float(np.mean(ens_ov)), float(np.std(ens_ov, ddof=1))),
        "new_onset": (float(np.mean(ens_no)), float(np.std(ens_no, ddof=1))),
    }
    records["untied_persite"]  = {
        "overall": (float(np.mean(ps_ov)), float(np.std(ps_ov, ddof=1))),
        "new_onset": (float(np.mean(ps_no)), float(np.std(ps_no, ddof=1))),
    }
    # Tied FedAvg (single canonical artifact — same failure mode across configs)
    d_tied = _load_json(ART["tied_fedavg"])["foundation"]
    records["tied_fedavg"] = {
        "overall": (d_tied["overall"]["auprc"], 0.0),
        "new_onset": (d_tied["new_onset"]["auprc"], 0.0),
    }

    for name in ("fedper_pers", "fedper_lora_pers", "untied_fedavg", "untied_fedprox",
                 "untied_ensemble", "untied_persite"):
        n = records[name].get("_n_seeds", "?")
        print(f"[figures] {name}: n_seeds={n}")

    # Consistency check — all methods must share the same test-set denominators
    flat_for_check = {}
    flat_for_check["fedprox"] = _load_json(ART["fedprox"])["foundation"]
    # Pick any one available FedPer/FedPer+LoRA artifact for the denominator check
    for label, multi, legacy in [
        ("fedper", ART["fedper"], ART["fedper_legacy"]),
        ("fedper_lora", ART["fedper_lora"], ART["fedper_lora_legacy"]),
    ]:
        chosen = next((p for p in multi if p.exists()), legacy)
        flat_for_check[label] = _load_json(chosen)["foundation"]
    for s in CENTRALIZED_SEEDS:
        flat_for_check[f"centralized_{s}"] = _load_json(PROC / f"eval_results_abnormal_{s}.json")["foundation"]
    _check_denominators(flat_for_check)

    # Load sweep at all alphas; headline is alpha = 0.5
    sweep_by_alpha = {a: _sweep_stats(a) for a in ALPHAS}
    sweep_a05 = sweep_by_alpha[HEADLINE_ALPHA]

    # Per-condition data: prefer personalized head-ensemble, fall back to global-head
    cond_fp = _cond_personalized(ART["cond_fedper_pers"])
    if cond_fp is None:
        cond_fp = _cond_fedper()
        print("[figures] cond_fedper: using global-head (personalized not available)")
    else:
        print(f"[figures] cond_fedper: using personalized head-ensemble")

    # Figures
    figure2_main_results(records, sweep_a05, _cond_centralized(), _cond_fedavg())
    figure3_alpha_robustness(sweep_by_alpha, records["centralized"])

    # Tables
    table1_methods_comparison(records, sweep_a05)
    table2_condition_auroc(sweep_a05, _cond_centralized(), cond_fp)
    table3_alpha_sweep(sweep_by_alpha)

    print("\nDone. Figures saved to figures/paper_fig{2,3,4}_*.png")


if __name__ == "__main__":
    main()
