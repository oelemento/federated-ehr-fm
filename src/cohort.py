"""Build the POC cohort and train/val/test splits from MIMIC-IV.

Keeps patients with >=2 hospital admissions, stratifies by admission-count
bucket, and writes patient-ID CSVs to data/splits/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIMIC = REPO_ROOT / "data" / "MIMIC-IV" / "hosp"
DEFAULT_OUT = REPO_ROOT / "data" / "splits"

SEED = 20260410
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # test gets the remainder


def bucket(n: int) -> str:
    if n <= 2:
        return "2"
    if n <= 4:
        return "3-4"
    if n <= 9:
        return "5-9"
    return "10+"


def build_cohort(mimic_dir: Path) -> pd.DataFrame:
    adm = pd.read_csv(
        mimic_dir / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
    )
    counts = adm.groupby("subject_id").size().rename("n_adm").reset_index()
    cohort = counts[counts["n_adm"] >= 2].copy()
    cohort["bucket"] = cohort["n_adm"].apply(bucket)
    return cohort.sort_values("subject_id").reset_index(drop=True)


def stratified_split(cohort: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for _, group in cohort.groupby("bucket", sort=True):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))
        parts["train"].append(cohort.loc[idx[:n_train]])
        parts["val"].append(cohort.loc[idx[n_train : n_train + n_val]])
        parts["test"].append(cohort.loc[idx[n_train + n_val :]])
    return {k: pd.concat(v).sort_values("subject_id").reset_index(drop=True) for k, v in parts.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=DEFAULT_MIMIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cohort] reading admissions from {args.mimic_dir}")
    cohort = build_cohort(args.mimic_dir)
    print(f"[cohort] patients with >=2 admissions: {len(cohort):,}")
    print("[cohort] admission-count bucket distribution:")
    print(cohort["bucket"].value_counts().reindex(["2", "3-4", "5-9", "10+"]).to_string())

    splits = stratified_split(cohort, args.seed)
    for name, df in splits.items():
        path = args.out_dir / f"{name}_patients.csv"
        df.to_csv(path, index=False)
        print(f"[cohort] wrote {path.name}: {len(df):,} patients")

    cohort.to_csv(args.out_dir / "all_patients.csv", index=False)
    print(f"[cohort] wrote all_patients.csv: {len(cohort):,} patients")


if __name__ == "__main__":
    main()
