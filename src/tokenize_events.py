"""Build vocabulary and tokenized per-patient sequences for the POC.

Inputs  : MIMIC-IV hosp/ tables + data/splits/*.csv
Outputs : data/processed/vocab.json
          data/processed/{train,val,test}.pkl  (list[dict] per patient)

Each per-patient record:
    subject_id   : int
    token_ids    : np.int32[L]
    block_ids    : np.int32[L]  0=<bos>, 1..N=admission blocks
    position_ids : np.int32[L]  day index for RoPE; <sep>_i carries day(i+1)
    sep_positions: np.int32[N]  index in token_ids of each <sep>
    n_adm        : int          number of admissions in the sequence
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MIMIC = REPO_ROOT / "data" / "MIMIC-IV" / "hosp"
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"

MAX_LEN = 1024
MIN_MED_COUNT = 50
TOP_LAB_ITEMS = 200     # UCLA used ~446 LOINC codes; MIMIC has 1650 itemids total.
N_LAB_DECILES = 10      # eCDF-based binning, decile tokens LAB:<itemid>_D0..D9.
LAB_CHUNK = 2_000_000   # rows per chunk when streaming labevents.csv.gz

SPECIALS = ["<pad>", "<bos>", "<sep>", "<unk>"]
PAD_ID, BOS_ID, SEP_ID, UNK_ID = 0, 1, 2, 3

_DRUG_RX = re.compile(r"[a-z][a-z\-]{2,}")


def normalize_drug(drug: object) -> str | None:
    if not isinstance(drug, str):
        return None
    m = _DRUG_RX.search(drug.lower())
    return m.group(0) if m else None


def icd_token(code: object, version: object, prefix: str) -> str | None:
    if not isinstance(code, str) or len(code) == 0:
        return None
    head = code.strip()[:3].upper()
    if not head:
        return None
    try:
        v_int = int(version)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    v = "10" if v_int == 10 else "9"
    return f"{prefix}{v}:{head}"


def load_admissions(mimic_dir: Path) -> pd.DataFrame:
    adm = pd.read_csv(
        mimic_dir / "admissions.csv.gz",
        usecols=["subject_id", "hadm_id", "admittime"],
        parse_dates=["admittime"],
    )
    return adm


def load_diagnosis_tokens(mimic_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        mimic_dir / "diagnoses_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str, "icd_version": "int8"},
    )
    df["token"] = [
        icd_token(c, v, "DX")
        for c, v in zip(df["icd_code"].values, df["icd_version"].values)
    ]
    return df[["subject_id", "hadm_id", "token"]].dropna()


def load_procedure_tokens(mimic_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        mimic_dir / "procedures_icd.csv.gz",
        usecols=["subject_id", "hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str, "icd_version": "int8"},
    )
    df["token"] = [
        icd_token(c, v, "PX")
        for c, v in zip(df["icd_code"].values, df["icd_version"].values)
    ]
    return df[["subject_id", "hadm_id", "token"]].dropna()


def load_lab_tokens(
    mimic_dir: Path,
    cohort_ids: set[int],
    train_ids: set[int],
    top_items_n: int = TOP_LAB_ITEMS,
    mode: str = "all_extreme",
) -> tuple[pd.DataFrame, dict[int, list[float]]]:
    """Stream labevents.csv.gz, return per-(patient, admission, itemid) decile tokens
    using training-split eCDF binning.

    Tokens look like ``LAB:<itemid>_D<k>`` where k in 0..9 is the decile on the training-set
    empirical CDF of that itemid's ``valuenum`` values.

    Modes
    -----
    ``all_extreme``
        Within each (admission, itemid), pick the decile furthest from the median
        (|d - 4.5| maximal) and emit one token per (admission, itemid). This was the
        original behavior.

    ``abnormal_only``
        Filter rows to only those with decile in {0, 1, 8, 9} before dedupe. If a
        given (admission, itemid) has no abnormal value, **no token is emitted** for
        that pair. This drops ~60% of lab tokens, which alleviates sequence-length
        inflation and focuses the signal on clinically meaningful deviations. Pairs
        well with a larger ``top_items_n`` (e.g. 400) since each itemid now contributes
        less on average.

    Returns
    -------
    (DataFrame with columns ['subject_id', 'hadm_id', 'token'],
     dict mapping itemid -> list of 11 bin edges)
    """
    if mode not in {"all_extreme", "abnormal_only"}:
        raise ValueError(f"unknown lab mode: {mode!r}")
    # -- Pass 1: load all numeric cohort rows into memory ----------------------------
    pieces: list[pd.DataFrame] = []
    reader = pd.read_csv(
        mimic_dir / "labevents.csv.gz",
        usecols=["subject_id", "hadm_id", "itemid", "valuenum"],
        dtype={"valuenum": "float64"},
        chunksize=LAB_CHUNK,
    )
    for i, chunk in enumerate(reader):
        chunk = chunk.dropna(subset=["hadm_id", "valuenum"])
        chunk = chunk[chunk["subject_id"].isin(cohort_ids)]
        if chunk.empty:
            continue
        pieces.append(
            chunk.astype({"subject_id": "int64", "hadm_id": "int64", "itemid": "int64"})
        )
        if (i + 1) % 5 == 0:
            kept = sum(len(p) for p in pieces)
            print(f"  [lab] streamed {(i + 1) * LAB_CHUNK:,} rows, kept {kept:,}")
    if not pieces:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "token"]), {}
    df = pd.concat(pieces, ignore_index=True)
    print(f"[lab] cohort rows after filter: {len(df):,}")

    # -- Choose top-N itemids by training-split frequency ----------------------------
    train_df = df[df["subject_id"].isin(train_ids)]
    top_counts = train_df["itemid"].value_counts().head(top_items_n)
    top_items: list[int] = top_counts.index.astype(int).tolist()
    print(
        f"[lab] top {len(top_items)} itemids cover "
        f"{top_counts.sum()/max(1, len(train_df)):.1%} of training lab rows"
    )
    df = df[df["itemid"].isin(top_items)].reset_index(drop=True)
    print(f"[lab] after top-{top_items_n} filter: {len(df):,}")

    # -- Per-itemid decile edges computed on training rows only ----------------------
    bin_edges: dict[int, list[float]] = {}
    q = np.linspace(0.0, 1.0, N_LAB_DECILES + 1)
    tr_in_top = train_df[train_df["itemid"].isin(top_items)]
    for itemid, g in tr_in_top.groupby("itemid", sort=False):
        vals = g["valuenum"].to_numpy()
        edges = np.quantile(vals, q)
        # Force strictly increasing edges so np.digitize behaves. Repeated quantiles
        # (e.g. when 80% of values cluster at one number) collapse into a single bin,
        # which is the correct behavior: that bin is wide, others are narrow.
        for k in range(1, len(edges)):
            if edges[k] <= edges[k - 1]:
                edges[k] = np.nextafter(edges[k - 1], np.inf)
        bin_edges[int(itemid)] = edges.tolist()

    # -- Assign each row a decile 0..9 using training edges --------------------------
    decile = np.full(len(df), -1, dtype=np.int8)
    itemid_arr = df["itemid"].to_numpy()
    valuenum_arr = df["valuenum"].to_numpy()
    for itemid in top_items:
        mask = itemid_arr == itemid
        if not mask.any():
            continue
        edges = np.asarray(bin_edges[int(itemid)])
        # Internal edges only (length 9) so np.digitize returns 0..9.
        bins = np.digitize(valuenum_arr[mask], edges[1:-1])
        decile[mask] = np.clip(bins, 0, N_LAB_DECILES - 1)
    assert (decile >= 0).all(), "decile assignment missed some rows"
    df["decile"] = decile

    # -- Mode-specific filtering before dedupe ---------------------------------------
    if mode == "abnormal_only":
        before = len(df)
        # Keep only clinically abnormal readings (lower two deciles and upper two).
        abnormal_mask = (df["decile"].values <= 1) | (df["decile"].values >= N_LAB_DECILES - 2)
        df = df[abnormal_mask].reset_index(drop=True)
        print(
            f"[lab] mode=abnormal_only: kept {len(df):,} rows with decile in {{0,1,8,9}} "
            f"({len(df)/max(1, before):.1%} of top-{top_items_n})"
        )

    # -- Most-extreme decile per (subject_id, hadm_id, itemid) -----------------------
    df["abs_dev"] = np.abs(df["decile"].astype(np.float32) - 4.5)
    df = df.sort_values(
        ["subject_id", "hadm_id", "itemid", "abs_dev"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )
    df = df.drop_duplicates(["subject_id", "hadm_id", "itemid"], keep="first").reset_index(drop=True)
    df["token"] = "LAB:" + df["itemid"].astype(str) + "_D" + df["decile"].astype(str)
    print(
        f"[lab] emitted {len(df):,} (admission, itemid) lab tokens "
        f"across {len(top_items)} itemids (mode={mode})"
    )
    return df[["subject_id", "hadm_id", "token"]], bin_edges


def load_medication_tokens(mimic_dir: Path, cohort_ids: set[int]) -> pd.DataFrame:
    """Chunked load of prescriptions; filter to cohort patients and normalize drug names."""
    pieces: list[pd.DataFrame] = []
    chunksize = 1_000_000
    reader = pd.read_csv(
        mimic_dir / "prescriptions.csv.gz",
        usecols=["subject_id", "hadm_id", "drug"],
        dtype={"drug": str},
        chunksize=chunksize,
    )
    for i, chunk in enumerate(reader):
        chunk = chunk[chunk["subject_id"].isin(cohort_ids)]
        if chunk.empty:
            continue
        chunk = chunk.assign(token="MED:" + chunk["drug"].map(normalize_drug).fillna(""))
        chunk = chunk[chunk["token"] != "MED:"]
        # dedupe within (patient, admission, drug) so a 30-dose order = 1 token
        chunk = chunk[["subject_id", "hadm_id", "token"]].drop_duplicates()
        pieces.append(chunk)
        if (i + 1) % 10 == 0:
            print(f"  [med] processed {(i + 1) * chunksize:,} rows")
    df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(
        columns=["subject_id", "hadm_id", "token"]
    )
    return df


def build_vocab(all_events: pd.DataFrame, train_ids: set[int]) -> dict[str, int]:
    """Vocab = all DX/PX/LAB tokens seen in training + MED tokens with count >= MIN_MED_COUNT.

    LAB tokens are already capped to TOP_LAB_ITEMS × N_LAB_DECILES by `load_lab_tokens`,
    so no additional min-count threshold is applied here.
    """
    train_events = all_events[all_events["subject_id"].isin(train_ids)]
    counts: Counter[str] = Counter()
    for tok in train_events["token"].values:
        counts[tok] += 1

    dx_px = sorted({t for t in counts if t.startswith(("DX", "PX"))})
    meds = sorted({t for t, c in counts.items() if t.startswith("MED") and c >= MIN_MED_COUNT})
    labs = sorted({t for t in counts if t.startswith("LAB")})

    vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIALS)}
    for tok in dx_px + meds + labs:
        vocab[tok] = len(vocab)

    dx_total = sum(1 for t in vocab if t.startswith("DX"))
    px_total = sum(1 for t in vocab if t.startswith("PX"))
    med_total = sum(1 for t in vocab if t.startswith("MED"))
    lab_total = sum(1 for t in vocab if t.startswith("LAB"))
    print(
        f"[vocab] specials={len(SPECIALS)} DX={dx_total} PX={px_total} "
        f"MED={med_total} LAB={lab_total} total={len(vocab)}"
    )
    return vocab


def tokenize_patient(
    subject_id: int,
    admissions: list[tuple[int, int]],  # list of (hadm_id, day_index)
    adm_tokens: dict[int, list[int]],   # hadm_id -> sorted unique token ids
    max_len: int,
    trunc_stats: dict | None = None,    # optional in-place counter dict for instrumentation
) -> dict | None:
    """Build one patient's sequence with block/position bookkeeping, left-truncating if needed.

    If `trunc_stats` is provided it is updated in-place with:
      - n_patients_truncated    : patients whose initial sequence exceeded max_len
      - n_blocks_dropped_total  : total number of admission blocks dropped via left-truncation
      - n_hard_truncated        : patients whose single admission still overflowed after block drops
    """
    blocks = []
    for i, (hadm_id, day) in enumerate(admissions, start=1):
        toks = adm_tokens.get(hadm_id, [])
        if not toks:
            continue
        blocks.append({"block_id": i, "hadm_id": hadm_id, "day": day, "tokens": toks})
    if len(blocks) < 2:
        return None  # need >=2 non-empty admissions

    # Re-index block_ids to be contiguous 1..N (drop any gaps from empty admissions)
    for new_idx, blk in enumerate(blocks, start=1):
        blk["block_id"] = new_idx

    # Build sequence; may exceed max_len, then left-truncate blocks
    def build_seq(bs: list[dict]) -> tuple[list[int], list[int], list[int], list[int]]:
        token_ids = [BOS_ID]
        block_ids = [0]
        position_ids = [0]
        sep_positions: list[int] = []
        for i, blk in enumerate(bs):
            for t in blk["tokens"]:
                token_ids.append(t)
                block_ids.append(blk["block_id"])
                position_ids.append(blk["day"])
            # <sep> carrying next admission's day (or current if last)
            # For <sep>_i we store day(i+1) so the model conditions the next-visit prediction
            # on the *next* visit's time. The final <sep>_N has no i+1; we fall back to day(N).
            # Training masks this final sep out (n_scored = len(seps)-1) and zero-shot eval uses
            # <sep>_{N-1}, so the fallback value is never actually read for loss or inference.
            next_day = bs[i + 1]["day"] if i + 1 < len(bs) else blk["day"]
            token_ids.append(SEP_ID)
            block_ids.append(blk["block_id"])
            position_ids.append(next_day)
            sep_positions.append(len(token_ids) - 1)
        return token_ids, block_ids, position_ids, sep_positions

    kept = blocks
    token_ids, block_ids, position_ids, sep_positions = build_seq(kept)
    n_original_blocks = len(blocks)
    was_truncated = len(token_ids) > max_len
    while len(token_ids) > max_len and len(kept) > 2:
        kept = kept[1:]
        base = kept[0]["day"]
        kept = [{**b, "day": b["day"] - base} for b in kept]
        for new_idx, b in enumerate(kept, start=1):
            b["block_id"] = new_idx
        token_ids, block_ids, position_ids, sep_positions = build_seq(kept)
    if was_truncated and trunc_stats is not None:
        trunc_stats["n_patients_truncated"] = trunc_stats.get("n_patients_truncated", 0) + 1
        trunc_stats["n_blocks_dropped_total"] = (
            trunc_stats.get("n_blocks_dropped_total", 0) + (n_original_blocks - len(kept))
        )

    if len(token_ids) > max_len:
        # Only 2 blocks remain and still too long; hard-truncate the token stream from the left
        # and re-base positions so the first surviving token sits at day 0. This is a rare edge
        # case (single admission contributing >max_len tokens after dedupe).
        if trunc_stats is not None:
            trunc_stats["n_hard_truncated"] = trunc_stats.get("n_hard_truncated", 0) + 1
        excess = len(token_ids) - max_len
        token_ids = token_ids[excess:]
        block_ids = block_ids[excess:]
        position_ids = position_ids[excess:]
        sep_positions = [p - excess for p in sep_positions if p - excess >= 0]
        if position_ids:
            base = position_ids[0]
            position_ids = [p - base for p in position_ids]

    if len(sep_positions) < 2:
        return None

    return {
        "subject_id": subject_id,
        "token_ids": np.asarray(token_ids, dtype=np.int32),
        "block_ids": np.asarray(block_ids, dtype=np.int32),
        "position_ids": np.asarray(position_ids, dtype=np.int32),
        "sep_positions": np.asarray(sep_positions, dtype=np.int32),
        "n_adm": len(kept),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=DEFAULT_MIMIC)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument(
        "--lab-mode", choices=["all_extreme", "abnormal_only"], default="all_extreme",
        help="How to summarize lab events per (admission, itemid).",
    )
    parser.add_argument(
        "--top-lab-items", type=int, default=TOP_LAB_ITEMS,
        help="Number of lab itemids to keep by training-set frequency.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[tokenize] loading splits")
    splits = {
        name: pd.read_csv(args.splits_dir / f"{name}_patients.csv")
        for name in ("train", "val", "test")
    }
    train_ids = set(splits["train"]["subject_id"].astype(int).tolist())
    cohort_ids = set()
    for df in splits.values():
        cohort_ids.update(df["subject_id"].astype(int).tolist())
    print(f"[tokenize] cohort size: {len(cohort_ids):,}  train: {len(train_ids):,}")

    print("[tokenize] loading admissions")
    adm = load_admissions(args.mimic_dir)
    adm = adm[adm["subject_id"].isin(cohort_ids)].copy()
    adm = adm.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    # Day index per admission, relative to the patient's first admission.
    adm["first"] = adm.groupby("subject_id")["admittime"].transform("min")
    adm["day"] = (adm["admittime"] - adm["first"]).dt.days.astype("int32")
    patient_hadms: dict[int, list[tuple[int, int]]] = {
        sid: list(zip(g["hadm_id"].astype(int), g["day"].astype(int)))
        for sid, g in adm.groupby("subject_id", sort=False)
    }
    print(f"[tokenize] admissions in cohort: {len(adm):,}")

    print("[tokenize] loading diagnoses")
    dx = load_diagnosis_tokens(args.mimic_dir)
    dx = dx[dx["subject_id"].isin(cohort_ids)]
    print(f"[tokenize] diagnosis rows: {len(dx):,}")

    print("[tokenize] loading procedures")
    px = load_procedure_tokens(args.mimic_dir)
    px = px[px["subject_id"].isin(cohort_ids)]
    print(f"[tokenize] procedure rows: {len(px):,}")

    print("[tokenize] loading medications (chunked)")
    med = load_medication_tokens(args.mimic_dir, cohort_ids)
    print(f"[tokenize] medication rows: {len(med):,}")

    print(f"[tokenize] loading labs (mode={args.lab_mode}, top={args.top_lab_items})")
    lab, lab_bin_edges = load_lab_tokens(
        args.mimic_dir, cohort_ids, train_ids,
        top_items_n=args.top_lab_items,
        mode=args.lab_mode,
    )
    print(f"[tokenize] lab token rows: {len(lab):,}")
    if lab_bin_edges:
        (args.out_dir / "lab_bins.json").write_text(
            json.dumps({str(k): v for k, v in lab_bin_edges.items()}, indent=2)
        )
        print(f"[tokenize] wrote lab_bins.json ({len(lab_bin_edges)} itemids)")

    all_events = pd.concat([dx, px, med, lab], ignore_index=True)
    # Dedupe (subject_id, hadm_id, token). Handles two cases:
    #  (1) ICD 3-char collapse: distinct 5-char codes in one admission map to the same token.
    #  (2) Prescription chunk boundaries: a patient's rx rows spanning a chunk boundary survive
    #      the per-chunk drop_duplicates and need a final pass after concat.
    before = len(all_events)
    all_events = all_events.drop_duplicates(subset=["subject_id", "hadm_id", "token"]).reset_index(drop=True)
    print(f"[tokenize] total event rows: {len(all_events):,} (dedupe removed {before - len(all_events):,})")

    vocab = build_vocab(all_events, train_ids)
    with (args.out_dir / "vocab.json").open("w") as f:
        json.dump(vocab, f, indent=2)
    print(f"[tokenize] wrote vocab.json ({len(vocab):,} tokens)")

    # Map tokens to ids, drop OOV (MED below count threshold) silently.
    all_events["token_id"] = all_events["token"].map(vocab)
    oov = all_events["token_id"].isna().sum()
    print(f"[tokenize] OOV event rows dropped: {oov:,}")
    all_events = all_events.dropna(subset=["token_id"]).astype({"token_id": "int32"})

    # Deterministic per-admission sorted unique token list.
    # Within each admission, tokens are ordered DX < PX < MED < LAB, then alphabetically.
    def prefix_rank(tok: str) -> int:
        if tok.startswith("DX"):
            return 0
        if tok.startswith("PX"):
            return 1
        if tok.startswith("MED"):
            return 2
        return 3  # LAB

    all_events["rank"] = all_events["token"].map(prefix_rank).astype("int8")
    # sort by (patient, admission, rank, token) so that within-admission ordering is stable
    all_events = all_events.sort_values(["subject_id", "hadm_id", "rank", "token"])
    adm_tokens: dict[int, list[int]] = {}
    for hadm_id, g in all_events.groupby("hadm_id", sort=False):
        adm_tokens[int(hadm_id)] = g["token_id"].tolist()

    print("[tokenize] building per-patient sequences")
    built_counts = {"train": 0, "val": 0, "test": 0}
    records = {"train": [], "val": [], "test": []}
    trunc_stats_per_split: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    split_of = {}
    for name, df in splits.items():
        for sid in df["subject_id"].astype(int):
            split_of[sid] = name

    for sid, admissions in patient_hadms.items():
        name = split_of.get(sid)
        if name is None:
            continue
        rec = tokenize_patient(
            sid, admissions, adm_tokens, args.max_len,
            trunc_stats=trunc_stats_per_split[name],
        )
        if rec is None:
            continue
        records[name].append(rec)
        built_counts[name] += 1

    for name, recs in records.items():
        path = args.out_dir / f"{name}.pkl"
        with path.open("wb") as f:
            pickle.dump(recs, f, protocol=pickle.HIGHEST_PROTOCOL)
        lens = np.array([len(r["token_ids"]) for r in recs])
        n_adm = np.array([r["n_adm"] for r in recs])
        stats = trunc_stats_per_split[name]
        n_trunc = stats.get("n_patients_truncated", 0)
        n_blocks_dropped = stats.get("n_blocks_dropped_total", 0)
        n_hard = stats.get("n_hard_truncated", 0)
        print(
            f"[tokenize] {name:5s}: {len(recs):,} sequences  "
            f"len mean/median/max={lens.mean():.0f}/{np.median(lens):.0f}/{lens.max()}  "
            f"n_adm mean/median/max={n_adm.mean():.1f}/{np.median(n_adm):.0f}/{n_adm.max()}"
        )
        if n_trunc or n_hard:
            pct = 100.0 * n_trunc / max(1, len(recs))
            avg_dropped = n_blocks_dropped / max(1, n_trunc)
            print(
                f"[tokenize] {name:5s}: truncated={n_trunc} ({pct:.1f}%)  "
                f"avg_blocks_dropped={avg_dropped:.1f}  hard_truncated={n_hard}"
            )


if __name__ == "__main__":
    main()
