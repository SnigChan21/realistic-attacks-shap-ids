"""Phase 2 for CIC-IDS2017 (MachineLearningCSV): same pipeline as UNSW-NB15.

 1. Read the 8 day files, keep the 11 CICFlowMeter columns of the FEINT schema, map them to
    the 15-feature schema (duration in seconds, derived features recomputed).
 2. Drop non-finite rows; normalise labels (strip, fix the broken dash in 'Web Attack').
 3. Remove exact duplicates and feature vectors that carry conflicting labels.
 4. Stratified sample: 10% of each class, at least 5,000 per class (smaller classes kept whole).
 5. Stratified 70/30 train/test split (seed 0).
 6. Spearman correlation filter |rho| > 0.95 on the training set (drop derived member first).
"""
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from schema import CIC_SCHEMA as S  # noqa: E402

ZIP = Path("/mnt/user-data/uploads/MachineLearningCSV.zip")
OUT = Path(__file__).resolve().parents[1]
CIC_MAP = {  # CICFlowMeter header -> (schema feature, scale); from FEINT data.py
    "Flow Duration": ("duration", 1e-6), "Total Fwd Packets": ("fwd_pkts", 1.0),
    "Total Backward Packets": ("bwd_pkts", 1.0), "Total Length of Fwd Packets": ("fwd_bytes", 1.0),
    "Total Length of Bwd Packets": ("bwd_bytes", 1.0), "Fwd Packet Length Max": ("fwd_pkt_len_max", 1.0),
    "Bwd Packet Length Max": ("bwd_pkt_len_max", 1.0), "SYN Flag Count": ("syn_count", 1.0),
    "Destination Port": ("dst_port", 1.0), "Init_Win_bytes_forward": ("init_win_fwd", 1.0),
    "Init_Win_bytes_backward": ("init_win_bwd", 1.0),
}
CORR_T, FRAC, MIN_PER_CLASS, SEED = 0.95, 0.10, 5000, 0


def norm_label(v: str) -> str:
    v = v.strip().replace("\u00ef\u00bf\u00bd", "-").replace("\ufffd", "-").replace("\u2013", "-")
    v = " ".join(v.replace(" - ", " ").replace("-", " ").split())
    return "BENIGN" if v.upper() == "BENIGN" else v


def read_day(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as fh:
        df = pd.read_csv(fh, usecols=lambda c: c.strip() in list(CIC_MAP) + ["Label"],
                         encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    X = np.zeros((len(df), len(S.features)), dtype=np.float64)
    ok = np.ones(len(df), dtype=bool)
    for col, (feat, scale) in CIC_MAP.items():
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float) * scale
        ok &= np.isfinite(v)
        X[:, S.idx[feat]] = np.maximum(np.nan_to_num(v), 0.0)  # CIC uses -1 for "no window"
    X[:, S.idx["fwd_pkts"]] = np.maximum(X[:, S.idx["fwd_pkts"]], 1.0)
    fb, mx, fp = S.idx["fwd_bytes"], S.idx["fwd_pkt_len_max"], S.idx["fwd_pkts"]
    X[:, mx] = np.clip(X[:, mx], np.ceil(np.round(X[:, fb] / X[:, fp], 6)), np.maximum(X[:, fb], 0))
    X = S.recompute(X)
    out = pd.DataFrame(X, columns=S.features)
    out["attack_cat"] = [norm_label(str(v)) for v in df["Label"].to_numpy()]
    out["day"] = name.split("/")[-1].split(".")[0]
    return out[ok]


def main() -> None:
    log = {}
    zf = zipfile.ZipFile(ZIP)
    parts = []
    for n in sorted(zf.namelist()):
        if n.endswith(".csv"):
            d = read_day(zf, n)
            log.setdefault("rows_per_day", {})[d.day.iloc[0]] = len(d)
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    del parts
    df["label"] = (df.attack_cat != "BENIGN").astype(int)
    log["raw_rows_finite"] = len(df)
    log["raw_class_counts"] = df.attack_cat.value_counts().to_dict()

    n0 = len(df)
    df = df.drop_duplicates(subset=S.features + ["attack_cat"])
    log["duplicates_removed"] = n0 - len(df)
    nlab = df.groupby(S.features)["label"].transform("nunique")
    log["rows_with_conflicting_labels_removed"] = int((nlab > 1).sum())
    df = df[nlab == 1]

    rng = np.random.default_rng(SEED)
    keep = []
    for cat, g in df.groupby("attack_cat"):
        k = len(g) if len(g) <= MIN_PER_CLASS else max(MIN_PER_CLASS, int(round(FRAC * len(g))))
        keep.append(g.iloc[rng.choice(len(g), size=k, replace=False)])
    df = pd.concat(keep).reset_index(drop=True)
    # drop classes too small to split/attack meaningfully
    small = df.attack_cat.value_counts()
    small = small[small < 10].index.tolist()
    log["classes_dropped_lt10"] = small
    df = df[~df.attack_cat.isin(small)]
    log["sampled_class_counts"] = df.attack_cat.value_counts().to_dict()

    tr, te = train_test_split(df, test_size=0.3, stratify=df.attack_cat, random_state=SEED)
    log["clean_rows"] = {"train": len(tr), "test": len(te)}
    log["attack_share"] = {"train": round(tr.label.mean(), 4), "test": round(te.label.mean(), 4)}

    rho = tr[S.features].corr(method="spearman").abs()
    pairs = [(a, b, round(float(rho.loc[a, b]), 4)) for i, a in enumerate(S.features)
             for b in S.features[i + 1:] if rho.loc[a, b] > CORR_T]
    drop = set()
    for a, b, _ in pairs:
        if a in drop or b in drop:
            continue
        drop.add(b if b in S.derived else a if a in S.derived else b)
    feats = [f for f in S.features if f not in drop]
    log["corr_pairs_over_0.95"] = pairs
    log["dropped_for_correlation"] = sorted(drop)
    log["model_features"] = feats
    roles = {f: ("controllable" if f in S.controllable else "derived" if f in S.derived else "fixed") for f in feats}
    log["model_feature_roles"] = roles
    log["role_counts"] = pd.Series(roles).value_counts().to_dict()

    tr.to_csv(OUT / "data/processed/cic_train.csv", index=False)
    te.to_csv(OUT / "data/processed/cic_test.csv", index=False)
    json.dump(log, open(OUT / "results/tables/cic_cleaning_log.json", "w"), indent=2, default=str)
    print(json.dumps(log, indent=1, default=str))


if __name__ == "__main__":
    main()
