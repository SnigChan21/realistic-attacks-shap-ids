"""Phase 2: clean UNSW-NB15 and build the constraint-aware feature matrix.

Steps (each logged to results/tables/unsw_cleaning_log.json):
 1. Map the official CSVs onto the 22-feature UNSW schema from FEINT
    (base features copied, proto one-hot for tcp/udp, derived features recomputed).
 2. Drop non-finite rows.
 3. Remove exact duplicates within train, within test, and test rows that also
    appear in train (cross-split leakage).
 4. Spearman correlation on the training set; report every pair with |rho| > 0.95
    and choose the model feature list (drop the derived member of each pair).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from schema import UNSW_SCHEMA as S  # noqa: E402

RAW = Path("/mnt/user-data/uploads")
OUT = Path(__file__).resolve().parents[1]
DIRECT = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl", "swin", "dwin",
          "trans_depth", "ct_srv_src", "ct_dst_ltm", "ct_src_ltm", "ct_srv_dst"]
CORR_T = 0.95


def to_frame(df: pd.DataFrame) -> pd.DataFrame:
    X = np.zeros((len(df), len(S.features)))
    for c in DIRECT:
        X[:, S.idx[c]] = np.maximum(df[c].to_numpy(dtype=float), 0.0)
    X[:, S.idx["sttl"]] = np.clip(X[:, S.idx["sttl"]], 1, 255)
    X[:, S.idx["spkts"]] = np.maximum(X[:, S.idx["spkts"]], 1.0)
    X[:, S.idx["proto_tcp"]] = (df["proto"] == "tcp").to_numpy(dtype=float)
    X[:, S.idx["proto_udp"]] = (df["proto"] == "udp").to_numpy(dtype=float)
    X = S.recompute(X)
    out = pd.DataFrame(X, columns=S.features)
    out["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip().values
    out["label"] = df["label"].astype(int).values
    return out


def main() -> None:
    log = {}
    tr = to_frame(pd.read_csv(RAW / "UNSW_NB15_training-set.csv"))
    te = to_frame(pd.read_csv(RAW / "UNSW_NB15_testing-set.csv"))
    log["raw_rows"] = {"train": len(tr), "test": len(te)}

    for name, d in (("train", tr), ("test", te)):
        fin = np.isfinite(d[S.features].to_numpy()).all(axis=1)
        log[f"nonfinite_dropped_{name}"] = int((~fin).sum())
    tr = tr[np.isfinite(tr[S.features].to_numpy()).all(axis=1)]
    te = te[np.isfinite(te[S.features].to_numpy()).all(axis=1)]

    key = S.features + ["label"]
    n0 = len(tr); tr = tr.drop_duplicates(subset=key); log["dup_within_train"] = n0 - len(tr)
    n0 = len(te); te = te.drop_duplicates(subset=key); log["dup_within_test"] = n0 - len(te)
    seen = pd.MultiIndex.from_frame(tr[key].round(9))
    in_train = pd.MultiIndex.from_frame(te[key].round(9)).isin(seen)
    log["test_rows_also_in_train"] = int(in_train.sum())
    te = te[~in_train]

    # label conflicts: identical features with different labels inside train
    conf = tr.groupby(S.features, dropna=False)["label"].nunique()
    log["train_feature_vectors_with_conflicting_labels"] = int((conf > 1).sum())

    log["clean_rows"] = {"train": len(tr), "test": len(te)}
    log["attack_share"] = {"train": round(tr.label.mean(), 4), "test": round(te.label.mean(), 4)}
    log["class_counts_train"] = tr.attack_cat.value_counts().to_dict()
    log["class_counts_test"] = te.attack_cat.value_counts().to_dict()

    # correlation control
    rho = tr[S.features].corr(method="spearman").abs()
    pairs = []
    for i, a in enumerate(S.features):
        for b in S.features[i + 1:]:
            if rho.loc[a, b] > CORR_T:
                pairs.append((a, b, round(float(rho.loc[a, b]), 4)))
    drop = set()
    for a, b, _ in pairs:
        if a in drop or b in drop:
            continue
        # prefer dropping the derived member, else the second of the pair
        if b in S.derived:
            drop.add(b)
        elif a in S.derived:
            drop.add(a)
        else:
            drop.add(b)
    model_feats = [f for f in S.features if f not in drop]
    log["corr_pairs_over_0.95"] = pairs
    log["dropped_for_correlation"] = sorted(drop)
    log["model_features"] = model_feats
    roles = {f: ("controllable" if f in S.controllable else "derived" if f in S.derived else "fixed")
             for f in model_feats}
    log["model_feature_roles"] = roles
    log["role_counts"] = pd.Series(roles).value_counts().to_dict()

    tr.to_csv(OUT / "data/processed/unsw_train.csv", index=False)
    te.to_csv(OUT / "data/processed/unsw_test.csv", index=False)
    with open(OUT / "results/tables/unsw_cleaning_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(json.dumps(log, indent=2, default=str))


if __name__ == "__main__":
    main()
