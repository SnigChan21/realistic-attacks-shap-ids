"""Phase 3: train Random Forest and XGBoost detectors (3 seeds) and report clean performance.

Models are trained on raw (unscaled) model features; tree ensembles are scale-invariant.
A StandardScaler fit on the training set is saved alongside for the attack phase,
which perturbs in standardised space.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DS = sys.argv[1] if len(sys.argv) > 1 else "unsw"
SEEDS = [0, 1, 2]


def models(seed: int):
    return {
        "rf": RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced",
                                     n_jobs=-1, random_state=seed),
        "xgb": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                             random_state=seed),
    }


def main() -> None:
    log = json.load(open(ROOT / f"results/tables/{DS}_cleaning_log.json"))
    feats = log["model_features"]
    tr = pd.read_csv(ROOT / f"data/processed/{DS}_train.csv")
    te = pd.read_csv(ROOT / f"data/processed/{DS}_test.csv")
    Xtr, ytr = tr[feats].to_numpy(), tr.label.to_numpy()
    Xte, yte = te[feats].to_numpy(), te.label.to_numpy()

    joblib.dump(StandardScaler().fit(Xtr), ROOT / f"models/{DS}_scaler.joblib")

    # leakage check: how well does each single feature separate the classes on its own?
    single = {f: round(max(roc_auc_score(ytr, tr[f]), 1 - roc_auc_score(ytr, tr[f])), 4) for f in feats}
    single = dict(sorted(single.items(), key=lambda kv: -kv[1]))

    rows = []
    for seed in SEEDS:
        for name, m in models(seed).items():
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xte)[:, 1]
            yhat = (p >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
            rows.append({"dataset": DS, "model": name, "seed": seed,
                         "accuracy": accuracy_score(yte, yhat), "precision": precision_score(yte, yhat),
                         "recall": recall_score(yte, yhat), "f1": f1_score(yte, yhat),
                         "fpr": fp / (fp + tn), "pr_auc": average_precision_score(yte, p),
                         "roc_auc": roc_auc_score(yte, p)})
            joblib.dump(m, ROOT / f"models/{DS}_{name}_seed{seed}.joblib")
            print(rows[-1])

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(ROOT / f"results/tables/{DS}_clean_performance_per_seed.csv", index=False)
    metrics = ["accuracy", "precision", "recall", "f1", "fpr", "pr_auc", "roc_auc"]
    agg = per_seed.groupby(["dataset", "model"])[metrics].agg(["mean", "std"])
    agg.to_csv(ROOT / f"results/tables/{DS}_clean_performance.csv")
    json.dump(single, open(ROOT / f"results/tables/{DS}_single_feature_auc.json", "w"), indent=2)
    print(agg.round(4).to_string())
    print("single-feature AUC (top 6):", list(single.items())[:6])


if __name__ == "__main__":
    main()
