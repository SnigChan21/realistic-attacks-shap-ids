"""Phases 4-5: constrained vs. unconstrained evasion, then SHAP drift.

Design: ONE black-box search algorithm (score-based, Square-Attack-style random search,
adapted from FEINT's ``random_search``, MIT licence) run in two modes that differ only in
the threat model:
  * unconstrained: every model feature may move by up to eps (textbook attack)
  * constrained:   only attacker-controllable features move; monotone / range / integer /
                   cross-feature relations enforced and derived features recomputed after
                   every step (FEINT schema projection), so every output is a realisable flow.
Because the search procedure, query budget and eps grid are identical, any difference in
explanation drift is attributable to the constraints, not to the attack algorithm.

Budgets are L_inf radii in standardised log1p space (z), eps grid EPS.
A noise baseline (one random eps-bounded perturbation, same mode, no search) separates
drift caused by perturbation size from drift caused by adversarial optimisation.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from schema import CIC_SCHEMA, UNSW_SCHEMA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = sys.argv[1] if len(sys.argv) > 1 else "unsw"
SCHEMA = {"unsw": UNSW_SCHEMA, "unsw_nosttl": UNSW_SCHEMA, "cic": CIC_SCHEMA}[DS]
import os as _os
EPS = [float(e) for e in _os.environ.get("EPS", "0.1,0.25,0.5,1.0,1.5,2.0").split(",")]
KINDS = _os.environ.get("KINDS", "attack,noise").split(",")
TAG = _os.environ.get("TAG", "")
N_FLOWS = 300
ITERS = 100
TOPK = 5
import os
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
MODELS = ["rf", "xgb"]


class LogZ:
    """x -> z = (log1p(max(x,0)) - mu) / sigma (as in FEINT)."""

    def fit(self, X):
        L = np.log1p(np.maximum(X, 0))
        self.mu, self.sigma = L.mean(0), L.std(0) + 1e-9
        return self

    def transform(self, X):
        return (np.log1p(np.maximum(X, 0)) - self.mu) / self.sigma

    def inverse(self, Z):
        return np.expm1(Z * self.sigma + self.mu).clip(min=0)


def random_search(predict, pre, X, schema, allowed, eps, constrained, seed, iters=ITERS, p_init=0.5):
    """Score-based black-box evasion; adapted from FEINT attack.random_search."""
    rng = np.random.default_rng(seed)
    Z0 = pre.transform(X)
    cur_X, cur_Z = X.copy(), Z0.copy()
    cur_p = predict(cur_X)
    for t in range(iters):
        active = cur_p >= 0.5
        if not active.any():
            break
        frac = p_init * (1 - t / iters) + 0.05
        k = max(1, int(round(frac * len(allowed))))
        idx = np.flatnonzero(active)
        D = np.zeros((len(idx), Z0.shape[1]))
        cols = allowed[np.argsort(rng.random((len(idx), len(allowed))), axis=1)[:, :k]]
        rows = np.repeat(np.arange(len(idx)), k)
        scale = rng.choice([0.25, 0.5, 1.0], size=(len(idx), 1))
        D[rows, cols.ravel()] = (rng.choice([-1.0, 1.0], size=(len(idx), k)) * eps * scale).ravel()
        Zc = cur_Z[idx].copy()
        sel = D != 0
        Zc[sel] = (Z0[idx] + D)[sel]
        Zc = Z0[idx] + np.clip(Zc - Z0[idx], -eps, eps)
        Xc = pre.inverse(Zc)
        if constrained:
            Xc = schema.project(Xc, X[idx])
            Zc = pre.transform(Xc)
        else:
            # textbook attack: features the attacker did not touch keep their exact values
            Xc[:, ~np.isin(np.arange(X.shape[1]), allowed)] = X[idx][:, ~np.isin(np.arange(X.shape[1]), allowed)]
        p = predict(Xc)
        better = p < cur_p[idx]
        upd = idx[better]
        cur_Z[upd], cur_X[upd], cur_p[upd] = Zc[better], Xc[better], p[better]
    if constrained:
        cur_X = schema.project(cur_X, X)
    return cur_X


def noise(pre, X, schema, allowed, eps, constrained, seed):
    rng = np.random.default_rng(seed + 10_000)
    Z0 = pre.transform(X)
    Z = Z0.copy()
    Z[:, allowed] += rng.uniform(-eps, eps, size=(len(X), len(allowed)))
    Xn = pre.inverse(Z)
    if constrained:
        return schema.project(Xn, X)
    Xn[:, ~np.isin(np.arange(X.shape[1]), allowed)] = X[:, ~np.isin(np.arange(X.shape[1]), allowed)]
    return Xn


def shap_values(explainer, X, model_name):
    v = explainer.shap_values(X, check_additivity=False)
    if isinstance(v, list):
        v = v[1]
    v = np.asarray(v)
    if v.ndim == 3:
        v = v[:, :, 1]
    return v


def drift_metrics(P0, P1, reach_mask):
    d = np.abs(P1 - P0)
    out = {
        "drift_mean": d.mean(1),                                    # mean |dphi| over features (ESI's D)
        "drift_rel": d.sum(1) / (np.abs(P0).sum(1) + 1e-12),        # L1 drift relative to clean attribution mass
        "reach_share": d[:, reach_mask].sum(1) / (d.sum(1) + 1e-12),  # share of drift on attacker-reachable features
        "fixed_drift_mean": d[:, ~reach_mask].mean(1),              # spill-over onto features the attacker cannot move
    }
    top0 = np.argsort(-np.abs(P0), axis=1)[:, :TOPK]
    top1 = np.argsort(-np.abs(P1), axis=1)[:, :TOPK]
    out["topk_overlap"] = np.array([len(set(a) & set(b)) / TOPK for a, b in zip(top0, top1)])
    out["spearman"] = np.array([spearmanr(np.abs(a), np.abs(b)).statistic for a, b in zip(P0, P1)])
    return out


def main() -> None:
    log = json.load(open(ROOT / f"results/tables/{DS}_cleaning_log.json"))
    mfeats = log["model_features"]
    feats = SCHEMA.features
    midx = [SCHEMA.idx[f] for f in mfeats]
    tr = pd.read_csv(ROOT / f"data/processed/{DS}_train.csv")
    te = pd.read_csv(ROOT / f"data/processed/{DS}_test.csv")
    pre = LogZ().fit(tr[feats].to_numpy())

    allowed_unc = np.array(midx)
    allowed_con = np.array([SCHEMA.idx[f] for f in SCHEMA.controllable])
    reachable = set(SCHEMA.controllable) | set(SCHEMA.depends_on_controllable())
    reach_mask = np.array([f in reachable for f in mfeats])
    json.dump({"reachable_model_features": [f for f in mfeats if f in reachable],
               "fixed_model_features": [f for f in mfeats if f not in reachable]},
              open(ROOT / f"results/tables/{DS}_reachability.json", "w"), indent=2)

    Xatt = te.loc[te.label == 1, feats].to_numpy()
    cats_all = te.loc[te.label == 1, "attack_cat"].to_numpy()
    rows, valid_rows, feat_rows = [], [], []
    for seed in SEEDS:
        for mname in MODELS:
            ck = ROOT / "results/tables/chunks"; ck.mkdir(parents=True, exist_ok=True)
            cf, cff = ck / f"{DS}{TAG}_s{seed}_{mname}.csv.gz", ck / f"{DS}{TAG}_s{seed}_{mname}_feat.csv"
            if cf.exists() and cff.exists():
                rows.append(pd.read_csv(cf)); feat_rows.extend(pd.read_csv(cff).to_dict("records"))
                print(f"seed={seed} {mname}: loaded checkpoint", flush=True)
                continue
            n_rows0, n_feat0 = len(rows), len(feat_rows)
            model = joblib.load(ROOT / f"models/{DS}_{mname}_seed{seed}.joblib")
            predict = lambda X, m=model: m.predict_proba(X[:, midx])[:, 1]  # noqa: E731
            det = predict(Xatt) >= 0.5
            rng = np.random.default_rng(seed)
            pick = rng.choice(np.flatnonzero(det), size=N_FLOWS, replace=False)
            X0, cats = Xatt[pick], cats_all[pick]
            expl = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
            P0 = shap_values(expl, X0[:, midx], mname)
            for mode, allowed in (("unconstrained", allowed_unc), ("constrained", allowed_con)):
                con = mode == "constrained"
                for eps in EPS:
                    for kind in KINDS:
                        if kind == "attack":
                            Xa = random_search(predict, pre, X0, SCHEMA, allowed, eps, con, seed)
                        else:
                            Xa = noise(pre, X0, SCHEMA, allowed, eps, con, seed)
                        pa = predict(Xa)
                        valid = SCHEMA.is_valid(Xa, X0, tol=1e-4)
                        P1 = shap_values(expl, Xa[:, midx], mname)
                        met = drift_metrics(P0, P1, reach_mask)
                        linf = np.abs(pre.transform(Xa)[:, midx] - pre.transform(X0)[:, midx]).max(1)
                        df = pd.DataFrame({"dataset": DS, "model": mname, "seed": seed, "mode": mode,
                                           "kind": kind, "eps": eps, "flow": np.arange(N_FLOWS),
                                           "attack_cat": cats, "p_adv": pa, "evaded": pa < 0.5,
                                           "valid": valid, "linf_z": linf, **met})
                        rows.append(df)
                        dd = np.abs(P1 - P0)
                        for j, f in enumerate(mfeats):
                            feat_rows.append({"model": mname, "seed": seed, "mode": mode, "kind": kind, "eps": eps, "feature": f,
                                              "role": ("controllable" if f in SCHEMA.controllable else "derived" if f in SCHEMA.derived else "fixed"),
                                              "mean_abs_dphi": dd[:, j].mean(), "mean_abs_phi_clean": np.abs(P0[:, j]).mean(),
                                              "mean_abs_dphi_evaded": dd[df.evaded.values, j].mean() if df.evaded.any() else np.nan})
                        print(f"seed={seed} {mname} {mode:13s} {kind:6s} eps={eps:<4} "
                              f"evade={df.evaded.mean():.3f} valid={valid.mean():.3f} "
                              f"drift={df.drift_mean.mean():.4f} top5={df.topk_overlap.mean():.3f} "
                              f"reach={df.reach_share.mean():.3f}", flush=True)
            pd.concat(rows[n_rows0:], ignore_index=True).to_csv(cf, index=False)
            pd.DataFrame(feat_rows[n_feat0:]).to_csv(cff, index=False)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(ROOT / f"results/tables/{DS}{TAG}_per_flow_results.csv.gz", index=False)
    pd.DataFrame(feat_rows).to_csv(ROOT / f"results/tables/{DS}{TAG}_per_feature_drift.csv", index=False)


if __name__ == "__main__":
    main()
