"""Phase 6: tables, ESI, statistical tests (RQ1-RQ3) and figures."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = sys.argv[1] if len(sys.argv) > 1 else "unsw"
T, F = ROOT / "results/tables", ROOT / "results/figures"
MNAME = {"rf": "Random Forest", "xgb": "XGBoost"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def rank_biserial(x, y):
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = pd.Series(np.abs(d)).rank().values
    return float((r[d > 0].sum() - r[d < 0].sum()) / r.sum())


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def esi(eps, drift):
    e = np.concatenate([[0.0], eps])
    d = np.concatenate([[0.0], drift]) / max(drift[-1], 1e-12)
    return 1 - np.trapezoid(d, e) / e[-1]


def main() -> None:
    R = pd.read_csv(T / f"{DS}_per_flow_results.csv.gz")
    A = R[R.kind == "attack"]
    out = {}

    # ---------------- robustness table
    rob = (A.groupby(["model", "mode", "eps", "seed"])
           .agg(evasion=("evaded", "mean"), valid=("valid", "mean"), linf=("linf_z", "mean"))
           .groupby(["model", "mode", "eps"]).agg(["mean", "std"]))
    rob.to_csv(T / f"{DS}_robustness.csv")

    # ---------------- drift summary
    met = ["drift_mean", "topk_overlap", "spearman", "reach_share", "fixed_drift_mean", "evaded"]
    summ = (R.groupby(["model", "mode", "kind", "eps", "seed"])[met].mean()
            .groupby(["model", "mode", "kind", "eps"]).agg(["mean", "std"]))
    summ.to_csv(T / f"{DS}_drift_summary.csv")

    # ---------------- ESI per model x seed x mode
    esi_rows = []
    for (m, mode, s), g in A.groupby(["model", "mode", "seed"]):
        c = g.groupby("eps").drift_mean.mean().sort_index()
        esi_rows.append({"model": m, "mode": mode, "seed": s, "ESI": esi(c.index.values, c.values),
                         "drift_at_max_eps": c.values[-1]})
    E = pd.DataFrame(esi_rows)
    E.groupby(["model", "mode"]).ESI.agg(["mean", "std"]).to_csv(T / f"{DS}_esi.csv")
    out["ESI"] = E.groupby(["model", "mode"]).ESI.agg(["mean", "std"]).round(3).reset_index().to_dict("records")

    tests = []
    key = ["model", "seed", "eps", "flow"]
    U = A[A["mode"] == "unconstrained"].set_index(key)
    C = A[A["mode"] == "constrained"].set_index(key)
    J = U.join(C, lsuffix="_u", rsuffix="_c", how="inner").reset_index()

    # ---------------- RQ1a: same budget, paired by flow
    for (m, e), g in J.groupby(["model", "eps"]):
        st = wilcoxon(g.drift_mean_c, g.drift_mean_u)
        tests.append({"rq": "RQ1a same-eps drift (con vs unc)", "model": m, "eps": e, "n": len(g),
                      "median_con": g.drift_mean_c.median(), "median_unc": g.drift_mean_u.median(),
                      "effect_rb": rank_biserial(g.drift_mean_c, g.drift_mean_u), "p": st.pvalue})
        st = wilcoxon(g.topk_overlap_c, g.topk_overlap_u)
        tests.append({"rq": "RQ1a same-eps top5 overlap (con vs unc)", "model": m, "eps": e, "n": len(g),
                      "median_con": g.topk_overlap_c.mean(), "median_unc": g.topk_overlap_u.mean(),
                      "effect_rb": rank_biserial(g.topk_overlap_c, g.topk_overlap_u), "p": st.pvalue})

    # ---------------- RQ1b: success-conditioned (flows that evaded in BOTH modes at the same eps)
    for (m, e), g in J[J.evaded_u & J.evaded_c].groupby(["model", "eps"]):
        if len(g) < 20:
            continue
        st = wilcoxon(g.drift_mean_c, g.drift_mean_u)
        tests.append({"rq": "RQ1b both-evaded drift (con vs unc)", "model": m, "eps": e, "n": len(g),
                      "median_con": g.drift_mean_c.median(), "median_unc": g.drift_mean_u.median(),
                      "effect_rb": rank_biserial(g.drift_mean_c, g.drift_mean_u), "p": st.pvalue})

    # ---------------- RQ1c: evasion-rate-matched budgets (paired by flow)
    ev = A.groupby(["model", "mode", "eps"]).evaded.mean()
    match_rows = []
    for m in A.model.unique():
        for ec in sorted(A.eps.unique()):
            rc = ev[(m, "constrained", ec)]
            cand = ev.loc[(m, "unconstrained")]
            eu = float((cand - rc).abs().idxmin())
            gc = A[(A.model == m) & (A["mode"] == "constrained") & (A.eps == ec)].set_index(["seed", "flow"])
            gu = A[(A.model == m) & (A["mode"] == "unconstrained") & (A.eps == eu)].set_index(["seed", "flow"])
            g = gc.join(gu, lsuffix="_c", rsuffix="_u", how="inner")
            st = wilcoxon(g.drift_mean_c, g.drift_mean_u)
            match_rows.append({"model": m, "eps_con": ec, "evasion_con": rc, "eps_unc": eu,
                               "evasion_unc": cand[eu], "median_drift_con": g.drift_mean_c.median(),
                               "median_drift_unc": g.drift_mean_u.median(),
                               "top5_con": g.topk_overlap_c.mean(), "top5_unc": g.topk_overlap_u.mean()})
            tests.append({"rq": "RQ1c evasion-matched drift (con vs unc)", "model": m, "eps": ec,
                          "eps_unc": eu, "n": len(g), "median_con": g.drift_mean_c.median(),
                          "median_unc": g.drift_mean_u.median(),
                          "effect_rb": rank_biserial(g.drift_mean_c, g.drift_mean_u), "p": st.pvalue})
    pd.DataFrame(match_rows).to_csv(T / f"{DS}_evasion_matched.csv", index=False)

    # ---------------- attack vs noise excess drift
    N = R[R.kind == "noise"].set_index(["model", "seed", "mode", "eps", "flow"])
    if len(N):
        AN = A.set_index(["model", "seed", "mode", "eps", "flow"]).join(N, rsuffix="_n", how="inner").reset_index()
        for (m, mode), g in AN.groupby(["model", "mode"]):
            st = wilcoxon(g.drift_mean, g.drift_mean_n)
            tests.append({"rq": "attack vs noise drift", "model": m, "mode": mode, "n": len(g),
                          "median_con": g.drift_mean.median(), "median_unc": g.drift_mean_n.median(),
                          "effect_rb": rank_biserial(g.drift_mean, g.drift_mean_n), "p": st.pvalue})

    # ---------------- RQ2: reachable share and fixed-feature spill-over
    for m, g in J.groupby("model"):
        st = wilcoxon(g.reach_share_c, g.reach_share_u)
        tests.append({"rq": "RQ2 reachable-drift share (con vs unc)", "model": m, "n": len(g),
                      "median_con": g.reach_share_c.median(), "median_unc": g.reach_share_u.median(),
                      "effect_rb": rank_biserial(g.reach_share_c, g.reach_share_u), "p": st.pvalue})
    spill = (C.reset_index().assign(spill=lambda d: d.fixed_drift_mean > 1e-9)
             .groupby("model").agg(share_flows_with_fixed_drift=("spill", "mean"),
                                   median_reach_share=("reach_share", "median")))
    out["RQ2_spillover_constrained"] = spill.round(3).reset_index().to_dict("records")

    # ---------------- RQ3: evaded vs not evaded (constrained)
    for (m, e), g in C.reset_index().groupby(["model", "eps"]):
        a, b = g[g.evaded].drift_mean, g[~g.evaded].drift_mean
        if min(len(a), len(b)) < 20:
            continue
        st = mannwhitneyu(a, b)
        tests.append({"rq": "RQ3 evaded vs not (constrained drift)", "model": m, "eps": e,
                      "n": f"{len(a)}/{len(b)}", "median_con": a.median(), "median_unc": b.median(),
                      "effect_rb": 2 * st.statistic / (len(a) * len(b)) - 1, "p": st.pvalue})
    for (m, e), g in U.reset_index().groupby(["model", "eps"]):
        a, b = g[g.evaded].drift_mean, g[~g.evaded].drift_mean
        if min(len(a), len(b)) < 20:
            continue
        st = mannwhitneyu(a, b)
        tests.append({"rq": "RQ3 evaded vs not (unconstrained drift)", "model": m, "eps": e,
                      "n": f"{len(a)}/{len(b)}", "median_con": a.median(), "median_unc": b.median(),
                      "effect_rb": 2 * st.statistic / (len(a) * len(b)) - 1, "p": st.pvalue})

    TT = pd.DataFrame(tests)
    TT["p_holm"] = holm(TT.p.values)
    TT.to_csv(T / f"{DS}_stat_tests.csv", index=False)
    json.dump(out, open(T / f"{DS}_key_numbers.json", "w"), indent=2, default=float)

    # ================= figures
    eps = sorted(A.eps.unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, m in zip(axes, ["rf", "xgb"]):
        for mode, mk in (("unconstrained", "o"), ("constrained", "s")):
            for kind, ls in (("attack", "-"), ("noise", "--")):
                g = R[(R.model == m) & (R["mode"] == mode) & (R.kind == kind)]
                if g.empty:
                    continue
                s = g.groupby(["eps", "seed"]).drift_mean.mean().groupby("eps").agg(["mean", "std"])
                ax.errorbar(s.index, s["mean"], yerr=s["std"], marker=mk, ls=ls, ms=3, lw=1, capsize=2,
                            color="k" if mode == "unconstrained" else "0.55",
                            label=f"{mode} {kind}")
        ax.set_title(MNAME[m]); ax.set_xlabel(r"budget $\varepsilon$ (log-z $L_\infty$)")
        ax.set_ylabel(r"mean SHAP drift $\bar D$" + (" (prob.)" if m == "rf" else " (log-odds)"))
        ax.grid(alpha=0.3, lw=0.4)
    axes[1].legend(fontsize=6.5, frameon=False)
    fig.savefig(F / f"{DS}_fig_drift_vs_eps.pdf"); fig.savefig(F / f"{DS}_fig_drift_vs_eps.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for ax, m in zip(axes, ["rf", "xgb"]):
        for mode, mk in (("unconstrained", "o"), ("constrained", "s")):
            g = A[(A.model == m) & (A["mode"] == mode)]
            s = g.groupby(["eps", "seed"]).agg(ev=("evaded", "mean"), t5=("topk_overlap", "mean"))
            s = s.groupby("eps").mean()
            c = "k" if mode == "unconstrained" else "0.55"
            ax.plot(s.index, s.ev, marker=mk, color=c, lw=1, ms=3, label=f"{mode}: evasion rate")
            ax.plot(s.index, s.t5, marker=mk, color=c, lw=1, ms=3, ls=":", label=f"{mode}: top-5 overlap")
        ax.set_ylim(0, 1.05); ax.set_title(MNAME[m]); ax.set_xlabel(r"budget $\varepsilon$")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")
    fig.savefig(F / f"{DS}_fig_evasion_top5.pdf"); fig.savefig(F / f"{DS}_fig_evasion_top5.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    for ax, m in zip(axes, ["rf", "xgb"]):
        data, labels = [], []
        for mode in ("unconstrained", "constrained"):
            g = A[(A.model == m) & (A["mode"] == mode)]
            for ev_, lab in ((True, "evaded"), (False, "not evaded")):
                v = g[g.evaded == ev_].topk_overlap.values
                if len(v):
                    data.append(v); labels.append(f"{mode[:5]}.\n{lab}")
        ax.boxplot(data, widths=0.5, showfliers=False)
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=6.5)
        ax.set_title(MNAME[m]); ax.grid(alpha=0.3, lw=0.4, axis="y")
    axes[0].set_ylabel("top-5 overlap")
    fig.savefig(F / f"{DS}_fig_top5_box.pdf"); fig.savefig(F / f"{DS}_fig_top5_box.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(eps)); w = 0.2
    for k, (m, mode) in enumerate([(m, md) for m in ("rf", "xgb") for md in ("unconstrained", "constrained")]):
        s = A[(A.model == m) & (A["mode"] == mode)].groupby("eps").reach_share.mean()
        ax.bar(x + (k - 1.5) * w, s.values, w, label=f"{MNAME[m]}, {mode}",
               color=["0.15", "0.6", "0.35", "0.85"][k], edgecolor="k", lw=0.3)
    ax.set_xticks(x, [str(e) for e in eps]); ax.set_xlabel(r"budget $\varepsilon$")
    ax.set_ylabel("reachable-drift share"); ax.set_ylim(0, 1); ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    fig.savefig(F / f"{DS}_fig_reach_share.pdf"); fig.savefig(F / f"{DS}_fig_reach_share.png"); plt.close(fig)

    fd = T / f"{DS}_featdrift_per_feature_drift.csv"
    if not fd.exists():
        fd = T / f"{DS}_per_feature_drift.csv"
    if fd.exists():
        P = pd.read_csv(fd)
        e0 = 1.0
        fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6))
        fig.subplots_adjust(hspace=0.9)
        for ax, m in zip(axes, ["rf", "xgb"]):
            g = P[(P.model == m) & (P.eps == e0) & (P.kind == "attack") & (P.seed == P.seed.min())]
            piv = g.pivot_table(index="feature", columns="mode", values="mean_abs_dphi")[["unconstrained", "constrained"]]
            roles = g.drop_duplicates("feature").set_index("feature").role
            order = sorted(piv.index, key=lambda f: ({"controllable": 0, "derived": 1, "fixed": 2}[roles[f]], f))
            piv = piv.loc[order]
            norm = piv / piv.values.max()
            im = ax.imshow(np.sqrt(norm.T.values), aspect="auto", cmap="Greys", vmin=0, vmax=1)
            for (yy, xx), v in np.ndenumerate(norm.T.values):
                ax.text(xx, yy, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color="w" if v > 0.3 else "k")
            ax.set_yticks([0, 1], ["unconstr.", "constr."])
            ax.set_xticks(range(len(order)), [f + {"controllable": " (C)", "derived": " (D)", "fixed": ""}[roles[f]] for f in order],
                          rotation=60, fontsize=6, ha="right")
            ax.set_title(f"{MNAME[m]}, $\\varepsilon$={e0}")
        fig.savefig(F / f"{DS}_fig_feature_heatmap.pdf"); fig.savefig(F / f"{DS}_fig_feature_heatmap.png"); plt.close(fig)

    print(rob.round(3).to_string())
    print(E.groupby(["model", "mode"]).ESI.agg(["mean", "std"]).round(3))
    print(pd.DataFrame(match_rows).round(4).to_string())
    print(TT.round(4).to_string())
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
