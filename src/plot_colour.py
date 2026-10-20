"""Re-draw the paper figures with a soft colour palette (same data and layout as analyze.py)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DS = sys.argv[1] if len(sys.argv) > 1 else "unsw"
T, F = ROOT / "results/tables", ROOT / "results/figures"
F.mkdir(parents=True, exist_ok=True)
MNAME = {"rf": "Random Forest", "xgb": "XGBoost"}
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.linewidth": 0.6,
                     "figure.dpi": 200, "savefig.bbox": "tight",
                     "axes.edgecolor": "#555555", "xtick.color": "#333333", "ytick.color": "#333333"})

# soft palette matching the method diagrams: muted orange = unconstrained, muted blue = constrained
COL = {"unconstrained": "#D4895A", "constrained": "#5B86B4"}
LIGHT = {"unconstrained": "#F2D3BC", "constrained": "#C9D8EA"}
BARS = {("rf", "unconstrained"): "#EDC3A3", ("rf", "constrained"): "#B4CBE3",
        ("xgb", "unconstrained"): "#C97D4E", ("xgb", "constrained"): "#4F78A6"}
CMAP = LinearSegmentedColormap.from_list("softblue", ["#FFFFFF", "#C9D8EA", "#5B86B4", "#2F4F75"])

R = pd.read_csv(T / f"{DS}_per_flow_results.csv.gz")
A = R[R.kind == "attack"]
eps = sorted(A.eps.unique())

# drift vs eps
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
for ax, m in zip(axes, ["rf", "xgb"]):
    for mode, mk in (("unconstrained", "o"), ("constrained", "s")):
        for kind, ls, al in (("attack", "-", 1.0), ("noise", "--", 0.75)):
            g = R[(R.model == m) & (R["mode"] == mode) & (R.kind == kind)]
            if g.empty:
                continue
            s = g.groupby(["eps", "seed"]).drift_mean.mean().groupby("eps").agg(["mean", "std"])
            ax.errorbar(s.index, s["mean"], yerr=s["std"], marker=mk, ls=ls, ms=3.2, lw=1.2, capsize=2,
                        color=COL[mode], alpha=al, mfc=COL[mode] if kind == "attack" else "white",
                        label=f"{mode} {kind}")
    ax.set_title(MNAME[m]); ax.set_xlabel(r"budget $\varepsilon$ (log-z $L_\infty$)")
    ax.set_ylabel(r"mean SHAP drift $\bar D$" + (" (prob.)" if m == "rf" else " (log-odds)"))
    ax.grid(alpha=0.35, lw=0.4, color="#BBBBBB")
axes[1].legend(fontsize=6.5, frameon=False)
fig.savefig(F / f"{DS}_fig_drift_vs_eps.pdf"); fig.savefig(F / f"{DS}_fig_drift_vs_eps.png"); plt.close(fig)

# evasion + top-5
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
for ax, m in zip(axes, ["rf", "xgb"]):
    for mode, mk in (("unconstrained", "o"), ("constrained", "s")):
        g = A[(A.model == m) & (A["mode"] == mode)]
        s = g.groupby(["eps", "seed"]).agg(ev=("evaded", "mean"), t5=("topk_overlap", "mean")).groupby("eps").mean()
        ax.plot(s.index, s.ev, marker=mk, color=COL[mode], lw=1.2, ms=3.2, label=f"{mode}: evasion rate")
        ax.plot(s.index, s.t5, marker=mk, color=COL[mode], lw=1.2, ms=3.2, ls=":", mfc="white",
                label=f"{mode}: top-5 overlap")
    ax.set_ylim(0, 1.05); ax.set_title(MNAME[m]); ax.set_xlabel(r"budget $\varepsilon$")
    ax.grid(alpha=0.35, lw=0.4, color="#BBBBBB")
axes[0].legend(fontsize=6, frameon=False, loc="lower right")
fig.savefig(F / f"{DS}_fig_evasion_top5.pdf"); fig.savefig(F / f"{DS}_fig_evasion_top5.png"); plt.close(fig)

# top-5 boxplot
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
for ax, m in zip(axes, ["rf", "xgb"]):
    data, labels, cols = [], [], []
    for mode in ("unconstrained", "constrained"):
        g = A[(A.model == m) & (A["mode"] == mode)]
        for ev_, lab in ((True, "evaded"), (False, "not evaded")):
            v = g[g.evaded == ev_].topk_overlap.values
            if len(v):
                data.append(v); labels.append(f"{mode[:5]}.\n{lab}"); cols.append(mode)
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#333333", lw=1))
    for patch, mode in zip(bp["boxes"], cols):
        patch.set_facecolor(LIGHT[mode]); patch.set_edgecolor(COL[mode])
    ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=6.5)
    ax.set_title(MNAME[m]); ax.grid(alpha=0.35, lw=0.4, axis="y", color="#BBBBBB")
axes[0].set_ylabel("top-5 overlap")
fig.savefig(F / f"{DS}_fig_top5_box.pdf"); fig.savefig(F / f"{DS}_fig_top5_box.png"); plt.close(fig)

# reachable-drift share
fig, ax = plt.subplots(figsize=(3.4, 2.2))
x = np.arange(len(eps)); w = 0.2
for k, (m, mode) in enumerate([(m, md) for m in ("rf", "xgb") for md in ("unconstrained", "constrained")]):
    s = A[(A.model == m) & (A["mode"] == mode)].groupby("eps").reach_share.mean()
    ax.bar(x + (k - 1.5) * w, s.values, w, label=f"{MNAME[m]}, {mode}",
           color=BARS[(m, mode)], edgecolor="#555555", lw=0.3)
ax.set_xticks(x, [str(e) for e in eps]); ax.set_xlabel(r"budget $\varepsilon$")
ax.set_ylabel("reachable-drift share"); ax.set_ylim(0, 1)
ax.grid(alpha=0.35, lw=0.4, axis="y", color="#BBBBBB"); ax.set_axisbelow(True)
ax.legend(fontsize=5.5, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
fig.savefig(F / f"{DS}_fig_reach_share.pdf"); fig.savefig(F / f"{DS}_fig_reach_share.png"); plt.close(fig)

# per-feature heatmap
fd = T / f"{DS}_featdrift_per_feature_drift.csv"
if not fd.exists():
    fd = T / f"{DS}_per_feature_drift.csv"
if fd.exists():
    P = pd.read_csv(fd); e0 = 1.0
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6)); fig.subplots_adjust(hspace=0.9)
    for ax, m in zip(axes, ["rf", "xgb"]):
        g = P[(P.model == m) & (P.eps == e0) & (P.kind == "attack") & (P.seed == P.seed.min())]
        piv = g.pivot_table(index="feature", columns="mode", values="mean_abs_dphi")[["unconstrained", "constrained"]]
        roles = g.drop_duplicates("feature").set_index("feature").role
        order = sorted(piv.index, key=lambda f: ({"controllable": 0, "derived": 1, "fixed": 2}[roles[f]], f))
        piv = piv.loc[order]; norm = piv / piv.values.max()
        ax.imshow(np.sqrt(norm.T.values), aspect="auto", cmap=CMAP, vmin=0, vmax=1)
        for (yy, xx), v in np.ndenumerate(norm.T.values):
            ax.text(xx, yy, f"{v:.2f}", ha="center", va="center", fontsize=4.5, color="w" if v > 0.3 else "#222222")
        ax.set_yticks([0, 1], ["unconstr.", "constr."])
        ax.set_xticks(range(len(order)), [f + {"controllable": " (C)", "derived": " (D)", "fixed": ""}[roles[f]] for f in order],
                      rotation=60, fontsize=6, ha="right")
        ax.set_title(f"{MNAME[m]}, $\\varepsilon$={e0}")
    fig.savefig(F / f"{DS}_fig_feature_heatmap.pdf"); fig.savefig(F / f"{DS}_fig_feature_heatmap.png"); plt.close(fig)
print("done", DS)
