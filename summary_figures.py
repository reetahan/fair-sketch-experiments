#!/usr/bin/env python3
"""Two compact summary figures for the main text, drawn from the result CSVs (no experiments are re-run).

fig_s4_summary.png  (Section 4): mean |relative change| against the sketch size m, one column per real data set, log-log axes.
    row 1  regression IF        (results/reg_s4_real/real_m_<ds>.csv;     one line per sketch family)
    row 2  classification IF    (results/class_s4_real/s4c_<ds>_m.csv;    uniform, importance, LogReg-sketch)
    row 3  classification DP    (same file)
    A dashed line of slope -1/2 shows the 1/sqrt(m) rate. Points where more than half the regression sketches were
    rank deficient, or where the exact gap is below --min_base, are not drawn.
    Row 3 plots the mean absolute change of the DP gap, which needs no nonzero exact gap.
fig_s5_summary.png  (Section 5, classification): distortion of each method divided by that of uniform sampling at the
    same m (values below 1 mean the method helps), one column per real data set, log axes.
    row 1  IF, row 2  DP gap                                          (results/class_s5_real/s5c_<ds>_m.csv)
fig_s4_lottery.png  (Section 4): HMDA loan approval, the compressed model's DP gap against m with its 10th-90th
    percentile band over repeated sketches, for uniform and importance sampling, against the full-data gap.
fig_s5_reg_predict.png  (Section 5, regression): predicted against observed benefit of stratification. One point per
    real data set or LSAC ablation, sketch family, metric and sketch size (results/reg_s5_real/s5_real_*.csv).
    x: analytic ratio O/V of the stratified oracle's second moment to the plain sketch's (column J_over_vanilla);
    y: observed ratio of the same second moments over repeated sketches (ratio of squared emp_rms).
    Values below 1 mean stratification helps; settings where more than half the sketches were rank deficient are dropped.
Usage: python summary_figures.py --results results --out figures [--min_base 0.01]"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

REG = [("law-race", "LSAC"), ("acs-CA2018-sex", "ACS"), ("hmda-CA2018-race-rate", "HMDA"),
       ("enem-ch-race", "ENEM"), ("diab-length_of_stay", "Diabetes")]
CLS = ["law-pass_bar", "acs-CA2018-sex-income50k", "hmda-CA2018-race-approval", "enem-ch-race-above_median", "diab-readmit30"]
FAM = {"gauss": ("Gaussian", "tab:green"), "cs": ("CountSketch", "tab:orange"), "is": ("Importance", "tab:red"),
       "unif": ("Uniform", "tab:blue")}
S4C = {"samp_unif": ("Uniform", "tab:blue"), "samp_imp": ("Importance", "tab:red"),
       "logreg_sketch": ("LogReg-sketch", "tab:purple")}
S5C = {"c_imp": ("Importance (Sec. 3)", "tab:gray"), "c_tgt_oracle": ("Targeted, oracle", "tab:red"),
       "c_tgt_proxy": ("Targeted, proxy", "tab:pink"), "c_strat_tgt_proxy": ("Stratified targeted, proxy", "tab:purple"),
       "logreg_code": ("LogReg-sketch", "tab:olive"), "newton_cs": ("Newton sketch, 3 steps", "tab:orange")}


def read(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def empty(ax, msg):
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=7, transform=ax.transAxes, color="0.4")


def slope_guide(ax, xs, ys):
    """Dashed 1/sqrt(m) line through the geometric middle of the plotted points."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    if ok.sum() < 2:
        return
    x0, y0 = np.exp(np.log(xs[ok]).mean()), np.exp(np.log(ys[ok]).mean())
    g = np.array([xs[ok].min(), xs[ok].max()])
    ax.plot(g, y0 * np.sqrt(x0 / g), "k--", lw=0.8, label=r"$\propto 1/\sqrt{m}$")


def fig_s4(res, out, min_base):
    fig, axes = plt.subplots(3, len(REG), figsize=(2.6 * len(REG), 6.6), squeeze=False)
    for j, ((rds, name), cds) in enumerate(zip(REG, CLS)):
        axes[0, j].set_title(name, fontsize=9)
        d = read(f"{res}/reg_s4_real/real_m_{rds}.csv")
        ax = axes[0, j]
        if d is None:
            empty(ax, "no data")
        else:
            d = d[(d.metric == "IF") & (d.frac_failed <= 0.5)]
            allx, ally = [], []
            for f, (lab, col) in FAM.items():
                g = d[d.family == f].sort_values("m")
                if len(g):
                    ax.plot(g.m, g.mean_abs_rel, "o-", ms=3, color=col, label=lab)
                    allx += list(g.m); ally += list(g.mean_abs_rel)
            slope_guide(ax, allx, ally)
        c = read(f"{res}/class_s4_real/s4c_{cds}_m.csv")
        for i, met in ((1, "IF"), (2, "DP")):
            ax = axes[i, j]
            if c is None:
                empty(ax, "no data"); continue
            g0 = c[c.metric == met]
            if met == "IF" and len(g0) and abs(g0.base.iloc[0]) < min_base:
                empty(ax, f"exact value below {min_base}"); continue
            allx, ally = [], []
            for me, (lab, col) in S4C.items():
                g = g0[g0.method == me].sort_values("m")
                if len(g):
                    yv = g.mean_abs_rel if met == "IF" else g.mean_abs_rel * g.base.abs()   # DP: absolute change
                    ax.plot(g.m, yv, "o-", ms=3, color=col, label=lab)
                    allx += list(g.m); ally += list(yv)
            slope_guide(ax, allx, ally)
    for ax in axes.ravel():
        if ax.lines:
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=7)
    for i, lab in enumerate(["Regression: IF\nmean |rel. change|", "Classification: IF\nmean |rel. change|",
                             "Classification: DP gap\nmean |abs. change|"]):
        axes[i, 0].set_ylabel(lab, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("sketch size $m$", fontsize=8)
    for i in range(3):
        h, l = next(((a.get_legend_handles_labels()) for a in axes[i] if a.lines), ([], []))
        axes[i, -1].legend(h, l, fontsize=6, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{out}/fig_s4_summary.png", dpi=200)
    plt.close(fig)


def fig_s5(res, out, min_base):
    fig, axes = plt.subplots(2, len(CLS), figsize=(2.6 * len(CLS), 4.6), squeeze=False)
    for j, ((_, name), cds) in enumerate(zip(REG, CLS)):
        axes[0, j].set_title(name, fontsize=9)
        c = read(f"{res}/class_s5_real/s5c_{cds}_m.csv")
        for i, met in ((0, "IF"), (1, "DP")):
            ax = axes[i, j]
            if c is None:
                empty(ax, "no data"); continue
            p = c[c.metric == met].pivot_table(index="m", columns="method", values="mean_abs_rel")
            for me, (lab, col) in S5C.items():
                if me in p and "c_unif" in p:          # the exact gap cancels in the ratio, so no mask is needed
                    ax.plot(p.index, p[me] / p["c_unif"], "o-", ms=3, color=col, label=lab)
    for ax in axes.ravel():
        if ax.lines:
            ax.axhline(1, color="k", lw=0.8)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(labelsize=7)
    for i, lab in enumerate(["IF\nmethod / uniform", "DP gap\nmethod / uniform"]):
        axes[i, 0].set_ylabel(lab, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("sketch size $m$", fontsize=8)
    h, l = next(((a.get_legend_handles_labels()) for a in axes[0] if a.lines), ([], []))
    fig.legend(h, l, fontsize=7, loc="lower center", ncol=len(l), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(f"{out}/fig_s5_summary.png", dpi=200)
    plt.close(fig)


ABL = {"ablate_noise1": "LSAC, noisier majority", "ablate_scale_col": "LSAC, worse conditioning",
       "ablate_subsample0": "LSAC, smaller minority"}
DSCOL = {"LSAC": "tab:blue", "ACS": "tab:orange", "HMDA": "tab:green", "ENEM": "tab:red", "Diabetes": "tab:purple",
         "LSAC, noisier majority": "tab:cyan", "LSAC, worse conditioning": "tab:olive", "LSAC, smaller minority": "tab:pink"}
MARK = {"gauss": "o", "cs": "s", "unif": "^", "is": "D"}


def fig_s5_reg_predict(res, out):
    pts = []
    files = [(f"{res}/reg_s5_real/s5_real_m_{k}.csv", name, "m") for k, name in REG]
    files += [(f"{res}/reg_s5_real/s5_real_{k}.csv", name, "x") for k, name in ABL.items()]
    for path, name, xcol in files:
        d = read(path)
        if d is None:
            continue
        d = d[d.frac_failed <= 0.5]
        for (fam, met, x), g in d.groupby(["family", "metric", xcol]):
            v = g.set_index("method")
            if not {"vanilla", "oracle"} <= set(v.index):
                continue
            pred = v.loc["oracle", "J_over_vanilla"]
            obs = (v.loc["oracle", "emp_rms"] / v.loc["vanilla", "emp_rms"]) ** 2
            if np.isfinite(pred) and np.isfinite(obs) and pred > 0 and obs > 0:
                pts.append((name, fam, pred, obs))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    if not pts:
        empty(ax, "no data")
    else:
        P = pd.DataFrame(pts, columns=["data", "fam", "pred", "obs"])
        for name, g in P.groupby("data", sort=False):
            for fam, h in g.groupby("fam"):
                ax.scatter(h.pred, h.obs, s=14, marker=MARK.get(fam, "o"), color=DSCOL.get(name, "0.5"), alpha=0.7,
                           linewidths=0)
        xlo, xhi = P.pred.min() - 0.02, P.pred.max() + 0.02
        ax.plot([xlo, xhi], [xlo, xhi], "k--", lw=0.8, label="observed = predicted")
        ax.axhline(1, color="k", lw=0.6); ax.axvline(1, color="k", lw=0.6)
        ax.set_xlim(xlo, xhi); ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        from matplotlib.ticker import FixedLocator, FixedFormatter
        yt = [0.25, 0.5, 1, 2, 4]
        ax.yaxis.set_major_locator(FixedLocator(yt)); ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in yt]))
        from matplotlib.lines import Line2D
        hs = [Line2D([], [], ls="", marker="o", color=DSCOL[n], label=n) for n in DSCOL if n in set(P.data)]
        hs += [Line2D([], [], ls="", marker=MARK[f], color="0.3", label=FAM[f][0]) for f in MARK if f in set(P.fam)]
        hs += [Line2D([], [], ls="--", color="k", lw=0.8, label="observed = predicted")]
        ax.legend(handles=hs, fontsize=6, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        small = P[P.pred < 0.9]
        if len(small):
            ax.set_title(f"predicted gain >10%: stratification wins in {100*(small.obs<1).mean():.0f}% "
                         f"of {len(small)} settings", fontsize=7)
    ax.set_xlabel(r"predicted ratio $\mathcal{O}/\mathcal{V}$ (stratified / plain)", fontsize=8)
    ax.set_ylabel("observed ratio of mean squared distortion", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_s5_reg_predict.png", dpi=200)
    plt.close(fig)


def fig_s4_lottery(res, out, ds="hmda-CA2018-race-approval", name="HMDA loan approval", met="DP"):
    """One data set: the sketched model's group gap (|gap|, percentage points) against m, mean and 10th-90th percentile
    band over repeated sketches, for uniform and importance sampling; the dashed line is the full-data gap."""
    c = read(f"{res}/class_s4_real/s4c_{ds}_m.csv")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    if c is None:
        empty(ax, "no data")
    else:
        g0 = c[c.metric == met]
        base = abs(g0.base.iloc[0])
        for me, (lab, col) in (("samp_unif", S4C["samp_unif"]), ("samp_imp", S4C["samp_imp"])):
            g = g0[g0.method == me].sort_values("m")
            ax.fill_between(g.m, 100 * base * (1 + g.q10_rel), 100 * base * (1 + g.q90_rel), color=col, alpha=0.2, lw=0)
            ax.plot(g.m, 100 * base * (1 + g.mean_rel), "o-", ms=3, color=col, label=f"{lab}: mean and 10th to 90th percentile")
        ax.axhline(100 * base, color="k", ls="--", lw=0.9, label=f"full-data model ({100 * base:.1f} points)")
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=6, loc="upper right", frameon=False)
    ax.set_xlabel("sketch size $m$", fontsize=8)
    ax.set_ylabel(f"{name}: {met} gap of the\ncompressed model (percentage points)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(f"{out}/fig_s4_lottery.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--min_base", type=float, default=0.01)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_s4(a.results, a.out, a.min_base)
    fig_s5(a.results, a.out, a.min_base)
    fig_s5_reg_predict(a.results, a.out)
    fig_s4_lottery(a.results, a.out)
    print(f"wrote fig_s4_summary.png, fig_s4_lottery.png, fig_s5_summary.png and fig_s5_reg_predict.png to {a.out}/")
