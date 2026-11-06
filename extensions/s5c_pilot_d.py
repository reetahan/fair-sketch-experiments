#!/usr/bin/env python3
"""Section 5 classification: proxy distortion against pilot size in multiples of d, at a fixed budget m."""
import argparse, glob, os, sys, time
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
import lib_classification as cl
from lib_classification import P, make_data
from lib_coreset import run_coreset
from lib_data import real_class_problems, add_real_class_args
from fs_common import PALETTE, legend_above, set_style
from fs_common import ensure_scienceplots

PILOTS = [1, 2.5, 5, 10, 20, 40, 80]
REFS = ["c_unif", "c_lev"]
PROXIES = ["c_tgt_proxy", "c_strat_tgt_proxy"]
STYLE = {"c_lev": ("Leverage", PALETTE["lev"], "--"), "c_tgt_proxy": ("Targeted, proxy", PALETTE["proxy"], "-"),
         "c_strat_tgt_proxy": ("Stratified targeted, proxy", PALETTE["strat_proxy"], "-")}
NAMES = {"law": "LSAC", "acs": "ACS Income", "hmda": "HMDA", "diab": "130-Hospitals"}


def sweep(p, reps, rng):
    frames = []
    for m in [k * p.d for k in (40, 160 if 160 * p.d < p.n // 4 else 80)]:
        p.m_p = None
        ref = cl.experiment(p, m, reps, rng, meths=REFS, runner=run_coreset)
        frames.append(ref.assign(budget=m / p.d, pilot=np.nan, d=p.d))
        for k in [k for k in PILOTS if k * p.d <= m / 2]:
            p.m_p = int(round(k * p.d))
            d = cl.experiment(p, m, reps, rng, meths=PROXIES, runner=run_coreset)
            frames.append(d.assign(budget=m / p.d, pilot=k, d=p.d))
            print(f"  m={m} pilot={k}d: done", flush=True)
    p.m_p = None
    return pd.concat(frames)


def ratios(df, min_base=0.01):
    ok = df.groupby("metric").base.first().abs()
    df = df[df.metric.isin(ok[ok >= min_base].index)]
    u = df[df.method == "c_unif"].set_index(["budget", "metric"]).mean_abs_rel.rename("u")
    df = df.join(u, on=["budget", "metric"])
    return df.assign(ratio=df.mean_abs_rel / df.u)


def combine(out):
    frames = {}
    for f in glob.glob(f"{out}/s5c_pilot_d_*.csv"):
        frames[os.path.basename(f)[len("s5c_pilot_d_"):].split("-")[0].replace(".csv", "")] = ratios(pd.read_csv(f))
    names = [k for k in NAMES if k in frames]
    fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 5.6), squeeze=False, sharex=True)
    for j, name in enumerate(names):
        d = frames[name]
        for i, b in enumerate(sorted(d.budget.unique())):
            ax, g = axes[i, j], d[d.budget == b]
            gm = lambda v: float(np.exp(np.log(v).mean()))
            for me, (lab, col, ls) in STYLE.items():
                h = g[g.method == me]
                if me == "c_lev":
                    ax.axhline(gm(h.ratio), color=col, ls=ls, lw=1.2, label=lab)
                    continue
                s = h.groupby("pilot").ratio.apply(gm)
                ax.plot(s.index, s.values, "o-", ms=4, color=col, label=lab)
            ax.axhline(1, color="k", lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks(PILOTS[::2])
            ax.set_xticklabels([f"{v:g}" for v in PILOTS[::2]])
            ax.set_title(f"{NAMES[name]}, $m={b:g}d$")
            if j == 0:
                ax.set_ylabel("distortion / uniform")
            if i == 1:
                ax.set_xlabel("pilot rows / $d$")
    legend_above(fig, axes, 0.93)
    fig.savefig(f"{out}/fig_pilot_d.png", dpi=200)
    print(f"wrote {out}/fig_pilot_d.png")


def main():
    ap = add_real_class_args(cl.add_args(argparse.ArgumentParser(), "results/pilot_d_s5c", m=300, T=3, synth=True))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--combine", action="store_true", help="draw the figure from the CSVs in --out")
    ap.set_defaults(data_seed=3)
    a = ap.parse_args()
    ensure_scienceplots()
    if a.combine:
        set_style(fontsize=11)
        combine(a.out)
        return
    rng = cl.start(a)
    probs = [P(*make_data(a.n, a.d, frac1=0.1, seed=a.data_seed))] if a.synthetic else real_class_problems(a)
    t0 = time.time()
    for p in probs:
        name = "synth" if a.synthetic else p.name
        print(f"[{name}] n={p.n} d={p.d}", flush=True)
        sweep(p, a.reps, rng).to_csv(f"{a.out}/s5c_pilot_d_{name}.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()
