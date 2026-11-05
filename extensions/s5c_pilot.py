#!/usr/bin/env python3
"""Section 5 classification: distortion of the targeted proxy against the pilot fraction f = m_p / m."""
import argparse, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
import lib_classification as cl
from lib_classification import P, make_data, MET
from lib_coreset import run_coreset
from lib_data import real_class_problems, add_real_class_args
from fs_common import style_axes, legend_above, PALETTE
from replot_all import ensure_scienceplots

METH = ["c_unif", "c_tgt_oracle", "c_tgt_proxy", "c_strat_tgt_proxy"]
FRACS = [0.05, 0.1, 0.25, 0.5, 0.75]
M_FACTORS = [10, 40, 160]
STYLE = {
    "c_tgt_oracle": ("targeted, oracle", PALETTE["oracle"]),
    "c_tgt_proxy": ("targeted, proxy", PALETTE["proxy"]),
    "c_strat_tgt_proxy": ("stratified targeted, proxy", PALETTE["strat_proxy"]),
}


def pilot_sweep(p, reps, rng, fracs, factors):
    frames = []
    for m in [k * p.d for k in factors if k * p.d < p.n // 4]:
        for f in fracs:
            p.m_p = max(int(round(f * m)), 1)
            d = cl.experiment(p, m, reps, rng, meths=METH, runner=run_coreset)
            frames.append(d.assign(f=f, m_p=p.m_p))
            print(f"  m={m} f={f}: done", flush=True)
    p.m_p = None
    return pd.concat(frames)


def ratio_table(df):
    uni = df[df.method == "c_unif"].groupby(["m", "metric"]).mean_abs_rel.mean().rename("uniform")
    d = df[df.method != "c_unif"].join(uni, on=["m", "metric"])
    d["ratio"] = d.mean_abs_rel / d.uniform
    return d


def plot(df, path):
    d = ratio_table(df)
    ms = sorted(d.m.unique())
    fig, axes = plt.subplots(len(ms), len(MET), figsize=(3.4 * len(MET), 2.9 * len(ms) + 0.6), squeeze=False, sharex=True)
    for i, m in enumerate(ms):
        for j, met in enumerate(MET):
            ax = axes[i, j]
            g = d[(d.m == m) & (d.metric == met)]
            for meth, (lab, col) in STYLE.items():
                h = g[g.method == meth].sort_values("f")
                ax.plot(h.f, h.ratio, "o-", color=col, label=lab)
            ax.axhline(1, color="k", lw=0.8)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(met)
            if i == len(ms) - 1:
                ax.set_xlabel("pilot fraction $f$")
            if j == 0:
                ax.set_ylabel(f"$m={m}$")
    style_axes(axes)
    legend_above(fig, axes, 1 - 0.6 / (2.9 * len(ms) + 0.6))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = add_real_class_args(cl.add_args(argparse.ArgumentParser(), "results/pilot_s5c", m=300, T=3, synth=True))
    ap.add_argument("--synthetic", action="store_true")
    ap.set_defaults(data_seed=3)
    a = ap.parse_args()
    ensure_scienceplots()
    rng = cl.start(a)
    fracs, factors = ([0.1, 0.5], [10, 40]) if a.quick else (FRACS, M_FACTORS)
    probs = [P(*make_data(a.n, a.d, frac1=0.1, seed=a.data_seed))] if a.synthetic else real_class_problems(a)
    t0 = time.time()
    for p in probs:
        print(f"[{p.name}] n={p.n} d={p.d}", flush=True)
        stem = f"{a.out}/s5c_pilot_{'synth' if a.synthetic else p.name}"
        df = pilot_sweep(p, a.reps, rng, fracs, factors)
        df.to_csv(f"{stem}.csv", index=False)
        plot(df, f"{stem}.png")
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()
