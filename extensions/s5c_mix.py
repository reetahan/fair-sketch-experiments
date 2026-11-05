#!/usr/bin/env python3
"""Section 5 classification: proxy sampling rule (targeted, leverage reference) against the uniform mixing weight mu."""
import argparse, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import matplotlib.pyplot as plt
import lib_classification as cl
import lib_coreset as lc
from lib_classification import P, make_data, MET, fit
from lib_data import real_class_problems, add_real_class_args
from fs_common import style_axes, legend_above, PALETTE
from fs_common import ensure_scienceplots

METH = ["unif", "tgt_oracle", "tgt_proxy", "lev"]
MUS = [0.2, 0.5, 0.8]
M_FACTORS = [10, 40, 160]
STYLE = {
    "tgt_oracle": ("targeted, oracle", PALETTE["oracle"]),
    "tgt_proxy": ("targeted, proxy", PALETTE["proxy"]),
    "lev": ("leverage", PALETTE["lev"]),
}


def runner(p, meth, m, rng):
    return lc.run_coreset(p, {"unif": "c_unif", "tgt_oracle": "c_tgt_oracle", "tgt_proxy": "c_tgt_proxy", "lev": "c_lev"}[meth], m, rng)


def mix_sweep(p, reps, rng, mus, factors, fracs):
    frames = []
    for m in [k * p.d for k in factors if k * p.d < p.n // 4]:
        for f in fracs:
            p.m_p = max(int(round(f * m)), 1)
            for mu in mus:
                p.mu = mu
                frames.append(cl.experiment(p, m, reps, rng, meths=METH, runner=runner).assign(mu=mu, f=f))
                print(f"  m={m} f={f} mu={mu}: done", flush=True)
    return pd.concat(frames)


def plot(df, path):
    uni = df[df.method == "unif"].groupby(["m", "metric"]).mean_abs_rel.mean().rename("uniform")
    d = df[df.method != "unif"].join(uni, on=["m", "metric"])
    d["ratio"] = d.mean_abs_rel / d.uniform
    ms = sorted(d.m.unique())
    fig, axes = plt.subplots(len(ms), len(MET), figsize=(3.4 * len(MET), 2.9 * len(ms) + 0.6), squeeze=False, sharex=True)
    for i, m in enumerate(ms):
        for j, met in enumerate(MET):
            ax = axes[i, j]
            g = d[(d.m == m) & (d.metric == met)]
            for meth, (lab, col) in STYLE.items():
                h = g[g.method == meth].sort_values("mu")
                ax.plot(h.mu, h.ratio, "o-", color=col, label=lab)
            ax.axhline(1, color="k", lw=0.8)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(met)
            if i == len(ms) - 1:
                ax.set_xlabel(r"mixing weight $\mu$")
            if j == 0:
                ax.set_ylabel(f"$m={m}$")
    style_axes(axes)
    legend_above(fig, axes, 1 - 0.6 / (2.9 * len(ms) + 0.6))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = add_real_class_args(cl.add_args(argparse.ArgumentParser(), "results/mix_s5c", m=300, T=3, synth=True))
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--mus", default=",".join(map(str, MUS)))
    ap.add_argument("--fracs", default="0.25")
    ap.set_defaults(data_seed=3)
    a = ap.parse_args()
    ensure_scienceplots()
    rng = cl.start(a)
    mus = [float(v) for v in a.mus.split(",")]
    fracs = [float(v) for v in a.fracs.split(",")]
    factors = [10, 40] if a.quick else M_FACTORS
    probs = [P(*make_data(a.n, a.d, frac1=0.1, seed=a.data_seed))] if a.synthetic else real_class_problems(a)
    t0 = time.time()
    for p in probs:
        print(f"[{p.name}] n={p.n} d={p.d}", flush=True)
        stem = f"{a.out}/s5c_mix_{'synth' if a.synthetic else p.name}"
        df = mix_sweep(p, a.reps, rng, mus, factors, fracs)
        df.to_csv(f"{stem}.csv", index=False)
        if len(fracs) == 1:
            plot(df, f"{stem}.png")
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()
