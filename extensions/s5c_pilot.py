#!/usr/bin/env python3
"""Section 5 classification: the targeted proxy against its pilot and mixing settings.

--sweep frac  pilot fraction f = m_p / m                                  (results/pilot_s5c)
--sweep d     pilot size in multiples of d at a fixed budget; --combine draws the figure   (results/pilot_d_s5c)
--sweep mu    uniform mixing weight mu, with leverage as a reference        (results/mix_s5c)
"""
import glob, os, sys, time
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib_classification as cl
from lib_classification import MET, run_coreset
from fs_common import style_axes, legend_above, set_style, ensure_scienceplots, PALETTE

OUT = {"frac": "results/pilot_s5c", "d": "results/pilot_d_s5c", "mu": "results/mix_s5c"}
STEM = {"frac": "s5c_pilot", "d": "s5c_pilot_d", "mu": "s5c_mix"}
M_FACTORS = [10, 40, 160]
PILOTS = [1, 2.5, 5, 10, 20, 40, 80]
FRAC_STYLE = {"c_tgt_oracle": ("targeted, oracle", PALETTE["oracle"]), "c_tgt_proxy": ("targeted, proxy", PALETTE["proxy"]),
              "c_strat_tgt_proxy": ("stratified targeted, proxy", PALETTE["strat_proxy"])}
MU_STYLE = {"tgt_oracle": ("targeted, oracle", PALETTE["oracle"]), "tgt_proxy": ("targeted, proxy", PALETTE["proxy"]),
            "lev": ("leverage", PALETTE["lev"])}
D_STYLE = {"c_lev": ("Leverage", PALETTE["lev"], "--"), "c_tgt_proxy": ("Targeted, proxy", PALETTE["proxy"], "-"),
           "c_strat_tgt_proxy": ("Stratified targeted, proxy", PALETTE["strat_proxy"], "-")}
NAMES = {"law": "LSAC", "acs": "ACS Income", "hmda": "HMDA", "diab": "130-Hospitals"}
MU_METH = {"unif": "c_unif", "tgt_oracle": "c_tgt_oracle", "tgt_proxy": "c_tgt_proxy", "lev": "c_lev"}


def sweep(p, a, rng):
    frames, run = [], lambda m, meths, runner=run_coreset: cl.experiment(p, m, a.reps, rng, meths=meths, runner=runner)
    if a.sweep == "d":
        for m in [k * p.d for k in (40, 160 if 160 * p.d < p.n // 4 else 80)]:
            p.m_p = None
            frames.append(run(m, ["c_unif", "c_lev"]).assign(budget=m / p.d, pilot=np.nan, d=p.d))
            for k in [k for k in PILOTS if k * p.d <= m / 2]:
                p.m_p = int(round(k * p.d))
                frames.append(run(m, ["c_tgt_proxy", "c_strat_tgt_proxy"]).assign(budget=m / p.d, pilot=k, d=p.d))
                print(f"  m={m} pilot={k}d: done", flush=True)
        p.m_p = None
        return pd.concat(frames)
    for m in [k * p.d for k in ([10, 40] if a.quick else M_FACTORS) if k * p.d < p.n // 4]:
        for f in a.fracs:
            p.m_p = max(int(round(f * m)), 1)
            if a.sweep == "frac":
                frames.append(run(m, ["c_unif", *FRAC_STYLE]).assign(f=f, m_p=p.m_p))
                print(f"  m={m} f={f}: done", flush=True)
                continue
            for mu in a.mus:
                p.mu = mu
                frames.append(run(m, list(MU_METH), lambda p, me, m, rng: run_coreset(p, MU_METH[me], m, rng))
                              .assign(mu=mu, f=f))
                print(f"  m={m} f={f} mu={mu}: done", flush=True)
    if a.sweep == "frac":
        p.m_p = None
    return pd.concat(frames)


def plot(df, path, x, xlabel, style, uni):
    u = df[df.method == uni].groupby(["m", "metric"]).mean_abs_rel.mean().rename("uniform")
    d = df[df.method != uni].join(u, on=["m", "metric"])
    d["ratio"] = d.mean_abs_rel / d.uniform
    ms = sorted(d.m.unique())
    fig, axes = plt.subplots(len(ms), len(MET), figsize=(3.4 * len(MET), 2.9 * len(ms) + 0.6), squeeze=False, sharex=True)
    for i, m in enumerate(ms):
        for j, met in enumerate(MET):
            ax, g = axes[i, j], d[(d.m == m) & (d.metric == met)]
            for meth, (lab, col) in style.items():
                h = g[g.method == meth].sort_values(x)
                ax.plot(h[x], h.ratio, "o-", color=col, label=lab)
            ax.axhline(1, color="k", lw=0.8)
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(met)
            if i == len(ms) - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(f"$m={m}$")
    style_axes(axes)
    legend_above(fig, axes, 1 - 0.6 / (2.9 * len(ms) + 0.6))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def combine(out, min_base=0.01):
    frames = {}
    for f in glob.glob(f"{out}/s5c_pilot_d_*.csv"):
        df = pd.read_csv(f)
        ok = df.groupby("metric").base.first().abs()
        df = df[df.metric.isin(ok[ok >= min_base].index)]
        df = df.join(df[df.method == "c_unif"].set_index(["budget", "metric"]).mean_abs_rel.rename("u"), on=["budget", "metric"])
        frames[os.path.basename(f)[len("s5c_pilot_d_"):].split("-")[0].replace(".csv", "")] = \
            df.assign(ratio=df.mean_abs_rel / df.u)
    names = [k for k in NAMES if k in frames]
    fig, axes = plt.subplots(2, len(names), figsize=(3.0 * len(names), 5.6), squeeze=False, sharex=True)
    gm = lambda v: float(np.exp(np.log(v).mean()))
    for j, name in enumerate(names):
        d = frames[name]
        for i, b in enumerate(sorted(d.budget.unique())):
            ax, g = axes[i, j], d[d.budget == b]
            for me, (lab, col, ls) in D_STYLE.items():
                h = g[g.method == me]
                if me == "c_lev":
                    ax.axhline(gm(h.ratio), color=col, ls=ls, lw=1.2, label=lab)
                else:
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
    ap = cl.ext_args(None)
    ap.add_argument("--sweep", required=True, choices=list(OUT))
    ap.add_argument("--combine", action="store_true", help="--sweep d: draw the figure from the CSVs in --out")
    ap.add_argument("--fracs", default=None, help="pilot fractions (default 0.05,...,0.75 for frac, 0.25 for mu)")
    ap.add_argument("--mus", default="0.2,0.5,0.8")
    a = ap.parse_args()
    a.out = a.out or OUT[a.sweep]
    a.mus = [float(v) for v in a.mus.split(",")]
    a.fracs = [float(v) for v in a.fracs.split(",")] if a.fracs else \
        ([0.25] if a.sweep == "mu" else [0.1, 0.5] if a.quick else [0.05, 0.1, 0.25, 0.5, 0.75])
    ensure_scienceplots()
    if a.combine:
        set_style(fontsize=11)
        return combine(a.out)
    rng = cl.start(a)
    probs = cl.ext_problems(a)
    t0 = time.time()
    for p in probs:
        print(f"[{p.name}] n={p.n} d={p.d}", flush=True)
        stem = f"{a.out}/{STEM[a.sweep]}_{'synth' if a.synthetic else p.name}"
        df = sweep(p, a, rng)
        df.to_csv(f"{stem}.csv", index=False)
        if a.sweep == "frac":
            plot(df, f"{stem}.png", "f", "pilot fraction $f$", FRAC_STYLE, "c_unif")
        elif a.sweep == "mu" and len(a.fracs) == 1:
            plot(df, f"{stem}.png", "mu", r"mixing weight $\mu$", MU_STYLE, "unif")
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()
