#!/usr/bin/env python3
"""Section 5, regression, synthetic data: unsketched (OLS) vs vanilla sketch vs stratified oracle vs stratified proxy.
  --exp table | m | kappa | gap | lev | pilot | beta | all
  table  one setting, all metrics and families            -> strat_synth.{csv,tex}
  m, kappa, gap, lev, pilot  sweeps                        -> figure per sketch family (OLS is the zero line)
  beta   how the group coefficients beta_g (noise ratio, group model difference, minority share, group gap) drive the oracle gain R = V/O
Outputs: strat_synth.*, s5_synth_<sweep>_<family>.png, s5_beta.{csv,tex,png}
Usage: python reg_s5_synth.py --out results/reg_s5_synth --reps 200 [--quick] [--exp m,beta]"""
import argparse, time
import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # library modules live at the repo root
from fs_common import *


def main():
    ap = add_synth_args(add_common_args(argparse.ArgumentParser()))
    ap.add_argument("--exp", default="all")
    args = finish_args(ap.parse_args())
    rng = np.random.default_rng(args.seed)
    exps = set(args.exp.split(","))
    want = lambda e: "all" in exps or e in exps
    synth = synth_builder(args)
    m, mp, reps, d, out = args.m, args.m_pilot, args.reps, args.d, args.out
    kw = dict(noise1=3.0, frac1=0.5)       # heteroskedastic, balanced groups, so that stratification matters
    run = lambda name, xs, builder, mfun, mpfun, xlabel, **k: run_sweep_methods(f"s5_synth_{name}", xs, builder, mfun, mpfun, reps, rng, out, xlabel,
                                                                                  pilot_fam=args.pilot_fam, **k)
    t0 = time.time()
    if want("table"):
        print("[table]")
        save_strat_table(strat_experiment(synth(noise1=3.0), m, mp, reps, rng, pilot_fam=args.pilot_fam), out, "synth")
    if want("m"):
        print("[m]")
        prob = synth(**kw)
        ms = sweep_pts(args, [8 * d, 16 * d, 32 * d, 64 * d, 128 * d, 256 * d], [16 * d, 64 * d, 256 * d])
        run("m", ms, lambda x: prob, lambda x: x, lambda x: max(2 * d + 2, min(mp, x // 2)), "sketch size m")
    if want("kappa"):
        print("[kappa]")
        ks = sweep_pts(args, [1, 3, 10, 30, 100, 300], [1, 10, 100])
        run("kappa", ks, lambda x: synth(cond=float(x), **kw), lambda x: m, lambda x: mp, "condition number")
    if want("gap"):
        print("[gap]")
        gs = sweep_pts(args, [0.1, 0.3, 1, 3, 10], [0.1, 1, 10])
        run("gap", gs, lambda x: synth(gap=float(x), **kw), lambda x: m, lambda x: mp, "group mean gap")
    if want("lev"):
        print("[lev]")
        ls = sweep_pts(args, [0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0])
        run("lev", ls, lambda x: synth(lev=float(x), **kw), lambda x: m, lambda x: mp, r"leverage spread $\sigma$", logx=False)
        run("lev_minority", ls, lambda x: synth(lev1=float(x), noise1=3.0, frac1=0.05), lambda x: m, lambda x: mp,
            r"minority leverage spread $\sigma$", logx=False)
    if want("pilot"):
        print("[pilot]")
        prob = synth(**kw)
        ps = sweep_pts(args, [2 * d, 4 * d, 8 * d, 16 * d, 32 * d], [2 * d, 8 * d, 32 * d])
        mbig = max(m, 4 * max(ps))
        run("pilot", ps, lambda x: prob, lambda x: mbig, lambda x: x, "pilot size")
    if want("beta"):
        print("[beta]")
        grid = ([("noise1", n1, dict(noise1=float(n1), frac1=0.5)) for n1 in [1, 2, 4, 8]]
                + [("mis", v, dict(noise1=3.0, frac1=0.5, mis=v)) for v in [0.0, 0.5, 1.0, 2.0]]
                + [("frac1", f1, dict(noise1=3.0, frac1=f1)) for f1 in [0.05, 0.1, 0.2, 0.35, 0.5]]
                + [("gap", gp, dict(noise1=3.0, frac1=0.5, gap=float(gp))) for gp in [0.1, 1, 10]])
        rows = []
        for var, val, kws in grid:
            bt = beta_table(synth(**kws))
            bt.insert(0, "value", val)
            bt.insert(0, "varied", var)
            rows.append(bt)
        df = pd.concat(rows)
        df.to_csv(f"{out}/s5_beta.csv", index=False)
        write_booktabs(df[df.family == "gauss"], ["varied", "value", "metric", "beta0", "beta1", "omega", "V", "O", "R"],
                       ["Varied", "Value", "Metric", "$\\beta_0$", "$\\beta_1$", "$\\omega$", "$V$", "$O$", "$R=V/O$"],
                       f"{out}/s5_beta.tex", caption="How group coefficients drive the oracle gain (Gaussian sketch).", label="tab:s5_beta")
        plot_beta_gain(df, f"{out}/s5_beta.png")
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()