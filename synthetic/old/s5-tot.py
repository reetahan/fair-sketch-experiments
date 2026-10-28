#!/usr/bin/env python3
"""Section 5 experiments: unsketched (OLS) vs sketched (vanilla) vs oracle-stratified vs proxy-stratified.

  synth_m      synthetic, vary m                       -> figure per sketch family (size row, signed-direction row)
  synth_kappa  synthetic, vary condition number        -> figure per sketch family
  synth_gap    synthetic, vary group difference        -> figure per sketch family
  synth_lev    synthetic, vary leverage heterogeneity (all rows; minority only) -> figure per sketch family
  real         real data (law school, ACS with --acs_state): table of all four methods
  beta         synthetic, how beta_g (noise ratio, misspecification, minority share, group gap) drive oracle vs vanilla
               -> TABLE (analytic beta_g, omega=sqrt(beta1/beta0), V, O, R=V/O) and scatter of R against omega
The OLS line is the zero line: all changes are relative to the unsketched solution; positive = less fair.
Usage: python section5.py --out results/s5 --reps 200 [--exp synth_m,real] [--acs_state CA] (law school and ACS both run by default; --no_law / --no_acs to skip)"""
import argparse, time
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # fs_common.py lives at the repo root
from fs_common import *


def main():
    ap = add_real_args(add_synth_args(add_common_args(argparse.ArgumentParser())))
    ap.add_argument("--exp", default="all")
    args = finish_args(ap.parse_args())
    rng = np.random.default_rng(args.seed)
    exps = set(args.exp.split(","))
    want = lambda e: "all" in exps or e in exps
    synth = synth_builder(args)
    m, mp, reps, d, out = args.m, args.m_pilot, args.reps, args.d, args.out
    kw = dict(noise1=3.0, frac1=0.5)       # heteroskedastic groups so that stratification matters
    t0 = time.time()

    if want("synth_m"):
        print("[synth_m]")
        prob = synth(**kw)
        ms = sweep_pts(args, [8 * d, 16 * d, 32 * d, 64 * d, 128 * d, 256 * d], [16 * d, 64 * d, 256 * d])
        run_sweep_methods("s5_synth_m", ms, lambda x: prob, lambda x: x, lambda x: max(2 * d + 2, min(mp, x // 2)), reps, rng, out,
                          "sketch size m", pilot_fam=args.pilot_fam, title=f"synthetic: vary m (cond={prob.cond:.0f})")
    if want("synth_kappa"):
        print("[synth_kappa]")
        ks = sweep_pts(args, [1, 3, 10, 30, 100, 300], [1, 10, 100])
        run_sweep_methods("s5_synth_kappa", ks, lambda x: synth(cond=float(x), **kw), lambda x: m, lambda x: mp, reps, rng, out,
                          "condition number", pilot_fam=args.pilot_fam, title=f"synthetic: vary condition number (m={m})")
    if want("synth_gap"):
        print("[synth_gap]")
        gs = sweep_pts(args, [0.1, 0.3, 1, 3, 10], [0.1, 1, 10])
        run_sweep_methods("s5_synth_gap", gs, lambda x: synth(gap=float(x), **kw), lambda x: m, lambda x: mp, reps, rng, out,
                          "size of average group difference", pilot_fam=args.pilot_fam, title=f"synthetic: vary group difference (m={m})")
    if want("synth_lev"):
        print("[synth_lev]")
        ls = sweep_pts(args, [0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0])
        run_sweep_methods("s5_synth_lev", ls, lambda x: synth(lev=float(x), **kw), lambda x: m, lambda x: mp, reps, rng, out,
                          "row-scale heterogeneity (lognormal sigma)", logx=False, pilot_fam=args.pilot_fam,
                          title=f"synthetic: heavy-tailed leverage, all rows (m={m})")
        run_sweep_methods("s5_synth_lev_minority", ls, lambda x: synth(lev1=float(x), noise1=3.0, frac1=0.05), lambda x: m, lambda x: mp,
                          reps, rng, out, "minority row-scale heterogeneity (lognormal sigma)", logx=False, pilot_fam=args.pilot_fam,
                          title=f"synthetic: heavy-tailed leverage in a 5% minority (m={m})")
    if want("real"):
        print("[real]")
        for rp in real_problems(args):
            mm = min(m, rp.n // 4)
            df = strat_experiment(rp, mm, min(mp, mm // 2), reps, rng, pilot_fam=args.pilot_fam)
            save_strat_table(df, out, f"s5_real_{rp.name}")
    if want("beta"):
        print("[beta]")
        rows = []
        grid = []
        for n1 in [1, 2, 4, 8]:
            grid.append(("noise1", n1, dict(noise1=float(n1), frac1=0.5)))
        for mis in [0.0, 0.5, 1.0, 2.0]:
            grid.append(("mis", mis, dict(noise1=3.0, frac1=0.5, mis=mis)))
        for f1 in [0.05, 0.1, 0.2, 0.35, 0.5]:
            grid.append(("frac1", f1, dict(noise1=3.0, frac1=f1)))
        for gp in [0.1, 1, 10]:
            grid.append(("gap", gp, dict(noise1=3.0, frac1=0.5, gap=float(gp))))
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
        plot_beta_gain(df, f"{out}/s5_beta.png", title="oracle gain vs group asymmetry (synthetic grid)")
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()