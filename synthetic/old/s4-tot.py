#!/usr/bin/env python3
"""Section 4 experiments, exactly the five requested outputs.

  synth_m       synthetic, fixed data, vary sketch size m          -> size and direction figure
  synth_kappa   synthetic, vary condition number                   -> size and direction figure
  synth_gap     synthetic, vary average group difference           -> size and direction figures (gap along top / bottom direction)
  bounds_table  Prop 1 and Thm 1 bounds vs empirical distortion    -> TABLE (synthetic; also each real dataset)
  real_m        real data (law school, ACS with --acs_state), vary m -> size and direction figure

Size = mean absolute change of the metric relative to its OLS value. Direction = mean signed change (positive means
the sketch made the metric less fair) with a 10th-90th percentile band. Rank-deficient sketches are excluded and reported.
Usage:  python section4.py --out results/s4 --reps 300 [--exp synth_m,bounds_table] [--acs_state CA] (law school and ACS both run by default; --no_law / --no_acs to skip)"""
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
    m, reps, d, out = args.m, args.reps, args.d, args.out
    t0 = time.time()

    if want("synth_m"):
        print("[synth_m]")
        prob = synth()
        ms = sweep_pts(args, [4 * d, 8 * d, 16 * d, 32 * d, 64 * d, 128 * d, 256 * d], [4 * d, 16 * d, 64 * d, 256 * d])
        run_sweep_sd("synth_m", ms, lambda x: prob, m, reps, rng, out, "sketch size m", extra_m=lambda x: x,
                     title=f"synthetic: effect of sketch size (n={prob.n}, d={prob.d}, cond={prob.cond:.0f})")
    if want("synth_kappa"):
        print("[synth_kappa]")
        ks = sweep_pts(args, [1, 3, 10, 30, 100, 300], [1, 10, 100])
        run_sweep_sd("synth_kappa", ks, lambda x: synth(cond=float(x)), m, reps, rng, out, "condition number",
                     title=f"synthetic: effect of condition number (m={m})")
    if want("synth_gap"):
        print("[synth_gap]")
        gs = sweep_pts(args, [0.1, 0.3, 1, 3, 10], [0.1, 1, 10])
        for gd in ("top", "bottom"):
            run_sweep_sd(f"synth_gap_{gd}", gs, lambda x, gd=gd: synth(gap=float(x), gap_dir=gd), m, reps, rng, out,
                         "size of average group difference",
                         title=f"synthetic: effect of group difference along the {gd} singular direction (m={m})")
    if want("synth_lev"):
        print("[synth_lev]")
        ls = sweep_pts(args, [0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0])
        run_sweep_sd("synth_lev", ls, lambda x: synth(lev=float(x)), m, reps, rng, out, "row-scale heterogeneity (lognormal sigma)", logx=False,
                     title=f"synthetic: heavy-tailed leverage, all rows (m={m})")
        run_sweep_sd("synth_lev_minority", ls, lambda x: synth(lev1=float(x), frac1=0.05), m, reps, rng, out,
                     "minority row-scale heterogeneity (lognormal sigma)", logx=False,
                     title=f"synthetic: heavy-tailed leverage in a 5% minority only (m={m})")
    if want("bounds_table"):
        print("[bounds_table]")
        prob = synth()
        save_vanilla_table(vanilla_table(prob, m, reps, rng), out, "bounds_synth")
        for rp in real_problems(args):
            save_vanilla_table(vanilla_table(rp, min(m, rp.n // 4), reps, rng), out, f"bounds_{rp.name}")
    if want("real_m"):
        print("[real_m]")
        for rp in real_problems(args):
            ms = [x for x in [4 * rp.d, 8 * rp.d, 16 * rp.d, 32 * rp.d, 64 * rp.d, 128 * rp.d, 256 * rp.d] if x < rp.n // 2]
            run_sweep_sd(f"real_m_{rp.name}", ms, lambda x, p=rp: p, m, reps, rng, out, "sketch size m", extra_m=lambda x: x,
                         title=f"{rp.name}: effect of sketch size (n={rp.n}, d={rp.d}, cond={rp.cond:.0f})")
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()