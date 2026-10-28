#!/usr/bin/env python3
"""Section 4.2.1, synthetic data: OLS vs {Gaussian, CountSketch, Uniform, Importance} sketches on IF / GFE / MSE parity,
next to the Proposition 1 and Theorem 1 bounds.
  --exp table | vary_m | vary_kappa | vary_gap | vary_ng | vary_noise | all
Outputs: table_synth.{csv,tex}, sweep_*.{csv,png}"""
import argparse, time
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # fs_common.py lives at the repo root
from fs_common import *


def main():
    ap = add_synth_args(add_common_args(argparse.ArgumentParser()))
    ap.add_argument("--exp", default="all")
    args = finish_args(ap.parse_args())
    rng = np.random.default_rng(args.seed)
    exps = set(args.exp.split(","))
    want = lambda e: "all" in exps or e in exps
    synth = synth_builder(args)
    m, reps, d, out = args.m, args.reps, args.d, args.out
    t0 = time.time()
    if want("table"):
        print("[table]")
        prob = synth()
        print(f"   n={prob.n} d={prob.d} cond={prob.cond:.1f} n0={prob.n0} n1={prob.n1} |r|={np.linalg.norm(prob.r):.1f}")
        save_vanilla_table(vanilla_table(prob, m, reps, rng), out, "synth")
    if want("vary_m"):
        print("[vary_m]")
        prob = synth()
        ms = sweep_pts(args, [4 * d, 8 * d, 16 * d, 32 * d, 64 * d, 128 * d, 256 * d], [4 * d, 16 * d, 64 * d, 256 * d])
        run_sweep_vanilla("vary_m", ms, lambda x: prob, m, reps, rng, out, "sketch size m", extra_m=lambda x: x)
    if want("vary_kappa"):
        print("[vary_kappa]")
        ks = sweep_pts(args, [1, 3, 10, 30, 100, 300], [1, 10, 100])
        run_sweep_vanilla("vary_kappa", ks, lambda x: synth(cond=float(x)), m, reps, rng, out, "condition number")
    if want("vary_gap"):
        print("[vary_gap]")
        gs = sweep_pts(args, [0.1, 0.3, 1, 3, 10], [0.1, 1, 10])
        for gd in ("top", "bottom"):
            run_sweep_vanilla(f"vary_gap_{gd}", gs, lambda x, gd=gd: synth(gap=float(x), gap_dir=gd), m, reps, rng, out,
                              f"size of group mean gap (along {gd} singular direction)")
    if want("vary_ng"):
        print("[vary_ng]")
        fs = sweep_pts(args, [0.5, 0.2, 0.1, 0.05, 0.02], [0.5, 0.1, 0.02])
        run_sweep_vanilla("vary_ng", fs, lambda x: synth(frac1=float(x)), m, reps, rng, out, "minority fraction n1/n")
    if want("vary_noise"):
        print("[vary_noise]")
        ss = sweep_pts(args, [0.1, 0.3, 1, 3], [0.1, 1, 3])
        run_sweep_vanilla("vary_noise", ss, lambda x: synth(noise0=float(x), noise1=float(x), mis=0.0), m, reps, rng, out,
                          "noise sd (=> residual size)")
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()