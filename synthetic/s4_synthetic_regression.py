#!/usr/bin/env python3
"""Section 4, regression, synthetic data: distortion of IF, GFE and MSE parity by the four sketch families (Gaussian, CountSketch, uniform, importance).
  --exp synth_m | synth_kappa | synth_gap | synth_lev | synth_spike | bounds_table | all
  synth_m       vary sketch size m                         -> size/direction figure
  synth_kappa   vary condition number                      -> size/direction figure
  synth_gap     vary group mean difference (top and bottom singular direction) -> two figures
  synth_lev     vary leverage heterogeneity (all rows; minority only) -> two figures
  synth_spike   coherent design (k high-leverage, high-residual minority rows): vary the spike outlier size, and the number of spikes -> two figures
                + synth_spike_theory.csv (max leverage, the Lemma-2 CountSketch/Gaussian variance ratio, and the empirical CS/Gaussian ratio of mean distortion, per point)
  bounds_table  Prop 1 and Thm 1 bounds vs empirical distortion      -> table
Outputs: synth_*.{csv,png}, table_bounds_synth.{csv,tex}
Usage: python reg_s4_synth.py --out results/reg_s4_synth --reps 200 [--quick] [--exp synth_m,bounds_table]"""
import argparse, time
import numpy as np
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
    m, reps, d, out = args.m, args.reps, args.d, args.out
    t0 = time.time()
    if want("synth_m"):
        print("[synth_m]")
        prob = synth()
        ms = sweep_pts(args, [4 * d, 8 * d, 16 * d, 32 * d, 64 * d, 128 * d, 256 * d], [4 * d, 16 * d, 64 * d, 256 * d])
        run_sweep_sd("synth_m", ms, lambda x: prob, m, reps, rng, out, "sketch size m", extra_m=lambda x: x)
    if want("synth_kappa"):
        print("[synth_kappa]")
        ks = sweep_pts(args, [1, 3, 10, 30, 100, 300], [1, 10, 100])
        run_sweep_sd("synth_kappa", ks, lambda x: synth(cond=float(x)), m, reps, rng, out, "condition number")
    if want("synth_gap"):
        print("[synth_gap]")
        gs = sweep_pts(args, [0.1, 0.3, 1, 3, 10], [0.1, 1, 10])
        for gd in ("top", "bottom"):
            run_sweep_sd(f"synth_gap_{gd}", gs, lambda x, gd=gd: synth(gap=float(x), gap_dir=gd), m, reps, rng, out,
                         "group mean gap")
    if want("synth_lev"):
        print("[synth_lev]")
        ls = sweep_pts(args, [0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0])
        run_sweep_sd("synth_lev", ls, lambda x: synth(lev=float(x)), m, reps, rng, out, "leverage spread σ", logx=False)
        run_sweep_sd("synth_lev_minority", ls, lambda x: synth(lev1=float(x), frac1=0.05), m, reps, rng, out,
                     "minority leverage spread σ", logx=False)
    if want("synth_spike"):
        print("[synth_spike]")
        ns = sweep_pts(args, [1, 3, 10, 30, 100], [1, 10, 100])
        ks = sweep_pts(args, [2, 3, 5, 10, 20, 50], [2, 10, 50])
        mk = {"outlier": lambda x: synth(spikes=2, spike_out=float(x)), "k": lambda x: synth(spikes=int(x), spike_out=100.0)}
        dfs = {"outlier": run_sweep_sd("synth_spike_outlier", ns, mk["outlier"], m, reps, rng, out,
                                       "spike outlier size"),
               "k": run_sweep_sd("synth_spike_k", ks, mk["k"], m, reps, rng, out, "spike rows k")}
        rows = []
        for kind, xs in (("outlier", ns), ("k", ks)):
            df = dfs[kind]
            for x in xs:
                ratio, hmax = cs_gauss_trace_ratio(mk[kind](x))
                row = dict(sweep=kind, x=x, max_leverage=hmax, theory_cs_over_gauss_variance=ratio)
                for met in ("IF", "GFE", "MSE"):
                    d = df[(df.x == x) & (df.metric == met)].set_index("family").mean_abs
                    row[f"empirical_cs_over_gauss_{met}"] = float(d["cs"] / d["gauss"])
                rows.append(row)
        pd.DataFrame(rows).round(4).to_csv(f"{out}/synth_spike_theory.csv", index=False)
    if want("bounds_table"):
        print("[bounds_table]")
        save_vanilla_table(vanilla_table(synth(), m, reps, rng), out, "bounds_synth")
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()