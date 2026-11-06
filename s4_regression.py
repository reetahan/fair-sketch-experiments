#!/usr/bin/env python3
"""Section 4 regression: distortion of plain sketches, on real data or (--synthetic) synthetic data."""
import argparse, time
import numpy as np
import pandas as pd
from fs_common import *
from lib_data import real_problems, add_real_args


def synthetic(args, rng, want):
    synth = synth_builder(args)
    m, reps, d, out = args.m, args.reps, args.d, args.out
    run = lambda name, xs, builder, xlabel, **k: run_sweep_sd(name, xs, builder, m, reps, rng, out, xlabel, **k)
    if want("synth_m"):
        prob = synth()
        ms = sweep_pts(args, [4 * d * 2**i for i in range(7)], [4 * d, 16 * d, 64 * d, 256 * d])
        run("synth_m", ms, lambda x: prob, "sketch size m", extra_m=lambda x: x)
    if want("synth_kappa"):
        run("synth_kappa", sweep_pts(args, [1, 3, 10, 30, 100, 300], [1, 10, 100]), lambda x: synth(cond=float(x)),
            "condition number")
    if want("synth_gap"):
        gs = sweep_pts(args, [0.1, 0.3, 1, 3, 10], [0.1, 1, 10])
        for gd in ("top", "bottom"):
            run(f"synth_gap_{gd}", gs, lambda x, gd=gd: synth(gap=float(x), gap_dir=gd), "group mean gap")
    if want("synth_lev"):
        ls = sweep_pts(args, [0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0])
        run("synth_lev", ls, lambda x: synth(lev=float(x)), r"leverage spread $\sigma$", logx=False)
        run("synth_lev_minority", ls, lambda x: synth(lev1=float(x), frac1=0.05), r"minority leverage spread $\sigma$",
            logx=False)
    if want("synth_spike"):
        mk = {"outlier": lambda x: synth(spikes=2, spike_out=float(x)), "k": lambda x: synth(spikes=int(x), spike_out=100.0)}
        xs = {"outlier": sweep_pts(args, [1, 3, 10, 30, 100], [1, 10, 100]),
              "k": sweep_pts(args, [2, 3, 5, 10, 20, 50], [2, 10, 50])}
        dfs = {"outlier": run("synth_spike_outlier", xs["outlier"], mk["outlier"], "spike outlier size"),
               "k": run("synth_spike_k", xs["k"], mk["k"], "spike rows k")}
        rows = []
        for kind in ("outlier", "k"):
            for x in xs[kind]:
                ratio, hmax = cs_gauss_trace_ratio(mk[kind](x))
                row = dict(sweep=kind, x=x, max_leverage=hmax, theory_cs_over_gauss_variance=ratio)
                for met in METRICS:
                    v = dfs[kind][(dfs[kind].x == x) & (dfs[kind].metric == met)].set_index("family").mean_abs
                    row[f"empirical_cs_over_gauss_{met}"] = float(v["cs"] / v["gauss"])
                rows.append(row)
        pd.DataFrame(rows).round(4).to_csv(f"{out}/synth_spike_theory.csv", index=False)
    if want("bounds_table"):
        save_vanilla_table(vanilla_table(synth(), m, reps, rng), out, "bounds_synth")


def real(args, rng, want):
    m, reps, out = args.m, args.reps, args.out
    probs = real_problems(args)
    for rp in probs:
        mm = min(m, rp.n // 4)
        print(f"[{rp.name}] n={rp.n} d={rp.d} cond={rp.cond:.1f} n0={rp.n0} n1={rp.n1}", flush=True)
        if want("bounds_table"):
            save_vanilla_table(vanilla_table(rp, mm, reps, rng), out, f"bounds_{rp.name}")
        if want("real_m"):
            ms = [x for x in [4 * rp.d * 2**i for i in range(7)] if x < rp.n // 2]
            run_sweep_sd(f"real_m_{rp.name}", ms, lambda x, p=rp: p, mm, reps, rng, out, "sketch size m",
                         extra_m=lambda x: x)
    if want("ablate") and probs:
        base = probs[0]
        mm = min(m, base.n // 4)
        for kind, vals in (("scale_col", [1, 3, 10, 30]), ("subsample0", [1.0, 0.5, 0.2, 0.1]),
                           ("noise1", [0.0, 0.5, 1.0, 2.0])):
            frames = []
            for v in vals:
                frames.append(vanilla_table(alter(base, kind, v, args.seed), mm, reps, rng).assign(kind=kind, x=v))
                print(f"  ablate {kind}={v}: done", flush=True)
            dd = pd.concat(frames)
            dd.to_csv(f"{out}/ablate_vanilla_{kind}.csv", index=False)
            plot_vanilla_sweep(dd, "x", ALTER_LABEL[kind], f"{out}/ablate_vanilla_{kind}.png", logx=(kind == "scale_col"))


def main():
    args = finish_args(add_real_args(add_common_args(argparse.ArgumentParser())).parse_args())
    exps = set(args.exp.split(","))
    t0 = time.time()
    (synthetic if args.synthetic else real)(args, np.random.default_rng(args.seed), lambda e: "all" in exps or e in exps)
    print(f"done in {time.time() - t0:.0f}s; outputs in {args.out}/")


if __name__ == "__main__":
    main()
