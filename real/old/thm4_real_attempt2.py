#!/usr/bin/env python3
"""Section 4.2.2, real data (law school; ACSIncome with --acs_state CA): table of OLS vs sketches vs bounds,
distortion vs m, and optional synthetic ablations of the law-school data.
  --exp table | vary_m | ablate | all
Outputs: table_<dataset>.{csv,tex}, sweep_vary_m_<dataset>.{csv,png}, ablate_vanilla_*.csv"""
import argparse, time
import numpy as np
import pandas as pd
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))   # fs_common.py lives at the repo root
from fs_common import *


def main():
    ap = add_real_args(add_common_args(argparse.ArgumentParser()))
    ap.add_argument("--exp", default="all")
    args = finish_args(ap.parse_args())
    rng = np.random.default_rng(args.seed)
    exps = set(args.exp.split(","))
    want = lambda e: "all" in exps or e in exps
    out, reps = args.out, args.reps
    t0 = time.time()
    probs = real_problems(args)
    for prob in probs:
        tag = prob.name
        mm = min(args.m, prob.n // 4)
        print(f"[{tag}] n={prob.n} d={prob.d} cond={prob.cond:.1f} n0={prob.n0} n1={prob.n1}")
        if want("table"):
            save_vanilla_table(vanilla_table(prob, mm, reps, rng), out, tag)
        if want("vary_m"):
            ms = [x for x in [4 * prob.d, 16 * prob.d, 64 * prob.d, 256 * prob.d] if x < prob.n // 2]
            run_sweep_vanilla(f"vary_m_{tag}", ms, lambda x, p=prob: p, mm, reps, rng, out, "sketch size m", extra_m=lambda x: x)
    if want("ablate") and probs:
        base = probs[0]
        mm = min(args.m, base.n // 4)
        for kind, vals in (("scale_col", [1, 3, 10, 30]), ("subsample0", [1.0, 0.5, 0.2, 0.1]), ("noise1", [0.0, 0.5, 1.0, 2.0])):
            frames = []
            for v in vals:
                dfv = vanilla_table(alter(base, kind, v, args.seed), mm, reps, rng)
                dfv["kind"], dfv["x"] = kind, v
                frames.append(dfv)
                print(f"  ablate {kind}={v}: done", flush=True)
            dd = pd.concat(frames)
            dd.to_csv(f"{out}/ablate_vanilla_{kind}.csv", index=False)
            plot_vanilla_sweep(dd, "x", kind, f"{out}/ablate_vanilla_{kind}.png", logx=(kind == "scale_col"), title=f"ablation: {kind}")
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()