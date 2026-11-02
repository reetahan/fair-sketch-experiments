#!/usr/bin/env python3
"""Section 4 regression on real data."""
import argparse, time
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fs_common import *


def main():
    ap = add_real_args(add_common_args(argparse.ArgumentParser()))
    ap.add_argument("--exp", default="all")
    args = finish_args(ap.parse_args())
    rng = np.random.default_rng(args.seed)
    exps = set(args.exp.split(","))
    want = lambda e: "all" in exps or e in exps
    m, reps, out = args.m, args.reps, args.out
    t0 = time.time()
    probs = real_problems(args)
    for rp in probs:
        mm = min(m, rp.n // 4)
        print(f"[{rp.name}] n={rp.n} d={rp.d} cond={rp.cond:.1f} n0={rp.n0} n1={rp.n1}", flush=True)
        if want("bounds_table"):
            save_vanilla_table(vanilla_table(rp, mm, reps, rng), out, f"bounds_{rp.name}")
        if want("real_m"):
            ms = [
                x
                for x in [4 * rp.d, 8 * rp.d, 16 * rp.d, 32 * rp.d, 64 * rp.d, 128 * rp.d, 256 * rp.d]
                if x < rp.n // 2
            ]
            run_sweep_sd(
                f"real_m_{rp.name}", ms, lambda x, p=rp: p, mm, reps, rng, out, "sketch size m", extra_m=lambda x: x
            )
    if want("ablate") and probs:
        base = probs[0]
        mm = min(m, base.n // 4)
        for kind, vals in (
            ("scale_col", [1, 3, 10, 30]),
            ("subsample0", [1.0, 0.5, 0.2, 0.1]),
            ("noise1", [0.0, 0.5, 1.0, 2.0]),
        ):
            frames = []
            for v in vals:
                dfv = vanilla_table(alter(base, kind, v, args.seed), mm, reps, rng)
                dfv["kind"], dfv["x"] = kind, v
                frames.append(dfv)
                print(f"  ablate {kind}={v}: done", flush=True)
            dd = pd.concat(frames)
            dd.to_csv(f"{out}/ablate_vanilla_{kind}.csv", index=False)
            plot_vanilla_sweep(
                dd, "x", ALTER_LABEL[kind], f"{out}/ablate_vanilla_{kind}.png", logx=(kind == "scale_col")
            )
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()
