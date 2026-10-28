#!/usr/bin/env python3
"""Theory-free demonstration of harm, real data (law school; ACSIncome with --acs_state CA).
  --exp table | vary_m | scatter | all
Outputs: harm_<dataset>.{csv,tex}, harm_sweep_vary_m_<dataset>.{csv,png}, harm_scatter_<dataset>.png"""
import argparse, time
import numpy as np
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
    for prob in real_problems(args):
        tag = prob.name
        mm = min(args.m, prob.n // 4)
        print(f"[{tag}] n={prob.n} d={prob.d} n0={prob.n0} n1={prob.n1} m={mm}")
        if want("table"):
            save_harm_table(harm_table(prob, mm, reps, rng), out, tag)
        if want("scatter"):
            harm_scatter(prob, mm, reps, rng, f"{out}/harm_scatter_{tag}.png")
        if want("vary_m"):
            ms = [x for x in [4 * prob.d, 16 * prob.d, 64 * prob.d, 256 * prob.d] if x < prob.n // 2]
            run_sweep_harm(f"vary_m_{tag}", ms, lambda x, p=prob: p, mm, reps, rng, out, "sketch size m", extra_m=lambda x: x)
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()