#!/usr/bin/env python3
"""Section 5.3, synthetic data: OLS -> vanilla sketch -> stratified oracle -> stratified proxy (pilot).
  --exp table | m | pilot | dissim | all
Outputs: strat_synth.{csv,tex}, strat_sweep_{m,pilot,dissim}.{csv,png}"""
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
    m, mp, reps, d, out = args.m, args.m_pilot, args.reps, args.d, args.out
    t0 = time.time()
    if want("table"):
        print("[table]")
        prob = synth(noise1=3.0)
        save_strat_table(strat_experiment(prob, m, mp, reps, rng, pilot_fam=args.pilot_fam), out, "synth")
    if want("m"):
        print("[m]")
        prob = synth(noise1=3.0, frac1=0.5)
        ms = sweep_pts(args, [8 * d, 16 * d, 32 * d, 64 * d, 128 * d], [8 * d, 32 * d, 128 * d])
        run_sweep_strat("m", ms, lambda x: prob, lambda x: x, lambda x: max(2 * d, x // 5), reps, rng, out,
                        "total sketch size m (pilot = m/5)", pilot_fam=args.pilot_fam)
    if want("pilot"):
        print("[pilot]")
        prob = synth(noise1=3.0, frac1=0.5)
        ps = sweep_pts(args, [2 * d, 4 * d, 8 * d, 16 * d, 32 * d], [2 * d, 8 * d, 32 * d])
        mbig = max(m, 4 * max(ps))
        run_sweep_strat("pilot", ps, lambda x: prob, lambda x: mbig, lambda x: x, reps, rng, out,
                        f"pilot size (m={mbig})", pilot_fam=args.pilot_fam)
    if want("dissim"):
        print("[dissim]")
        rs = sweep_pts(args, [1, 2, 4, 8, 16], [1, 4, 16])
        run_sweep_strat("dissim", rs, lambda x: synth(noise0=1.0, noise1=float(x), frac1=0.5), lambda x: m, lambda x: mp,
                        reps, rng, out, "noise ratio of group 1 to group 0 (group dissimilarity)", pilot_fam=args.pilot_fam)
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()