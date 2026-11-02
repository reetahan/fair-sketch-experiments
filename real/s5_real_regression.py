#!/usr/bin/env python3
"""Section 5 regression on real data."""
import argparse, time
import numpy as np
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
        mp = max(2 * rp.d + 2, min(args.m_pilot, mm // 2))
        print(f"[{rp.name}] n={rp.n} d={rp.d} cond={rp.cond:.1f} m={mm} m_pilot={mp}", flush=True)
        if want("table"):
            save_strat_table(
                strat_experiment(rp, mm, mp, reps, rng, pilot_fam=args.pilot_fam), out, f"s5_real_{rp.name}"
            )
        if want("m"):
            ms = [x for x in [8 * rp.d, 16 * rp.d, 32 * rp.d, 64 * rp.d, 128 * rp.d, 256 * rp.d] if x < rp.n // 4]
            run_sweep_methods(
                f"s5_real_m_{rp.name}",
                ms,
                lambda x, p=rp: p,
                lambda x: x,
                lambda x, p=rp: max(2 * p.d + 2, min(args.m_pilot, x // 2)),
                reps,
                rng,
                out,
                "sketch size m",
                pilot_fam=args.pilot_fam,
            )
    if want("ablate") and probs:
        base = probs[0]
        mm = min(m, base.n // 4)
        mp = max(2 * base.d + 2, min(args.m_pilot, mm // 2))
        for kind, vals in (
            ("scale_col", [1, 3, 10, 30]),
            ("subsample0", [1.0, 0.5, 0.2, 0.1]),
            ("noise1", [0.0, 0.5, 1.0, 2.0]),
        ):
            run_sweep_methods(
                f"s5_real_ablate_{kind}",
                vals,
                lambda v, k=kind: alter(base, k, v, args.seed),
                lambda v: mm,
                lambda v: mp,
                reps,
                rng,
                out,
                ALTER_LABEL[kind],
                logx=(kind == "scale_col"),
                pilot_fam=args.pilot_fam,
            )
    print(f"done in {time.time() - t0:.0f}s; outputs in {out}/")


if __name__ == "__main__":
    main()
