#!/usr/bin/env python3
"""Section 5 regression: group-stratified sketches (oracle and proxy), on real data or (--synthetic) synthetic data."""
import argparse, time
import numpy as np
import pandas as pd
from fs_common import *
from lib_data import real_problems, add_real_args


def synthetic(args, rng, want):
    synth = synth_builder(args)
    m, mp, reps, d, out = args.m, args.m_pilot, args.reps, args.d, args.out
    kw = dict(noise1=3.0, frac1=0.5)
    run = lambda name, xs, builder, mfun, mpfun, xlabel, **k: run_sweep_methods(
        f"s5_synth_{name}", xs, builder, mfun, mpfun, reps, rng, out, xlabel, pilot_fam=args.pilot_fam, **k)
    if want("table"):
        save_strat_table(strat_experiment(synth(noise1=3.0), m, mp, reps, rng, pilot_fam=args.pilot_fam), out, "synth")
    if want("m"):
        prob = synth(**kw)
        ms = sweep_pts(args, [8 * d * 2**i for i in range(6)], [16 * d, 64 * d, 256 * d])
        run("m", ms, lambda x: prob, lambda x: x, lambda x: max(2 * d + 2, min(mp, x // 2)), "sketch size m")
    if want("kappa"):
        run("kappa", sweep_pts(args, [1, 3, 10, 30, 100, 300], [1, 10, 100]), lambda x: synth(cond=float(x), **kw),
            lambda x: m, lambda x: mp, "condition number")
    if want("gap"):
        run("gap", sweep_pts(args, [0.1, 0.3, 1, 3, 10], [0.1, 1, 10]), lambda x: synth(gap=float(x), **kw),
            lambda x: m, lambda x: mp, "group mean gap")
    if want("lev"):
        ls = sweep_pts(args, [0.0, 0.5, 1.0, 1.5, 2.0], [0.0, 1.0, 2.0])
        run("lev", ls, lambda x: synth(lev=float(x), **kw), lambda x: m, lambda x: mp, r"leverage spread $\sigma$",
            logx=False)
        run("lev_minority", ls, lambda x: synth(lev1=float(x), noise1=3.0, frac1=0.05), lambda x: m, lambda x: mp,
            r"minority leverage spread $\sigma$", logx=False)
    if want("pilot"):
        prob = synth(**kw)
        ps = sweep_pts(args, [2 * d * 2**i for i in range(5)], [2 * d, 8 * d, 32 * d])
        mbig = max(m, 4 * max(ps))
        run("pilot", ps, lambda x: prob, lambda x: mbig, lambda x: x, "pilot size")
    if want("beta"):
        grid = ([("noise1", v, dict(noise1=float(v), frac1=0.5)) for v in [1, 2, 4, 8]]
                + [("mis", v, dict(noise1=3.0, frac1=0.5, mis=v)) for v in [0.0, 0.5, 1.0, 2.0]]
                + [("frac1", v, dict(noise1=3.0, frac1=v)) for v in [0.05, 0.1, 0.2, 0.35, 0.5]]
                + [("gap", v, dict(noise1=3.0, frac1=0.5, gap=float(v))) for v in [0.1, 1, 10]])
        frames = []
        for var, val, kws in grid:
            bt = beta_table(synth(**kws))
            bt.insert(0, "value", val)
            bt.insert(0, "varied", var)
            frames.append(bt)
        df = pd.concat(frames)
        df.to_csv(f"{out}/s5_beta.csv", index=False)
        write_booktabs(df[df.family == "gauss"], ["varied", "value", "metric", "beta0", "beta1", "omega", "V", "O", "R"],
                       ["Varied", "Value", "Metric", "$\\beta_0$", "$\\beta_1$", "$\\omega$", "$V$", "$O$", "$R=V/O$"],
                       f"{out}/s5_beta.tex", caption="How group coefficients drive the oracle gain (Gaussian sketch).",
                       label="tab:s5_beta")
        plot_beta_gain(df, f"{out}/s5_beta.png")


def real(args, rng, want):
    m, reps, out = args.m, args.reps, args.out
    pilot = lambda d, mm: max(2 * d + 2, min(args.m_pilot, mm // 2))
    probs = real_problems(args)
    for rp in probs:
        mm = min(m, rp.n // 4)
        mp = pilot(rp.d, mm)
        print(f"[{rp.name}] n={rp.n} d={rp.d} cond={rp.cond:.1f} m={mm} m_pilot={mp}", flush=True)
        if want("table"):
            save_strat_table(strat_experiment(rp, mm, mp, reps, rng, pilot_fam=args.pilot_fam), out, f"s5_real_{rp.name}")
        if want("m"):
            ms = [x for x in [8 * rp.d * 2**i for i in range(6)] if x < rp.n // 4]
            run_sweep_methods(f"s5_real_m_{rp.name}", ms, lambda x, p=rp: p, lambda x: x,
                              lambda x, p=rp: pilot(p.d, x), reps, rng, out, "sketch size m", pilot_fam=args.pilot_fam)
    if want("ablate") and probs:
        base = probs[0]
        mm = min(m, base.n // 4)
        mp = pilot(base.d, mm)
        for kind, vals in (("scale_col", [1, 3, 10, 30]), ("subsample0", [1.0, 0.5, 0.2, 0.1]),
                           ("noise1", [0.0, 0.5, 1.0, 2.0])):
            run_sweep_methods(f"s5_real_ablate_{kind}", vals, lambda v, k=kind: alter(base, k, v, args.seed),
                              lambda v: mm, lambda v: mp, reps, rng, out, ALTER_LABEL[kind], logx=(kind == "scale_col"),
                              pilot_fam=args.pilot_fam)


def main():
    args = finish_args(add_real_args(add_common_args(argparse.ArgumentParser())).parse_args())
    exps = set(args.exp.split(","))
    t0 = time.time()
    (synthetic if args.synthetic else real)(args, np.random.default_rng(args.seed), lambda e: "all" in exps or e in exps)
    print(f"done in {time.time() - t0:.0f}s; outputs in {args.out}/")


if __name__ == "__main__":
    main()
