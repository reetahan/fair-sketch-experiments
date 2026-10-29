#!/usr/bin/env python3
"""Section 5, classification, synthetic data: exact model vs vanilla Newton-Hessian sketch (CountSketch, Gaussian) vs oracle vs proxy
(stratified importance sampling, see lib_intervention.py). Distortion is measured after T Newton steps (the iteration converges to the exact model).
Outputs: s5c_synth_table.{csv,tex}; s5c_synth_{m,steps,minority,mis}.{csv,png}
Usage: python class_s5_synth.py --out results/class_s5_synth --reps 60 [--quick] [--sweep m,steps,minority,mis] [--replot]"""
import argparse, time
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # library modules live at the repo root
import lib_classification as cl
from lib_classification import P, make_data
from lib_intervention import S5_METH, run_newton, DEFAULT_T

SPECS = {
    "s5c_synth_m": dict(x="m", label="sketch size m", logx=True, meths=S5_METH),
    "s5c_synth_steps": dict(x="T", label="Newton steps", logx=False, meths=S5_METH),
    "s5c_synth_minority": dict(x="frac1", label="minority fraction", logx=False, meths=S5_METH),
    "s5c_synth_mis": dict(x="mis", label="group model gap", logx=False, meths=S5_METH),
}


def main():
    ap = cl.add_args(argparse.ArgumentParser(), "results/class_s5_synth", m=300, T=DEFAULT_T, synth=True)
    ap.set_defaults(data_seed=3)          # a draw for which all exact-model gaps are clearly nonzero
    ap.add_argument("--sweep", default="m,steps,minority,mis")
    a = ap.parse_args()
    if a.replot:
        cl.set_style(a.latex)
        return cl.replot(a.out, SPECS)
    rng = cl.start(a)
    t0 = time.time()
    mk = lambda **kw: P(*make_data(a.n, a.d, **{"frac1": 0.1, "seed": a.data_seed, **kw}))
    base = mk()
    base.T = a.T
    print("exact-model gaps:", {k: round(float(v), 4) for k, v in base.base.items()}, flush=True)
    cl.table(cl.experiment(base, a.m, a.reps, rng, meths=S5_METH, runner=run_newton), a.out, "s5c_synth_table", S5_METH,
             f"Classification, synthetic, {a.T} Newton steps, m={a.m}: vanilla vs oracle vs proxy.")
    sw = set(a.sweep.split(","))
    q = a.quick
    run = lambda builder, xs, xname, **kw: cl.sweep(builder, xs, xname, a.reps, rng, a.m, S5_METH, runner=run_newton, **kw)
    if "m" in sw:
        ms = [5 * a.d, 20 * a.d, 80 * a.d] if q else [5 * a.d * 2 ** i for i in range(6)]
        cl.emit(a.out, "s5c_synth_m", run(lambda x: base, ms, "m"), SPECS["s5c_synth_m"])
    if "steps" in sw:
        cl.emit(a.out, "s5c_synth_steps", run(lambda x: base, [1, 2, 3] if q else [1, 2, 3, 4, 6], "T"), SPECS["s5c_synth_steps"])
    if "minority" in sw:
        fs = [0.02, 0.1, 0.5] if q else [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
        cl.emit(a.out, "s5c_synth_minority", run(lambda x: mk(frac1=x), fs, "frac1", T=a.T), SPECS["s5c_synth_minority"])
    if "mis" in sw:
        vs = [0.0, 1.0, 3.0] if q else [0.0, 0.5, 1.0, 2.0, 3.0]
        cl.emit(a.out, "s5c_synth_mis", run(lambda x: mk(mis=x), vs, "mis", T=a.T), SPECS["s5c_synth_mis"])
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()