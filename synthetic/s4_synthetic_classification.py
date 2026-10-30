#!/usr/bin/env python3
"""Section 4, classification, synthetic data: exact model vs approach 1 (sampled: uniform, importance), approach 2 (Newton-Hessian:
CountSketch, Gaussian) and approach 3 (LogReg-sketch; naive Gaussian/CountSketch in the table only), on IF, DP, TPR gap, FPR gap.
Outputs: s4c_synth_table.{csv,tex}; s4c_synth_{m,gap,minority,steps}.{csv,png}
Usage: python class_s4_synth.py --out results/class_s4_synth --reps 60 [--quick] [--sweep m,gap,minority,steps] [--replot]"""
import argparse, time
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # library modules live at the repo root
import lib_classification as cl
from lib_classification import P, make_data, METH, PLOT_METH, NEWTON

SPECS = {
    "s4c_synth_m": dict(x="m", label="sketch size m", logx=True, meths=PLOT_METH),
    "s4c_synth_gap": dict(x="gap", label="group shift", logx=False, meths=PLOT_METH),
    "s4c_synth_minority": dict(x="frac1", label="minority fraction", logx=False, meths=PLOT_METH),
    "s4c_synth_steps": dict(x="T", label="Newton steps", logx=False, meths=NEWTON),
}


def main():
    ap = cl.add_args(argparse.ArgumentParser(), "results/class_s4_synth", m=500, T=10, synth=True)
    ap.add_argument("--sweep", default="m,gap,minority,steps")
    a = ap.parse_args()
    if a.replot:
        cl.set_style(a.latex)
        return cl.replot(a.out, SPECS)
    rng = cl.start(a)
    t0 = time.time()
    mk = lambda **kw: P(*make_data(a.n, a.d, seed=a.data_seed, **kw))
    p = mk()
    p.T = a.T
    print("exact-model gaps:", {k: round(float(v), 4) for k, v in p.base.items()}, flush=True)
    cl.table(cl.experiment(p, a.m, a.reps, rng), a.out, "s4c_synth_table", METH,
             f"Classification, synthetic (m={a.m}): distortion relative to the exact logistic model; positive = less fair.")
    sw = set(a.sweep.split(","))
    q = a.quick
    if "m" in sw:
        ms = [5 * a.d, 20 * a.d, 80 * a.d, 320 * a.d] if q else [5 * a.d * 2 ** i for i in range(7)]     # m >= 5d
        cl.emit(a.out, "s4c_synth_m", cl.sweep(lambda x: p, ms, "m", a.reps, rng, a.m, PLOT_METH), SPECS["s4c_synth_m"])
    if "gap" in sw:
        gs = [0.5, 2.0, 3.5] if q else [0.5, 1.0, 2.0, 3.0, 4.0]
        cl.emit(a.out, "s4c_synth_gap", cl.sweep(lambda x: mk(gap=x), gs, "gap", a.reps, rng, a.m, PLOT_METH, T=a.T), SPECS["s4c_synth_gap"])
    if "minority" in sw:
        fs = [0.05, 0.2, 0.5] if q else [0.02, 0.05, 0.1, 0.2, 0.35, 0.5]
        cl.emit(a.out, "s4c_synth_minority", cl.sweep(lambda x: mk(frac1=x), fs, "frac1", a.reps, rng, a.m, PLOT_METH, T=a.T), SPECS["s4c_synth_minority"])
    if "steps" in sw:
        Ts = [1, 2, 4] if q else [1, 2, 3, 4, 6, 10]
        cl.emit(a.out, "s4c_synth_steps", cl.sweep(lambda x: p, Ts, "T", a.reps, rng, a.m, NEWTON, T=None), SPECS["s4c_synth_steps"])
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()