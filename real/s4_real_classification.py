#!/usr/bin/env python3
"""Section 4, classification, real data: exact model vs approaches 1-3 on LSAC (pass_bar), ACS California 2018 (income > 50k) AND HMDA California 2018 (loan approval), all three by default.
Outputs per dataset: s4c_<name>_table.{csv,tex}; s4c_<name>_{m,steps}.{csv,png}
Usage: python class_s4_real.py --out results/class_s4_real --reps 60 [--quick] [--no_law] [--no_acs] [--no_hmda] [--acs_state CA] [--replot]"""
import argparse, time
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # library modules live at the repo root
import lib_classification as cl
from lib_classification import METH, PLOT_METH, NEWTON
from lib_data import real_class_problems, add_real_class_args


def specs(name):
    return {f"s4c_{name}_m": dict(x="m", label="sketch size m", logx=True, meths=PLOT_METH),
            f"s4c_{name}_steps": dict(x="T", label="Newton steps", logx=False, meths=NEWTON)}


def main():
    ap = add_real_class_args(cl.add_args(argparse.ArgumentParser(), "results/class_s4_real", m=500, T=10))
    ap.add_argument("--min_base", type=float, default=0.01, help="omit rate-gap points whose exact gap is below this")
    a = ap.parse_args()
    if a.replot:
        cl.set_style(a.latex)
        return cl.replot_datasets(a.out, "s4c_", specs, a.min_base)
    rng = cl.start(a)
    t0 = time.time()
    for p in real_class_problems(a):
        p.T = a.T
        m = min(max(a.m, 10 * p.d), p.n // 4)          # sketched Newton needs m well above d (ENEM: d = 145)
        sp = specs(p.name)
        print(f"[{p.name}] n={p.n} d={p.d} minority={p.g.mean():.3f} exact gaps:", {k: round(float(v), 4) for k, v in p.base.items()}, flush=True)
        cl.table(cl.experiment(p, m, a.reps, rng), a.out, f"s4c_{p.name}_table", METH,
                 f"{p.name} (m={m}): distortion relative to the exact logistic model; positive = less fair.")
        ms = [x for x in [4 * p.d * 2 ** i for i in range(7)] if x < p.n // 4]
        cl.emit(a.out, f"s4c_{p.name}_m", cl.sweep(lambda x: p, ms, "m", a.reps, rng, m, PLOT_METH), sp[f"s4c_{p.name}_m"], a.min_base)
        cl.emit(a.out, f"s4c_{p.name}_steps", cl.sweep(lambda x: p, [1, 2, 3, 4, 6], "T", a.reps, rng, m, NEWTON), sp[f"s4c_{p.name}_steps"], a.min_base)
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()