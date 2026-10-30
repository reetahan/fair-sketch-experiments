#!/usr/bin/env python3
"""Section 5, classification, real data: one-shot targeted coreset (oracle, proxy, stratified) vs uniform, Section 3
importance sampling, LogReg-sketch (authors' code) and the Newton sketch (T steps) as a reference; see lib_coreset.py.
on LSAC (pass_bar), ACS California 2018 (income > 50k) AND HMDA California 2018 (loan approval), all three by default.
Outputs per dataset: s5c_<name>_table.{csv,tex}; s5c_<name>_m.{csv,png}
Usage: python class_s5_real.py --out results/class_s5_real --reps 60 [--quick] [--no_law] [--no_acs] [--no_hmda] [--acs_state CA] [--replot]"""
import argparse, time
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # library modules live at the repo root
import lib_classification as cl
from lib_coreset import S5C_METH as S5_METH, run_coreset
DEFAULT_T = 3       # Newton steps for the Newton-sketch reference line only
from lib_data import real_class_problems, add_real_class_args


def specs(name):
    return {f"s5c_{name}_m": dict(x="m", label="sketch size m", logx=True, meths=S5_METH)}


def main():
    ap = add_real_class_args(cl.add_args(argparse.ArgumentParser(), "results/class_s5_real", m=300, T=DEFAULT_T))
    ap.add_argument("--min_base", type=float, default=0.01, help="omit rate-gap points whose exact gap is below this")
    a = ap.parse_args()
    if a.replot:
        cl.set_style(a.latex)
        return cl.replot_datasets(a.out, "s5c_", specs, a.min_base)
    rng = cl.start(a)
    t0 = time.time()
    for p in real_class_problems(a):
        p.T = a.T
        m = min(max(a.m, 10 * p.d), p.n // 4)          # sketched Newton needs m well above d (ENEM: d = 145)
        sp = specs(p.name)
        print(f"[{p.name}] n={p.n} d={p.d} minority={p.g.mean():.3f} exact gaps:", {k: round(float(v), 4) for k, v in p.base.items()}, flush=True)
        cl.table(cl.experiment(p, m, a.reps, rng, meths=S5_METH, runner=run_coreset), a.out, f"s5c_{p.name}_table", S5_METH,
                 f"{p.name}, m={m}: one-shot coresets; Newton sketch reference with {a.T} steps.")
        ms = [x for x in [5 * p.d * 2 ** i for i in range(6)] if x < p.n // 4]
        run = lambda xs, xname: cl.sweep(lambda x: p, xs, xname, a.reps, rng, m, S5_METH, runner=run_coreset)
        cl.emit(a.out, f"s5c_{p.name}_m", run(ms, "m"), sp[f"s5c_{p.name}_m"], a.min_base)
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()