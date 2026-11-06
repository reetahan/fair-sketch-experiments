#!/usr/bin/env python3
"""Section 5 classification: one-shot targeted coresets, on real data or (--synthetic) synthetic data."""
import argparse, time
import lib_classification as cl
from lib_classification import P, make_data, S5C_METH, run_coreset
from lib_data import real_problems, add_real_args

SPECS = {"s5c_synth_m": dict(x="m", label="sketch size m", logx=True, meths=S5C_METH),
         "s5c_synth_minority": dict(x="frac1", label="minority fraction", logx=False, meths=S5C_METH),
         "s5c_synth_mis": dict(x="mis", label="group model gap", logx=False, meths=S5C_METH)}


def specs(name):
    return {f"s5c_{name}_m": dict(x="m", label="sketch size m", logx=True, meths=S5C_METH)}


def synthetic(a, rng):
    mk = lambda **kw: P(*make_data(a.n, a.d, **{"frac1": 0.1, "seed": a.data_seed, **kw}))
    base = mk()
    base.T = a.T
    print("exact-model gaps:", {k: round(float(v), 4) for k, v in base.base.items()}, flush=True)
    cl.table(cl.experiment(base, a.m, a.reps, rng, meths=S5C_METH, runner=run_coreset), a.out, "s5c_synth_table",
             S5C_METH, f"Classification, synthetic, m={a.m}: one-shot coresets; Newton sketch reference with {a.T} steps.")
    sw, q = set(a.sweep.split(",")), a.quick
    run = lambda name, builder, xs, x, **kw: cl.emit(
        a.out, name, cl.sweep(builder, xs, x, a.reps, rng, a.m, S5C_METH, runner=run_coreset, **kw), SPECS[name])
    if "m" in sw:
        run("s5c_synth_m", lambda x: base, [5 * a.d, 20 * a.d, 80 * a.d] if q else [5 * a.d * 2**i for i in range(6)], "m")
    if "minority" in sw:
        run("s5c_synth_minority", lambda x: mk(frac1=x), [0.02, 0.1, 0.5] if q else [0.02, 0.05, 0.1, 0.2, 0.35, 0.5],
            "frac1", T=a.T)
    if "mis" in sw:
        run("s5c_synth_mis", lambda x: mk(mis=x), [0.0, 1.0, 3.0] if q else [0.0, 0.5, 1.0, 2.0, 3.0], "mis", T=a.T)


def real(a, rng):
    for p in real_problems(a, "classification"):
        p.T = a.T
        m = min(max(a.m, 10 * p.d), p.n // 4)
        print(f"[{p.name}] n={p.n} d={p.d} minority={p.g.mean():.3f} exact gaps:",
              {k: round(float(v), 4) for k, v in p.base.items()}, flush=True)
        cl.table(cl.experiment(p, m, a.reps, rng, meths=S5C_METH, runner=run_coreset), a.out, f"s5c_{p.name}_table",
                 S5C_METH, f"{p.name}, m={m}: one-shot coresets; Newton sketch reference with {a.T} steps.")
        ms = [x for x in [5 * p.d * 2**i for i in range(6)] if x < p.n // 4]
        cl.emit(a.out, f"s5c_{p.name}_m", cl.sweep(lambda x: p, ms, "m", a.reps, rng, m, S5C_METH, runner=run_coreset),
                specs(p.name)[f"s5c_{p.name}_m"], a.min_base)


def main():
    ap = add_real_args(cl.add_args(argparse.ArgumentParser(), None, m=300, T=3, synth=True), 50000)
    ap.set_defaults(data_seed=3)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sweep", default="m,minority,mis", help="synthetic sweeps to run")
    ap.add_argument("--min_base", type=float, default=0.01, help="omit rate-gap points whose exact gap is below this")
    a = ap.parse_args()
    a.out = a.out or ("results/class_s5_synth" if a.synthetic else "results/class_s5_real")
    if a.replot:
        cl.set_style(a.latex)
        return cl.replot(a.out, SPECS) if a.synthetic else cl.replot_datasets(a.out, "s5c_", specs, a.min_base)
    rng = cl.start(a)
    t0 = time.time()
    (synthetic if a.synthetic else real)(a, rng)
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()
