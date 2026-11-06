#!/usr/bin/env python3
"""Section 4 classification: distortion of sketched ridge logistic regression, on real data or (--synthetic) synthetic."""
import argparse, time
import lib_classification as cl
from lib_classification import P, make_data, METH, PLOT_METH, NEWTON
from lib_data import real_problems, add_real_args

SPECS = {"s4c_synth_m": dict(x="m", label="sketch size m", logx=True, meths=PLOT_METH),
         "s4c_synth_gap": dict(x="gap", label="group shift", logx=False, meths=PLOT_METH),
         "s4c_synth_minority": dict(x="frac1", label="minority fraction", logx=False, meths=PLOT_METH),
         "s4c_synth_steps": dict(x="T", label="Newton steps", logx=False, meths=NEWTON)}


def specs(name):
    return {f"s4c_{name}_m": dict(x="m", label="sketch size m", logx=True, meths=PLOT_METH),
            f"s4c_{name}_steps": dict(x="T", label="Newton steps", logx=False, meths=NEWTON)}


def synthetic(a, rng):
    mk = lambda **kw: P(*make_data(a.n, a.d, seed=a.data_seed, **kw))
    p = mk()
    p.T = a.T
    print("exact-model gaps:", {k: round(float(v), 4) for k, v in p.base.items()}, flush=True)
    cl.table(cl.experiment(p, a.m, a.reps, rng), a.out, "s4c_synth_table", METH,
             f"Classification, synthetic (m={a.m}): distortion relative to the exact logistic model; positive = less fair.")
    sw, q = set(a.sweep.split(",")), a.quick
    run = lambda name, builder, xs, x, meths=PLOT_METH, **kw: cl.emit(
        a.out, name, cl.sweep(builder, xs, x, a.reps, rng, a.m, meths, **kw), SPECS[name])
    if "m" in sw:
        run("s4c_synth_m", lambda x: p, [5 * a.d, 20 * a.d, 80 * a.d, 320 * a.d] if q else [5 * a.d * 2**i for i in range(7)], "m")
    if "gap" in sw:
        run("s4c_synth_gap", lambda x: mk(gap=x), [0.5, 2.0, 3.5] if q else [0.5, 1.0, 2.0, 3.0, 4.0], "gap", T=a.T)
    if "minority" in sw:
        run("s4c_synth_minority", lambda x: mk(frac1=x), [0.05, 0.2, 0.5] if q else [0.02, 0.05, 0.1, 0.2, 0.35, 0.5],
            "frac1", T=a.T)
    if "steps" in sw:
        run("s4c_synth_steps", lambda x: p, [1, 2, 4] if q else [1, 2, 3, 4, 6, 10], "T", NEWTON, T=None)


def real(a, rng):
    for p in real_problems(a, "classification"):
        p.T = a.T
        m = min(max(a.m, 10 * p.d), p.n // 4)
        sp = specs(p.name)
        print(f"[{p.name}] n={p.n} d={p.d} minority={p.g.mean():.3f} exact gaps:",
              {k: round(float(v), 4) for k, v in p.base.items()}, flush=True)
        cl.table(cl.experiment(p, m, a.reps, rng), a.out, f"s4c_{p.name}_table", METH,
                 f"{p.name} (m={m}): distortion relative to the exact logistic model; positive = less fair.")
        ms = [x for x in [5 * p.d * 2**i for i in range(7)] if x < p.n // 4]
        cl.emit(a.out, f"s4c_{p.name}_m", cl.sweep(lambda x: p, ms, "m", a.reps, rng, m, PLOT_METH),
                sp[f"s4c_{p.name}_m"], a.min_base)
        cl.emit(a.out, f"s4c_{p.name}_steps", cl.sweep(lambda x: p, [1, 2, 3, 4, 6], "T", a.reps, rng, m, NEWTON),
                sp[f"s4c_{p.name}_steps"], a.min_base)


def main():
    ap = add_real_args(cl.add_args(argparse.ArgumentParser(), None, m=500, T=10, synth=True), 50000)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sweep", default="m,gap,minority,steps", help="synthetic sweeps to run")
    ap.add_argument("--min_base", type=float, default=0.01, help="omit rate-gap points whose exact gap is below this")
    a = ap.parse_args()
    a.out = a.out or ("results/class_s4_synth" if a.synthetic else "results/class_s4_real")
    if a.replot:
        cl.set_style(a.latex)
        return cl.replot(a.out, SPECS) if a.synthetic else cl.replot_datasets(a.out, "s4c_", specs, a.min_base)
    rng = cl.start(a)
    t0 = time.time()
    (synthetic if a.synthetic else real)(a, rng)
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()
