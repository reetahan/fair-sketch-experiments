#!/usr/bin/env python3
"""Section 5 classification: wall-clock cost of each method from scratch, single-threaded, on the same node."""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[v] = "1"
import sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib_classification as cl
from lib_classification import fit, sig
from lib_classification import run_coreset

M_FACTORS = [10, 40, 160]
METHODS = ["c_unif", "c_lev", "c_tgt_oracle", "c_tgt_proxy", "c_strat_tgt_proxy", "logreg_code", "newton_cs"]


def full_fit(p):
    x = fit(p.A, np.ones(p.n), p.lam)
    s = sig(-(p.A @ x))
    return x, (p.A * (s * (1 - s))[:, None]).T @ p.A + p.lam * np.eye(p.d)


def leverage(p):
    lev = (p.Z * np.linalg.solve(p.Z.T @ p.Z + p.lam * np.eye(p.d), p.Z.T).T).sum(1)
    return lev / lev.sum()


SETUP = {"c_lev": leverage, "c_tgt_oracle": full_fit}


def one(p, meth, m, rng):
    t0 = time.perf_counter()
    if meth in SETUP:
        SETUP[meth](p)
    t1 = time.perf_counter()
    run_coreset(p, meth, m, rng)
    t2 = time.perf_counter()
    return t1 - t0, t2 - t1


def bench(p, name, reps, rng, factors):
    rows = []
    t0 = time.perf_counter()
    full_fit(p)
    full = time.perf_counter() - t0
    for m in [k * p.d for k in factors if k * p.d < p.n // 4]:
        times = {me: [] for me in METHODS}
        for r in range(reps):
            for i in range(len(METHODS)):
                me = METHODS[(i + r) % len(METHODS)]
                times[me].append(one(p, me, m, rng))
        for me, t in times.items():
            t = np.array(t)
            tot = t.sum(1)
            rows.append(dict(data=name, n=p.n, d=p.d, m=m, method=me, reps=reps, setup_median=np.median(t[:, 0]),
                             draw_fit_median=np.median(t[:, 1]), total_median=np.median(tot),
                             total_q10=np.quantile(tot, 0.1), total_q90=np.quantile(tot, 0.9), full_fit_once=full))
        print(f"  m={m}: done", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = cl.ext_args("results/time_s5c")
    a = ap.parse_args()
    rng = cl.start(a)
    probs = cl.ext_problems(a)
    for p in probs:
        p.T = a.T
        name = "synth" if a.synthetic else p.name
        print(f"[{name}] n={p.n} d={p.d}", flush=True)
        d = bench(p, name, a.reps, rng, [10, 40] if a.quick else M_FACTORS)
        d.to_csv(f"{a.out}/s5c_time_{name}.csv", index=False)
        print(d.drop(columns=["data", "n", "d", "reps"]).round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
