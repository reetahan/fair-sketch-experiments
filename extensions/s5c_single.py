#!/usr/bin/env python3
"""Section 5 classification: targeted scores aimed at one metric, compared with leverage sampling on that metric."""
import os, sys, time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib_classification as cl
from lib_classification import run_coreset

TARGETS = ["IF", "DP", "TPR", "FPR", "all"]
M_FACTORS = [10, 40, 160]


def single_sweep(p, reps, rng, fracs, mus, factors):
    frames = []
    for m in [k * p.d for k in factors if k * p.d < p.n // 4]:
        for f in fracs:
            p.m_p = max(int(round(f * m)), 1)
            ref = cl.experiment(p, m, reps, rng, meths=["c_unif", "c_lev"], runner=run_coreset)
            frames.append(ref.assign(f=f, mu=pd.NA, target="none"))
            for mu in mus:
                p.mu = mu
                for t in TARGETS:
                    p.target = t
                    d = cl.experiment(p, m, reps, rng, meths=["c_tgt_oracle", "c_tgt_proxy"], runner=run_coreset)
                    frames.append(d.assign(f=f, mu=mu, target=t))
                print(f"  m={m} f={f} mu={mu}: done", flush=True)
    return pd.concat(frames)


def main():
    ap = cl.ext_args("results/single_s5c")
    ap.add_argument("--mus", default="0.2,0.5")
    ap.add_argument("--fracs", default="0.25,0.5")
    a = ap.parse_args()
    rng = cl.start(a)
    mus = [float(v) for v in a.mus.split(",")]
    fracs = [float(v) for v in a.fracs.split(",")]
    factors = [10, 40] if a.quick else M_FACTORS
    probs = cl.ext_problems(a)
    t0 = time.time()
    for p in probs:
        print(f"[{p.name}] n={p.n} d={p.d}", flush=True)
        name = "synth" if a.synthetic else p.name
        single_sweep(p, a.reps, rng, fracs, mus, factors).to_csv(f"{a.out}/s5c_single_{name}.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()
