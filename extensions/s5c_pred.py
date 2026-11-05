#!/usr/bin/env python3
"""Predicted variance ratio of targeted to leverage sampling on the real data, from the full-data fit (no sketching)."""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib_classification as cl
import lib_coreset as lc
from lib_classification import P, make_data
from lib_data import real_class_problems, add_real_class_args


def objective(K, q):
    return float(np.sum(np.sum(K**2, 1) / q) - np.sum(np.sum(K, 0) ** 2))


def main():
    ap = add_real_class_args(cl.add_args(argparse.ArgumentParser(), "results/pred_s5c", m=300, T=3, synth=True))
    ap.add_argument("--synthetic", action="store_true")
    ap.set_defaults(data_seed=3)
    a = ap.parse_args()
    cl.start(a)
    probs = [P(*make_data(a.n, a.d, frac1=0.1, seed=a.data_seed))] if a.synthetic else real_class_problems(a)
    for p in probs:
        name = "synth" if a.synthetic else p.name
        rows = []
        for target in ["all", "IF", "DP", "TPR", "FPR"]:
            s, K = lc.scores(p, p.xh, p.H, target)
            base = None if target == "all" else p.base[target]
            V_unif = objective(K, np.full(p.n, 1.0 / p.n))
            for mu in [0.2, 0.5]:
                q = lc._mix(s, mu)
                V_lev, V_tgt = objective(K, p.q_lev), objective(K, q)
                row = dict(data=name, target=target, mu=mu, V_lev=V_lev, V_tgt=V_tgt, V_unif=V_unif, pred=np.sqrt(V_tgt / V_lev),
                           tgt_over_unif=np.sqrt(V_tgt / V_unif), lev_over_unif=np.sqrt(V_lev / V_unif), base=base)
                if target == "all":
                    for j, k in enumerate(["IF", "DP", "TPR", "FPR"]):
                        Kj = K[:, [j]]
                        row[f"{k}_tgt_over_unif"] = np.sqrt(objective(Kj, q) / objective(Kj, np.full(p.n, 1.0 / p.n)))
                        row[f"{k}_share"] = objective(Kj, q) / V_tgt
                rows.append(row)
        d = pd.DataFrame(rows)
        d.to_csv(f"{a.out}/s5c_pred_{name}.csv", index=False)
        print(d.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
