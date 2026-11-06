#!/usr/bin/env python3
"""Predicted variance ratio of targeted to leverage sampling on the real data, from the full-data fit (no sketching)."""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib_classification as cl


def main():
    ap = cl.ext_args("results/pred_s5c")
    a = ap.parse_args()
    cl.start(a)
    probs = cl.ext_problems(a)
    for p in probs:
        name = "synth" if a.synthetic else p.name
        rows = []
        for target in ["all", "IF", "DP", "TPR", "FPR"]:
            s, K = cl.scores(p, p.xh, p.H, target)
            base = None if target == "all" else p.base[target]
            V_unif = cl.objective(K, np.full(p.n, 1.0 / p.n))
            for mu in [0.2, 0.5]:
                q = cl._mix(s, mu)
                V_lev, V_tgt = cl.objective(K, p.q_lev), cl.objective(K, q)
                row = dict(data=name, target=target, mu=mu, V_lev=V_lev, V_tgt=V_tgt, V_unif=V_unif, pred=np.sqrt(V_tgt / V_lev),
                           tgt_over_unif=np.sqrt(V_tgt / V_unif), lev_over_unif=np.sqrt(V_lev / V_unif), base=base)
                if target == "all":
                    for j, k in enumerate(["IF", "DP", "TPR", "FPR"]):
                        Kj = K[:, [j]]
                        row[f"{k}_tgt_over_unif"] = np.sqrt(cl.objective(Kj, q) / cl.objective(Kj, np.full(p.n, 1.0 / p.n)))
                        row[f"{k}_share"] = cl.objective(Kj, q) / V_tgt
                rows.append(row)
        d = pd.DataFrame(rows)
        d.to_csv(f"{a.out}/s5c_pred_{name}.csv", index=False)
        print(d.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
