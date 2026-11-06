"""When does targeted sampling beat leverage sampling? Sweeps tau, the tail weight of the feature that drives the DP gap.

Usage: python extensions/tgt_vs_imp_regime.py [--reps 100] [--quick] [--target DP] [--out results/tgt_vs_imp]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib_classification import P, sig
import lib_coreset as lc
from fs_common import PALETTE, legend_above

METHODS = ["c_unif", "c_lev", "c_tgt_oracle", "c_tgt_proxy"]


def make(tau1, n=20000, d=30, k_noise=26, frac1=0.2, gap=1.0, off=-1.0, mis=0.5, tau=0.0, seed=0):
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < frac1).astype(int)
    Z = np.c_[np.ones(n), rng.standard_normal((n, d - 1))]
    Z[g == 1, 1] += gap
    b1 = np.exp(tau1 * rng.standard_normal(n))
    Z[:, 1] *= b1 / np.sqrt((b1 ** 2).mean())
    beta = rng.standard_normal(d) * 0.8 / np.sqrt(d / 5)
    dev = mis * rng.standard_normal(d) / np.sqrt(d)
    noise = slice(d - k_noise, d)
    beta[noise] = 0.0
    dev[noise] = 0.0
    y = np.where(rng.random(n) < sig(Z @ beta + g * (Z @ dev + off)), 1.0, -1.0)
    a = np.exp(tau * rng.standard_normal(n))
    Z[:, noise] *= (a / np.sqrt((a ** 2).mean()))[:, None]
    return P(Z, y, g, lam=1.0, name=f"tau1={tau1}")


def objective(K, q):
    return float(np.sum(np.sum(K ** 2, 1) / q) - np.sum(np.sum(K, 0) ** 2))


def one_tau(tau, ms, reps, target, mu, f, seed):
    p = make(tau, seed=seed)
    p.target, p.mu = target, mu
    a, K = lc.scores(p, p.xh, p.H, target)
    pred = objective(K, p.q_lev) / objective(K, lc._mix(a, mu))
    mets = ["IF", "DP", "TPR", "FPR"] if target == "all" else [target]
    rows = []
    for m in ms:
        p.m_p = max(int(f * m), p.d + 1)
        rng = np.random.default_rng(seed + 1000 + m)
        err = {me: [] for me in METHODS}
        for _ in range(reps):
            for me in METHODS:
                s = p.signed(lc.run_coreset(p, me, m, rng))
                err[me].append(np.mean([abs(s[k]) for k in mets]))
        base = np.mean(err["c_lev"])
        for me in METHODS:
            rows.append(dict(tau=tau, m=m, method=me, mean_abs_rel=np.mean(err[me]),
                             ratio_to_lev=np.mean(err[me]) / base, pred_lev_over_tgt=pred,
                             max_leverage=float(np.max(np.sum(p.A * (p.A @ p.Hi), 1) * p.w))))
    print(f"tau={tau}: predicted V_imp/V_tgt = {pred:.2f}", flush=True)
    return rows


def plot(df, target, out):
    ms = sorted(df.m.unique())
    fig, axes = plt.subplots(1, len(ms), figsize=(3.2 * len(ms), 3.1), squeeze=False)
    COL = {"c_unif": PALETTE["unif"], "c_lev": PALETTE["lev"], "c_tgt_oracle": PALETTE["oracle"], "c_tgt_proxy": PALETTE["proxy"]}
    lab = {"c_unif": "uniform", "c_lev": "leverage",
           "c_tgt_oracle": "targeted (oracle)", "c_tgt_proxy": "targeted (proxy)"}
    for ax, m in zip(axes[0], ms):
        d = df[df.m == m]
        for me in METHODS:
            s = d[d.method == me]
            ax.plot(s.tau, s.ratio_to_lev, marker="o", ms=3, color=COL[me], label=lab[me])
        pr = d[d.method == "c_lev"]
        ax.plot(pr.tau, np.sqrt(1 / pr.pred_lev_over_tgt), "k--", lw=0.8, label=r"predicted oracle, $\sqrt{V_{tgt}/V_{lev}}$")
        ax.axhline(1, color="0.7", lw=0.6)
        ax.set_title(f"m = {m}", fontsize=9)
        ax.set_xlabel(r"$\tau$ (how heavy-tailed the gap feature is)", fontsize=8)
        ax.set_yscale("log")
    axes[0, 0].set_ylabel(f"{target} distortion / leverage", fontsize=8)
    legend_above(fig, axes, 0.88, fontsize=7)
    fig.savefig(f"{out}/tgt_vs_imp_{target}.png", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--target", default="DP", help="IF | DP | TPR | FPR | all")
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--f", type=float, default=0.5, help="pilot fraction m_p / m")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/tgt_vs_imp")
    args = ap.parse_args()
    taus = [0.0, 1.0, 2.0] if args.quick else [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ms = [400] if args.quick else [200, 400, 800]
    reps = 10 if args.quick else args.reps
    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame([r for t in taus for r in one_tau(t, ms, reps, args.target, args.mu, args.f, args.seed)])
    df.to_csv(f"{args.out}/tgt_vs_imp_{args.target}.csv", index=False)

    plot(df, args.target, args.out)
    print(f"wrote {args.out}/tgt_vs_imp_{args.target}.csv and .png")


if __name__ == "__main__":
    main()
