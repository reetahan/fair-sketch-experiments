#!/usr/bin/env python3
"""Fairness distortion of row-sampling sketches when the model is an MLP instead of a linear model.

Sampling probabilities come from the full-data linear model; the MLP and the linear model are fit on the same draws.

Usage:
  python extensions/mlp_fairness.py --task both --out results/mlp_ext --diab --enem_path ENEM.csv
  python extensions/mlp_fairness.py --task both --synthetic --quick --out results/mlp_test
"""
import argparse, math, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import fs_common as fc
import lib_classification as cl
from lib_data import real_problems, add_real_args
from lib_classification import scores as tgt_scores, _strat as strat_draw, _mix, _draw

REG_METRICS = ["IF", "IF_p95", "GFE", "MSE"]
CLS_METRICS = ["IF", "IF_p95", "DP", "TPR", "FPR"]
REG_METH = ["unif", "lev", "strat"]
CLS_METH = ["unif", "lev", "tgt", "strat_tgt"]
LABEL = {"unif": "uniform", "lev": "leverage", "strat": "stratified uniform (Sec. 5)",
         "tgt": "targeted (Sec. 5)", "strat_tgt": "stratified targeted (Sec. 5)"}
P = fc.PALETTE
COLOR = {"unif": P["unif"], "lev": P["lev"], "strat": P["strat_unif"], "tgt": P["oracle"], "strat_tgt": P["strat_proxy"]}


class Task:

    def __init__(self, kind, p):
        self.kind, self.p, self.name = kind, p, p.name
        Z = p.A if kind == "reg" else p.Z
        self.keep = np.std(Z, axis=0) > 1e-12
        mu, sd = Z[:, self.keep].mean(0), Z[:, self.keep].std(0)
        self.X = ((Z[:, self.keep] - mu) / sd).astype(np.float32)
        self.sd = sd
        self.g = np.asarray(p.g).astype(int)
        self.n = len(self.g)
        if kind == "reg":
            self.y_mean, self.y_sd = p.b.mean(), p.b.std()
            self.t = ((p.b - self.y_mean) / self.y_sd).astype(np.float32)
            self.metrics = REG_METRICS
        else:
            self.t = (p.y > 0).astype(np.float32)
            self.sets = p.sets
            self.metrics = CLS_METRICS


def load_tasks(a):
    out = []
    if a.task in ("reg", "both"):
        if a.synthetic:
            out.append(Task("reg", fc.make_synth(n=a.n, d=a.d, frac1=0.05, lev1=1.0, noise1=3.0, seed=a.seed,
                                                  name="synth-reg")))
        else:
            out += [Task("reg", p) for p in real_problems(a)]
    if a.task in ("cls", "both"):
        if a.synthetic:
            Z, y, g = cl.make_data(n=a.n, d=a.d, seed=a.seed)
            out.append(Task("cls", cl.P(Z, y, g, 1.0, name="synth-cls")))
        else:
            out += [Task("cls", p) for p in real_problems(a, "classification")]
    return out


def reg_strat_split(T, m, target):
    p = T.p
    C, r = p.C[target], p.r
    beta = []
    for k in (0, 1):
        ii = T.g == k
        Cg, y = C[ii], r[ii]
        beta.append(max(ii.sum() * np.sum((Cg ** 2).sum(1) * y ** 2) - np.sum((Cg.T @ y) ** 2), 0.0))
    sb = np.sqrt(beta)
    m0 = int(np.clip(round(m * sb[0] / sb.sum()), 2, m - 2))
    return m0, m - m0


def draw(T, meth, m, rng, a):
    n = T.n
    if meth == "unif":
        return _draw(np.full(n, 1.0 / n), m, rng)
    if meth == "lev":
        return _draw(T.p.q_lev, m, rng)
    if meth == "strat":
        m0, m1 = reg_strat_split(T, m, a.strat_target)
        I, W = [], []
        for k, mg in ((0, m0), (1, m1)):
            idx = np.nonzero(T.g == k)[0]
            Ig, wg = _draw(np.full(len(idx), 1.0 / len(idx)), mg, rng)
            I.append(idx[Ig]); W.append(wg)
        return np.concatenate(I), np.concatenate(W)
    if meth in ("tgt", "strat_tgt"):
        if not hasattr(T, "_scores"):
            T._scores = tgt_scores(T.p, T.p.xh, T.p.H, "all")
        s, K = T._scores
        if meth == "tgt":
            return _draw(_mix(s, a.mu), m, rng)
        return strat_draw(T.p, s, K, m, a.mu, rng)
    raise KeyError(meth)


def fit_linear(T, I, w):
    p = T.p
    if T.kind == "reg":
        sw = np.sqrt(w)
        return np.linalg.lstsq(p.A[I] * sw[:, None], p.b[I] * sw, rcond=None)[0]
    return cl.fit(p.A[I], w, p.lam)


def linear_outputs(T, x):
    Z = T.p.A if T.kind == "reg" else T.p.Z
    f = Z @ x
    if T.kind == "reg":
        f = (f - T.y_mean) / T.y_sd
        gnorm = np.linalg.norm(x[T.keep] * T.sd) / T.y_sd
    else:
        gnorm = np.linalg.norm(x[T.keep] * T.sd)
    return f, np.full(T.n, gnorm)


def fit_mlp(T, I, w, seed, a):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = a.device_
    dims = [T.X.shape[1]] + a.hidden
    layers = []
    for d_in, d_out in zip(dims[:-1], dims[1:]):
        layers += [nn.Linear(d_in, d_out), nn.ReLU()]
    model = nn.Sequential(*layers, nn.Linear(dims[-1], 1)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    w = np.asarray(w, np.float64)
    w = w / w.mean()
    X = torch.as_tensor(T.X[I], device=dev)
    y = torch.as_tensor(T.t[I], device=dev)
    W = torch.as_tensor(w.astype(np.float32), device=dev)
    rs = np.random.default_rng(seed + 7919)
    order = rs.permutation(len(I))
    n_hold = max(1, int(round(0.10 * len(I))))
    hold, fit = (order, order) if len(I) - n_hold < 4 else (order[:n_hold], order[n_hold:])
    bs = int(min(a.batch_size, max(8, math.ceil(len(fit) / 4))))
    lossf = (lambda o, t: (o - t) ** 2) if T.kind == "reg" else \
        (lambda o, t: nn.functional.binary_cross_entropy_with_logits(o, t, reduction="none"))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(a.max_epochs):
        model.train()
        perm = torch.as_tensor(fit)[torch.randperm(len(fit), generator=gen)]
        for lo in range(0, len(perm), bs):
            b = perm[lo:lo + bs].to(dev)
            opt.zero_grad(set_to_none=True)
            loss = torch.mean(W[b] * lossf(model(X[b]).squeeze(-1), y[b]))
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            hb = torch.as_tensor(hold, device=dev)
            val = float(torch.mean(W[hb] * lossf(model(X[hb]).squeeze(-1), y[hb])).cpu())
        if val + 1e-10 < best:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def mlp_outputs(T, model, a):
    import torch
    f, gn = [], []
    for lo in range(0, T.n, 8192):
        xb = torch.as_tensor(T.X[lo:lo + 8192], device=a.device_).requires_grad_(True)
        out = model(xb).squeeze(-1)
        (gr,) = torch.autograd.grad(out.sum(), xb)
        f.append(out.detach().cpu().numpy())
        gn.append(gr.norm(dim=1).cpu().numpy())
    return np.concatenate(f).astype(np.float64), np.concatenate(gn).astype(np.float64)


def metrics(T, f, gnorm):
    out = {"IF": gnorm.mean(), "IF_p95": np.quantile(gnorm, 0.95)}
    g0, g1 = T.g == 0, T.g == 1
    if T.kind == "reg":
        out["GFE"] = f[g0].mean() - f[g1].mean()
        res2 = (f - T.t) ** 2
        out["MSE"] = res2[g0].mean() - res2[g1].mean()
    else:
        pr = 1.0 / (1.0 + np.exp(-f))
        for k in ("DP", "TPR", "FPR"):
            s0, s1 = T.sets[k]
            out[k] = pr[s0].mean() - pr[s1].mean()
    return out


def summarize(draws, floors):
    rows = []
    for (model, meth, m, met), d in draws.groupby(["model", "method", "m", "metric"]):
        ref = d["reference"].iloc[0]
        rel = np.abs(d["value"].values) / abs(ref) - 1.0
        fl = floors.get((model, met), np.nan)
        rows.append(dict(model=model, method=meth, m=m, metric=met, reps=len(d), base=ref, mean_rel=rel.mean(),
                         mean_abs_rel=np.abs(rel).mean(), q10_rel=np.quantile(rel, 0.1), q90_rel=np.quantile(rel, 0.9),
                         frac_worse=(rel > 0).mean(), floor_abs_rel=fl))
    return pd.DataFrame(rows)


def plot(summary, T, path, min_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mets = [m for m in T.metrics if m != "IF_p95"]
    fig, axes = plt.subplots(1, len(mets), figsize=(3.2 * len(mets), 2.9), squeeze=False)
    for ax, met in zip(axes[0], mets):
        d = summary[summary.metric == met]
        if d.empty or abs(d.base.iloc[0]) < min_base:
            ax.text(0.5, 0.5, f"full-data gap below {min_base}", ha="center", va="center", transform=ax.transAxes, fontsize=7)
        else:
            for model, ls in (("mlp", "-"), ("linear", "--")):
                for meth, g in d[d.model == model].groupby("method"):
                    g = g.sort_values("m")
                    ax.plot(g.m, g.mean_abs_rel, ls, marker="o", ms=3, color=COLOR[meth],
                            label=f"{LABEL[meth]}" if model == "mlp" else None)
            fl = d[d.model == "mlp"].floor_abs_rel.iloc[0] if (d.model == "mlp").any() else np.nan
            if np.isfinite(fl):
                ax.axhspan(0, fl, color="0.85", lw=0, label="MLP seed-to-seed spread (full data)")
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(met, fontsize=9); ax.set_xlabel("sketch size $m$", fontsize=8); ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("mean $|$relative change$|$\n(MLP solid, linear dashed)", fontsize=8)
    fc.legend_above(fig, axes, 0.9, fontsize=7)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_task(T, a, rng):
    tag = f"mlp_{T.kind}_{T.name}"
    meths = [m for m in (REG_METH if T.kind == "reg" else CLS_METH) if m in a.methods]
    models = ["mlp"] + ([] if a.no_linear else ["linear"])
    d_lin = T.p.A.shape[1]
    ms = a.ms or [x for x in [4 * d_lin * 2 ** i for i in range(7)] if x < T.n // 4]
    print(f"[{T.name}] {T.kind} n={T.n} features={T.X.shape[1]} minority={T.g.mean():.3f} m={ms}", flush=True)


    ref, floors = {}, {}
    full_vals = []
    for s in range(a.full_seeds):
        t0 = time.time()
        f, gn = mlp_outputs(T, fit_mlp(T, np.arange(T.n), np.ones(T.n), 10_000 + s, a), a)
        full_vals.append(metrics(T, f, gn))
        print(f"  full-data MLP seed {s}: {time.time() - t0:.0f}s", flush=True)
    for met in T.metrics:
        v = np.array([fv[met] for fv in full_vals])
        ref[("mlp", met)] = v.mean()
        floors[("mlp", met)] = np.abs(np.abs(v) / abs(v.mean()) - 1).mean() if len(v) > 1 else np.nan
    lin_full = metrics(T, *linear_outputs(T, fit_linear(T, np.arange(T.n), np.ones(T.n))))
    for met in T.metrics:
        ref[("linear", met)] = lin_full[met]
        floors[("linear", met)] = 0.0

    rows = [dict(model="mlp", method="full (other seeds)", m=T.n, rep=s, metric=met, value=fv[met], reference=ref[("mlp", met)])
            for s, fv in enumerate(full_vals) for met in T.metrics]
    for m in ms:
        t0 = time.time()
        for meth in meths:
            for rep in range(a.reps):
                I, w = draw(T, meth, m, rng, a)
                for model in models:
                    if model == "mlp":
                        f, gn = mlp_outputs(T, fit_mlp(T, I, w, 20_000 + 1000 * rep + ms.index(m), a), a)
                    else:
                        f, gn = linear_outputs(T, fit_linear(T, I, w))
                    for met, v in metrics(T, f, gn).items():
                        rows.append(dict(model=model, method=meth, m=m, rep=rep, metric=met, value=v, reference=ref[(model, met)]))
        print(f"  m={m}: {time.time() - t0:.0f}s", flush=True)
        draws = pd.DataFrame(rows)
        draws.to_csv(os.path.join(a.out, f"{tag}_draws.csv"), index=False)
        summ = summarize(draws[draws.method != "full (other seeds)"], floors)
        summ.to_csv(os.path.join(a.out, f"{tag}_m.csv"), index=False)
    plot(summ, T, os.path.join(a.out, f"{tag}_m.png"), a.min_base)
    return summ


def main():
    global a_hidden_str
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", default="both", choices=["reg", "cls", "both"])
    ap.add_argument("--out", default="results/mlp_ext")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--full_seeds", type=int, default=5)
    ap.add_argument("--ms", default=None, help="comma-separated sketch sizes (default 4d * 2^i below n/4)")
    ap.add_argument("--methods", default="unif,lev,strat,tgt,strat_tgt")
    ap.add_argument("--strat_target", default="GFE", choices=["IF", "GFE", "MSE"])
    ap.add_argument("--mu", type=float, default=0.2, help="uniform mixing weight for targeted sampling")
    ap.add_argument("--hidden", default="64,32", help="hidden widths (misssketch mlp_small = 64,32; mlp_medium = 256,128,64)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no_linear", action="store_true", help="skip the linear model on the same draws")
    ap.add_argument("--min_base", type=float, default=0.01, help="do not plot a gap whose full-data value is below this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="tiny run for testing")
    ap.add_argument("--synthetic", action="store_true", help="synthetic data instead of the real data sets")
    ap.add_argument("--n", type=int, default=20000, help="synthetic rows")
    ap.add_argument("--d", type=int, default=10, help="synthetic features")
    add_real_args(ap)
    a = ap.parse_args()
    import torch
    a.device_ = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    a.hidden = [int(v) for v in a.hidden.split(",") if v]
    a_hidden_str = "x".join(map(str, a.hidden))
    a.methods = set(a.methods.split(","))
    a.ms = [int(v) for v in a.ms.split(",")] if a.ms else None
    if a.quick:
        a.reps, a.full_seeds, a.max_epochs, a.n = min(a.reps, 3), min(a.full_seeds, 2), min(a.max_epochs, 30), min(a.n, 4000)
        a.ms = a.ms or [80, 320]
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    for T in load_tasks(a):
        s = run_task(T, a, rng)
        show = s[s.metric.isin(["IF", "GFE", "MSE", "DP"])].pivot_table(index=["metric", "m"], columns=["model", "method"],
                                                                         values="mean_abs_rel")
        print(show.round(3).to_string(), "\n", flush=True)
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/ (device {a.device_})")


a_hidden_str = ""
if __name__ == "__main__":
    main()
