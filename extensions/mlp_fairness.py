#!/usr/bin/env python3
"""Extension: fairness distortion of row-sampling sketches when the model is an MLP instead of a linear model.

Question. Sections 4 and 5 analyze least squares and ridge logistic regression. Does compressing the data by row sampling
distort the fairness of a neural network in the same way, and do the paper's sampling rules (leverage, stratified,
targeted), computed from the LINEAR model, still help when the deployed model is an MLP?

Design.
  * Only row-sampling sketches are run. Gaussian sketches and CountSketch replace rows by random combinations of rows, which
    is meaningful for a linear model (the loss is a function of A x) but not for an MLP, whose prediction on a mixed row is
    not a mix of predictions; for classification they also erase the labels (Proposition badchoice).
  * Each sketch draws m indices with probabilities q (multinomial counts K_i) and trains on the distinct rows with weights
    w_i = K_i / (m q_i), the same estimator as the paper. Weights are rescaled to mean one before training.
  * The same draws are used to fit the linear model of the paper and the MLP, so the two are compared on identical samples.
  * The MLP and its training loop follow misssketch/broader.py: ReLU MLP, AdamW, early stopping on 10% of the training rows,
    weighted squared loss (regression) or weighted logistic loss (classification).
  * Training noise. An MLP fit on the FULL data still changes with the random seed (initialization, minibatch order). The
    reference value of each metric is the average over --full_seeds full-data fits, and the script also reports the
    distortion of each full-data fit from that average ("full data, other seeds"). Sketch distortion below this floor is
    not attributable to sketching.

Metrics (evaluated on all n rows, the population the model serves, as in the paper).
  IF   the model's average input-gradient norm, mean_i ||grad_z f(z_i)||, over the non-constant features. For a linear model
       this is ||x|| without the intercept, the Lipschitz constant used in the paper. f is the prediction (regression) or
       the logit (classification). Also IF_p95, the 95th percentile, a local-Lipschitz estimate.
  GFE  regression: mean prediction in group 0 minus group 1 (equals v'x for a linear model).
  MSE  regression: MSE in group 0 minus MSE in group 1.
  DP, TPR, FPR  classification: gaps in mean predicted probability, overall and among positives / negatives, as in the paper.
  Distortion of a draw: rel = |metric(sketch)| / |reference| - 1 (signed; positive = less fair), summarized as
  mean_abs_rel, mean_rel, q10_rel, q90_rel and frac_worse, the same columns as the classification CSVs.

Methods (probabilities computed from the full-data LINEAR model, the oracle versions of the paper's rules).
  regression:      unif; lev (leverage, p.q_lev); strat (uniform within groups, split m_g ~ sqrt(beta_g) for the
                   --strat_target, Section 5).
  classification:  unif; lev (p.q_lev); tgt (targeted, scores for IF, DP, TPR, FPR, mu = 0.2); strat_tgt (stratified targeted).

Outputs in --out:
  mlp_<task>_<dataset>_draws.csv     every draw: model, method, m, rep, metric, value, reference
  mlp_<task>_<dataset>_m.csv         summary per model, method, m, metric (+ floor_abs_rel, the seed-noise floor)
  mlp_<task>_<dataset>_m.png         mean |relative change| against m, MLP solid and linear dashed, seed floor shaded

Usage:
  python extensions/mlp_fairness.py --task both --out results/mlp_ext --diab --enem_path ENEM.csv          # real data
  python extensions/mlp_fairness.py --task both --synthetic --quick --out results/mlp_test                  # smoke test
Options: --hidden 64,32 (misssketch mlp_small; 256,128,64 = mlp_medium), --full_seeds 5, --reps 20, --ms 80,160,...,
--device auto, --max_epochs 200, --patience 20, --no_linear."""
import argparse, math, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # library modules live at the repo root
import fs_common as fc
import lib_classification as cl
from lib_coreset import scores as tgt_scores, _strat as strat_draw, _mix, _draw

REG_METRICS = ["IF", "IF_p95", "GFE", "MSE"]
CLS_METRICS = ["IF", "IF_p95", "DP", "TPR", "FPR"]
REG_METH = ["unif", "lev", "strat"]
CLS_METH = ["unif", "lev", "tgt", "strat_tgt"]
LABEL = {"unif": "uniform", "lev": "leverage", "strat": "stratified uniform (Sec. 5)",
         "tgt": "targeted (Sec. 5)", "strat_tgt": "stratified targeted (Sec. 5)"}
P = fc.PALETTE
COLOR = {"unif": P["unif"], "lev": P["lev"], "strat": P["strat_unif"], "tgt": P["oracle"], "strat_tgt": P["strat_proxy"]}


# ------------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------------
class Task:
    """Common view of a regression Problem or a classification P: features X (non-constant columns only, for the MLP),
    the full design used by the linear model, target, groups, evaluation sets."""

    def __init__(self, kind, p):
        self.kind, self.p, self.name = kind, p, p.name
        Z = p.A if kind == "reg" else p.Z
        self.keep = np.std(Z, axis=0) > 1e-12                 # drop the intercept; the MLP has its own biases
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
            out += [Task("reg", p) for p in fc.real_problems(a)]
    if a.task in ("cls", "both"):
        if a.synthetic:
            Z, y, g = cl.make_data(n=a.n, d=a.d, seed=a.seed)
            out.append(Task("cls", cl.P(Z, y, g, 1.0, name="synth-cls")))
        else:
            from lib_data import real_class_problems
            out += [Task("cls", p) for p in real_class_problems(a)]
    return out


# ------------------------------------------------------------------------------------------------
# sampling (probabilities from the full-data linear model)
# ------------------------------------------------------------------------------------------------
def reg_strat_split(T, m, target):
    """Oracle split for uniform sampling within groups: m_g ~ sqrt(beta_g), beta_g = n_g s_g - zeta_g (Definition beta)."""
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
    """Indices of the distinct sampled rows and their weights K_i/(m q_i)."""
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


# ------------------------------------------------------------------------------------------------
# models
# ------------------------------------------------------------------------------------------------
def fit_linear(T, I, w):
    """The paper's model on the same draw: weighted least squares, or weighted ridge logistic regression (lib_classification.fit)."""
    p = T.p
    if T.kind == "reg":
        sw = np.sqrt(w)
        return np.linalg.lstsq(p.A[I] * sw[:, None], p.b[I] * sw, rcond=None)[0]
    return cl.fit(p.A[I], w, p.lam)


def linear_outputs(T, x):
    """Prediction (or logit) and the input-gradient norm of the linear model, in the MLP's standardized feature units."""
    Z = T.p.A if T.kind == "reg" else T.p.Z
    f = Z @ x
    if T.kind == "reg":
        f = (f - T.y_mean) / T.y_sd                            # same scale as the MLP's standardized target
        gnorm = np.linalg.norm(x[T.keep] * T.sd) / T.y_sd
    else:
        gnorm = np.linalg.norm(x[T.keep] * T.sd)
    return f, np.full(T.n, gnorm)


def fit_mlp(T, I, w, seed, a):
    """ReLU MLP trained as in misssketch/broader.py (_torch_fit): AdamW, early stopping on 10% of the training rows."""
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
    w = w / w.mean()                                           # misssketch _normalize_sample_weights (floor 0)
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
    """Output and per-row input-gradient norm on all n rows (each output depends only on its own row)."""
    import torch
    f, gn = [], []
    for lo in range(0, T.n, 8192):
        xb = torch.as_tensor(T.X[lo:lo + 8192], device=a.device_).requires_grad_(True)
        out = model(xb).squeeze(-1)
        (gr,) = torch.autograd.grad(out.sum(), xb)
        f.append(out.detach().cpu().numpy())
        gn.append(gr.norm(dim=1).cpu().numpy())
    return np.concatenate(f).astype(np.float64), np.concatenate(gn).astype(np.float64)


# ------------------------------------------------------------------------------------------------
# metrics
# ------------------------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------------------------
def run_task(T, a, rng):
    tag = f"mlp_{T.kind}_{T.name}"
    meths = [m for m in (REG_METH if T.kind == "reg" else CLS_METH) if m in a.methods]
    models = ["mlp"] + ([] if a.no_linear else ["linear"])
    d_lin = T.p.A.shape[1]
    ms = a.ms or [x for x in [4 * d_lin * 2 ** i for i in range(7)] if x < T.n // 4]
    print(f"[{T.name}] {T.kind} n={T.n} features={T.X.shape[1]} minority={T.g.mean():.3f} m={ms}", flush=True)

    # references: average over full-data fits (MLP: several seeds; linear: deterministic), and the seed-noise floor
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
        draws = pd.DataFrame(rows)                              # rewrite after every m, so a long run can be inspected
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
    fc.add_real_args(ap)
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
