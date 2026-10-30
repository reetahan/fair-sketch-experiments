
import os
import numpy as np, pandas as pd
from fs_common import set_style, plot_sd, sd_layout, save_method_table
from lib_logreg_sketch import fit_logreg_sketch_code

sig = lambda t: 0.5 * (1 + np.tanh(0.5 * t))
METH = ["samp_unif", "samp_imp", "newton_cs", "newton_gauss", "logreg_sketch", "naive_gauss", "naive_cs"]
LAB = {"samp_unif": "1: sampled, uniform", "samp_imp": "1: sampled, importance (oracle)", "newton_cs": "2: Newton-Hessian, CountSketch",
       "newton_gauss": "2: Newton-Hessian, Gaussian", "logreg_sketch": "3: LogReg-sketch (Munteanu et al. 2021)", "naive_gauss": "3: naive Gaussian", "naive_cs": "3: naive CountSketch"}
COL = {"samp_unif": "tab:blue", "samp_imp": "tab:red", "newton_cs": "tab:orange", "newton_gauss": "tab:green", "logreg_sketch": "tab:purple",
       "naive_gauss": "tab:gray", "naive_cs": "tab:brown"}
MET = ["IF", "DP", "TPR", "FPR"]
RATE_METRICS = ("DP", "TPR", "FPR")


def make_data(n=10000, d=10, frac1=0.3, gap=2.0, mis=1.0, off=-2.0, seed=0):
    rng = np.random.default_rng(seed)
    g = (rng.random(n) < frac1).astype(int)
    Z = np.c_[np.ones(n), rng.standard_normal((n, d - 1))]
    Z[g == 1, 1] += gap
    beta = rng.standard_normal(d) * 0.8 / np.sqrt(d / 5)
    dev = mis * rng.standard_normal(d) / np.sqrt(d)
    y = np.where(rng.random(n) < sig(Z @ beta + g * (Z @ dev + off)), 1.0, -1.0)
    return Z, y, g


def fit(A, wts, lam, iters=60):
    """argmin sum_j wts_j ln(1+exp(-a_j^T x)) + lam/2 ||x||^2 by damped Newton; A rows are label-folded."""
    d = A.shape[1]
    x = np.zeros(d)
    L = lambda x: wts @ np.logaddexp(0, -A @ x) + 0.5 * lam * x @ x
    for _ in range(iters):
        s = sig(-(A @ x))
        gr = -(A.T @ (wts * s)) + lam * x
        H = (A * (wts * s * (1 - s))[:, None]).T @ A + lam * np.eye(d)
        st = np.linalg.solve(H, gr)
        t, f0 = 1.0, L(x)
        while L(x - t * st) > f0 - 1e-4 * t * gr @ st and t > 1e-12:
            t *= 0.5
        x = x - t * st
        if np.linalg.norm(gr) < 1e-10:
            break
    return x


class P:
    """Exact logistic model plus fairness metrics. Z includes the intercept column; y in {-1,+1}; g in {0,1} (1 = minority)."""
    def __init__(self, Z, y, g, lam=1.0, name="synthetic"):
        self.Z, self.y, self.g, self.lam, self.name = Z, y, g, lam, name
        self.n, self.d = Z.shape
        self.A = y[:, None] * Z
        self.xh = fit(self.A, np.ones(self.n), lam)
        s = sig(-(self.A @ self.xh))
        self.w = s * (1 - s)
        self.H = (self.A * self.w[:, None]).T @ self.A + lam * np.eye(self.d)
        self.Hi = np.linalg.inv(self.H)
        L = np.linalg.cholesky(self.Hi)
        self.grad_i = self.A * s[:, None]                      # per-row gradient at x^ (up to sign)
        self.q_imp = np.linalg.norm(self.grad_i @ L, axis=1)   # ||H^{-1/2} s_i||
        self.q_imp = 0.5 * self.q_imp / self.q_imp.sum() + 0.5 / self.n
        self.sets = {"DP": [g == 0, g == 1], "TPR": [(g == 0) & (y > 0), (g == 1) & (y > 0)],
                     "FPR": [(g == 0) & (y < 0), (g == 1) & (y < 0)]}
        self.f = lambda x: np.logaddexp(0, -(self.A @ x)).sum() + 0.5 * lam * x @ x
        self.fh = self.f(self.xh)
        self.base = self.gaps(self.xh)

    def gaps(self, x):
        p = sig(self.Z @ x)
        o = {k: p[r[0]].mean() - p[r[1]].mean() for k, r in self.sets.items()}
        o["IF"] = np.linalg.norm(x)
        return o

    def signed(self, x):
        gp = self.gaps(x)
        out = {"IF": gp["IF"] / self.base["IF"] - 1}
        for k in ("DP", "TPR", "FPR"):
            out[k] = (abs(gp[k]) - abs(self.base[k])) / abs(self.base[k])
        return out


def logreg_sketch(A, m, rng, b=3.0, levels=3, frac_unif=1 / 3):
    """LogReg-sketch of Munteanu, Omlor, Woodruff (ICML 2021), following Sec. 3.1 and 5 of the paper.
    Each folded row p gets a level h_p in {0..levels-1} with Pr = 1/(beta b^h) and a uniform bucket of N at that level.
    Bucket rows are UNSIGNED sums; the factor b^h beta = 1/Pr(h) multiplies the ROW (inside the loss), and every sketch row
    carries the same loss weight 1/(N*levels). A weighted uniform sample T (weight n/mT) is appended.
    Returns (rows, weights, level) with level = -1 for the uniform sample."""
    n, d = A.shape
    mT = max(d + 1, int(round(frac_unif * m)))
    N = max(1, (m - mT) // levels)
    pr = b ** (-np.arange(levels, dtype=float))
    pr /= pr.sum()
    lev = rng.choice(levels, n, p=pr)
    bucket = lev * N + rng.integers(0, N, n)
    SA = np.zeros((levels * N, d))
    np.add.at(SA, bucket, A)
    keep = np.bincount(bucket, minlength=levels * N) > 0
    blev = np.repeat(np.arange(levels), N)
    SA = SA * (N * levels / pr[blev])[:, None]              # A'' = N*levels * S A
    idx = rng.choice(n, mT)
    rows = np.vstack([SA[keep], A[idx]])
    w = np.concatenate([np.full(keep.sum(), 1.0 / (N * levels)), np.full(mT, n / mT)])
    lv = np.concatenate([blev[keep], np.full(mT, -1)])
    return rows, w, lv


def fit_logreg_sketch(A, m, rng, lam, clip=None, iters=8):
    """Weighted logistic fit on a LogReg-sketch. clip=0.25 reproduces the paper's Ky Fan cut-off used in its experiments:
    on each level only the `clip` fraction of rows with the largest loss (smallest a.x in our sign convention) is kept (the uniform sample is never cut). Solved by
    alternating refit / re-selection; clip=None is the unclipped f_w used by the O(log n) theorem."""
    B, w, lv = logreg_sketch(A, m, rng)
    x = fit(B, w, lam)
    if clip is None:
        return x
    for _ in range(iters):
        mar = B @ x
        wk = w.copy()
        for h in range(lv.max() + 1):
            ix = np.where(lv == h)[0]
            k = max(1, int(np.ceil(clip * len(ix))))
            drop = ix[np.argsort(mar[ix])[k:]]   # our loss is ln(1+exp(-a.x)): large loss = SMALLEST margin a.x
            wk[drop] = 0.0
        xn = fit(B, wk, lam)
        if np.linalg.norm(xn - x) < 1e-8 * (1 + np.linalg.norm(x)):
            x = xn
            break
        x = xn
    return x


def damped_step(p, x, step, gr, c=0.1):
    """x - t*step with Armijo backtracking on the exact objective p.f (t = 1, 1/2, 1/4, ...), as in the Newton sketch of
    Pilanci & Wainwright (2017). A sketched Hessian can be far from H (m close to d), and the undamped step then diverges;
    when the full step already decreases f enough it is taken unchanged. The sketched Hessian is positive definite, so -step
    is a descent direction and the search ends. c = 0.1 and halving are the values Pilanci & Wainwright give (Sec. 2.1, footnote 1)."""
    t, f0, slope = 1.0, p.f(x), gr @ step
    while p.f(x - t * step) > f0 - c * t * slope and t > 1e-10:
        t *= 0.5
    return x - t * step


def run(p, meth, m, rng, T=10):
    """One draw of approach 1 (samp_*), 2 (newton_*) or 3 (logreg_sketch, naive_*). Returns the sketched coefficient vector."""
    T = getattr(p, "T", T)
    n, d, lam = p.n, p.d, p.lam
    if meth in ("samp_unif", "samp_imp"):
        q = np.full(n, 1 / n) if meth == "samp_unif" else p.q_imp
        idx = rng.choice(n, m, p=q)
        return fit(p.A[idx], 1.0 / (m * q[idx]), lam)
    if meth == "logreg_sketch":
        return fit_logreg_sketch_code(p.A, m, rng, lam)                 # authors' code version (lib_logreg_sketch.py)
    if meth == "logreg_sketch_ky":
        return fit_logreg_sketch_code(p.A, m, rng, lam, clip=0.25)
    if meth == "logreg_sketch_paper":
        return fit_logreg_sketch(p.A, m, rng, lam)                      # paper-text version, kept for comparison
    if meth.startswith("naive"):
        if meth == "naive_gauss":
            SA = rng.standard_normal((m, n)) @ p.A / np.sqrt(m)
        else:
            SA = np.zeros((m, d))
            np.add.at(SA, rng.integers(0, m, n), p.A * rng.choice([-1.0, 1.0], n)[:, None])
        return fit(SA, np.ones(m), lam)
    x = np.zeros(d)
    for _ in range(T):
        s = sig(-(p.A @ x))
        gr = -(p.A.T @ s) + lam * x
        B = p.A * np.sqrt(s * (1 - s))[:, None]
        if meth == "newton_gauss":
            SB = rng.standard_normal((m, n)) @ B / np.sqrt(m)
        else:
            SB = np.zeros((m, d))
            np.add.at(SB, rng.integers(0, m, n), B * rng.choice([-1.0, 1.0], n)[:, None])
        x = damped_step(p, x, np.linalg.solve(SB.T @ SB + lam * np.eye(d), gr), gr)
    return x


def experiment(p, m, reps, rng, meths=METH, runner=None):
    runner = runner or run
    rows = []
    for meth in meths:
        S = {k: [] for k in MET}
        LR = []
        for _ in range(reps):
            xt = runner(p, meth, m, rng)
            LR.append(p.f(xt) / p.fh)
            sg = p.signed(xt)
            for k in MET:
                S[k].append(sg[k])
        for k in MET:
            v = np.array(S[k])
            rows.append(dict(method=meth, metric=k, m=m, reps=reps, mean_rel=v.mean(), mean_abs_rel=np.abs(v).mean(),
                             q10_rel=np.quantile(v, .1), q90_rel=np.quantile(v, .9), frac_worse=(v > 0).mean(),
                             base=p.base[k], loss_ratio=float(np.median(LR))))
    return pd.DataFrame(rows)



PLOT_METH = [m for m in METH if m.startswith(("samp", "logreg"))]    # naive: table only; Newton: its own figure (vs steps)
NEWTON = ["newton_cs", "newton_gauss"]


def class_view(df, xcol, logx, meths, min_base=0.03, floor=1e-6):
    """What a classification figure plots. IF stays relative to the exact model's norm; the DP/TPR/FPR gaps are shown as
    ABSOLUTE changes |gap(x~)| - |gap(x^)| (= relative change * |base|), so small exact gaps no longer leave empty panels.
    A Newton-steps axis is trimmed after convergence, judged on relative change where the exact gap is at least min_base;
    sketch-size axes are never trimmed."""
    if xcol == "T":
        kept, _ = sd_layout(df, xcol, logx, "method", meths, MET, min_base, RATE_METRICS, floor, trim=True)
        df = df[df[xcol].isin(kept[xcol].unique())]
    df = df.copy()
    rate = df.metric.isin(RATE_METRICS)
    for c in ("mean_rel", "mean_abs_rel", "q10_rel", "q90_rel"):
        df.loc[rate, c] = df.loc[rate, c] * df.loc[rate, "base"].abs()
    return df


def draw(df, xcol, xlabel, path, meths=None, logx=True, min_base=0.03, floor=1e-6):
    """Size/direction figure for classification (one line per method)."""
    meths = meths or PLOT_METH
    series = [dict(key=k, label=LAB[k], color=COL[k]) for k in meths]
    plot_sd(class_view(df, xcol, logx, meths, min_base, floor), xcol, xlabel, path, series, "method", MET, logx=logx,
            floor=floor, tight_size=True, abs_metrics=RATE_METRICS)


def emit(out, name, df, spec, min_base=0.03):
    """Save a sweep's CSV and draw its figure. spec = dict(x, label, logx, meths)."""
    df.to_csv(f"{out}/{name}.csv", index=False)
    draw(df, spec["x"], spec["label"], f"{out}/{name}.png", meths=spec.get("meths"), logx=spec.get("logx", True), min_base=min_base)


def replot(out, specs, min_base=0.03):
    """Redraw figures from the CSVs already in `out` (--replot)."""
    for name, spec in specs.items():
        f = f"{out}/{name}.csv"
        if os.path.exists(f):
            draw(pd.read_csv(f), spec["x"], spec["label"], f"{out}/{name}.png", meths=spec.get("meths"), logx=spec.get("logx", True),
                 min_base=min_base)


def replot_datasets(out, prefix, spec_fn, min_base=0.01):
    """--replot for the real-data drivers: one figure set per dataset found as <prefix><name>_m.csv in `out`."""
    import glob
    for f in sorted(glob.glob(f"{out}/{prefix}*_m.csv")):
        name = os.path.basename(f)[len(prefix):-len("_m.csv")]
        replot(out, spec_fn(name), min_base)


def sweep(builder, xs, xname, reps, rng, m, meths, runner=None, T=None):
    """Run `experiment` for each x and stack the results with an extra column `xname`.
    builder(x) returns the problem P; xname == "m" sets the sketch size to x; xname == "T" sets the number of Newton steps to x
    (the problem's own T is restored afterwards); otherwise T (if given) is the number of Newton steps and m the sketch size."""
    frames = []
    for x in xs:
        p = builder(x)
        old = getattr(p, "T", None)
        if xname == "T":
            p.T = x
        elif T is not None:
            p.T = T
        frames.append(experiment(p, x if xname == "m" else m, reps, rng, meths=meths, runner=runner).assign(**{xname: x}))
        if xname == "T" and old is not None:
            p.T = old
        print(f"  {xname}={x}: done", flush=True)
    return pd.concat(frames)


def add_args(ap, out, m=500, T=10, reps=60, synth=False):
    ap.add_argument("--out", default=out)
    ap.add_argument("--reps", type=int, default=reps)
    ap.add_argument("--quick", action="store_true", help="tiny run for testing")
    ap.add_argument("--m", type=int, default=m, help="sketch size for tables and sweeps other than m")
    ap.add_argument("--T", type=int, default=T, help="Newton steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--replot", action="store_true", help="redraw figures from the CSVs already in --out")
    if synth:
        ap.add_argument("--n", type=int, default=10000)
        ap.add_argument("--d", type=int, default=10)
        ap.add_argument("--data_seed", type=int, default=0, help="synthetic data draw")
    return ap


def start(args):
    """Apply --quick, create the output folder, set the plot style, return the rng."""
    if args.quick:
        args.reps = min(args.reps, 10)
        if hasattr(args, "n"):
            args.n = min(args.n, 4000)
        if hasattr(args, "acs_n"):
            args.acs_n = min(args.acs_n, 8000)
    os.makedirs(args.out, exist_ok=True)
    set_style(args.latex)
    return np.random.default_rng(args.seed)


def table(df, out, stem, methods, caption):
    save_method_table(df, f"{out}/{stem}", MET, methods, LAB, caption, f"tab:{stem}")