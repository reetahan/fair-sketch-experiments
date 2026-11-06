"""Ridge logistic regression, its sketches (Section 4) and one-shot targeted coresets (Section 5)."""
import argparse, os
import numpy as np, pandas as pd
import scipy.optimize as so
from fs_common import set_style, plot_sd, sd_layout, save_method_table, PALETTE

sig = lambda t: 0.5 * (1 + np.tanh(0.5 * t))
METH = ["samp_unif", "samp_lev", "newton_cs", "newton_gauss", "logreg_sketch", "naive_gauss", "naive_cs"]
LAB = {"samp_unif": "1: sampled, uniform", "samp_lev": "1: sampled, leverage",
       "samp_imp": "1: sampled, importance (oracle)", "newton_cs": "2: Newton-Hessian, CountSketch",
       "newton_gauss": "2: Newton-Hessian, Gaussian", "logreg_sketch": "3: LogReg-sketch (Munteanu et al. 2021)",
       "naive_gauss": "3: naive Gaussian", "naive_cs": "3: naive CountSketch"}
COL = {"samp_unif": PALETTE["unif"], "samp_lev": PALETTE["lev"], "samp_imp": PALETTE["is"], "newton_cs": PALETTE["cs"],
       "newton_gauss": PALETTE["gauss"], "logreg_sketch": PALETTE["logreg"], "naive_gauss": "tab:gray",
       "naive_cs": "0.6"}
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
    """Ridge logistic regression by damped Newton; rows of A are label-folded."""
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
        self.grad_i = self.A * s[:, None]
        self.q_imp = np.linalg.norm(self.grad_i @ L, axis=1)
        self.q_imp = 0.5 * self.q_imp / self.q_imp.sum() + 0.5 / self.n
        lev = (Z * np.linalg.solve(Z.T @ Z + lam * np.eye(self.d), Z.T).T).sum(1)
        self.q_lev = lev / lev.sum()
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


def damped_step(p, x, step, gr, c=0.1):
    """Armijo backtracking on the exact objective."""
    t, f0, slope = 1.0, p.f(x), gr @ step
    while p.f(x - t * step) > f0 - c * t * slope and t > 1e-10:
        t *= 0.5
    return x - t * step


def run(p, meth, m, rng, T=10):
    T = getattr(p, "T", T)
    n, d, lam = p.n, p.d, p.lam
    if meth in ("samp_unif", "samp_lev", "samp_imp"):
        q = {"samp_unif": np.full(n, 1 / n), "samp_lev": p.q_lev, "samp_imp": p.q_imp}[meth]
        idx = rng.choice(n, m, p=q)
        return fit(p.A[idx], 1.0 / (m * q[idx]), lam)
    if meth == "logreg_sketch":
        return fit_logreg_sketch_code(p.A, m, rng, lam)
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
            rows.append(
                dict(method=meth, metric=k, m=m, reps=reps, mean_rel=v.mean(), mean_abs_rel=np.abs(v).mean(),
                     q10_rel=np.quantile(v, 0.1), q90_rel=np.quantile(v, 0.9), frac_worse=(v > 0).mean(),
                     base=p.base[k], loss_ratio=float(np.median(LR)))
            )
    return pd.DataFrame(rows)


PLOT_METH = [m for m in METH if m.startswith(("samp", "logreg"))]
NEWTON = ["newton_cs", "newton_gauss"]


def class_view(df, xcol, logx, meths, min_base=0.03, floor=1e-6):
    if xcol == "T":
        kept, _ = sd_layout(df, xcol, logx, "method", meths, MET, min_base, RATE_METRICS, floor, trim=True)
        df = df[df[xcol].isin(kept[xcol].unique())]
    df = df.copy()
    rate = df.metric.isin(RATE_METRICS)
    for c in ("mean_rel", "mean_abs_rel", "q10_rel", "q90_rel"):
        df.loc[rate, c] = df.loc[rate, c] * df.loc[rate, "base"].abs()
    return df


def draw(df, xcol, xlabel, path, meths=None, logx=True, min_base=0.03, floor=1e-6):
    meths = meths or PLOT_METH
    series = [dict(key=k, label=LAB[k], color=COL[k]) for k in meths]
    plot_sd(class_view(df, xcol, logx, meths, min_base, floor), xcol, xlabel, path, series, "method", MET, logx=logx,
            floor=floor, tight_size=True, abs_metrics=RATE_METRICS)


def emit(out, name, df, spec, min_base=0.03):
    df.to_csv(f"{out}/{name}.csv", index=False)
    draw(df, spec["x"], spec["label"], f"{out}/{name}.png", meths=spec.get("meths"), logx=spec.get("logx", True),
         min_base=min_base)


def replot(out, specs, min_base=0.03):
    for name, spec in specs.items():
        f = f"{out}/{name}.csv"
        if os.path.exists(f):
            draw(pd.read_csv(f), spec["x"], spec["label"], f"{out}/{name}.png", meths=spec.get("meths"),
                 logx=spec.get("logx", True), min_base=min_base)


def replot_datasets(out, prefix, spec_fn, min_base=0.01):
    import glob

    for f in sorted(glob.glob(f"{out}/{prefix}*_m.csv")):
        name = os.path.basename(f)[len(prefix) : -len("_m.csv")]
        replot(out, spec_fn(name), min_base)


def sweep(builder, xs, xname, reps, rng, m, meths, runner=None, T=None):
    frames = []
    for x in xs:
        p = builder(x)
        old = getattr(p, "T", None)
        if xname == "T":
            p.T = x
        elif T is not None:
            p.T = T
        frames.append(
            experiment(p, x if xname == "m" else m, reps, rng, meths=meths, runner=runner).assign(**{xname: x})
        )
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


def ext_args(out):
    from lib_data import add_real_args

    ap = add_real_args(add_args(argparse.ArgumentParser(), out, m=300, T=3, synth=True), 50000)
    ap.add_argument("--synthetic", action="store_true")
    ap.set_defaults(data_seed=3)
    return ap


def ext_problems(a):
    from lib_data import real_problems

    return [P(*make_data(a.n, a.d, frac1=0.1, seed=a.data_seed))] if a.synthetic else real_problems(a, "classification")


def objective(K, q):
    """Variance objective V(q) = sum_i ||kappa_i||^2 / q_i - ||sum_i kappa_i||^2 of a sampling rule q."""
    return float(np.sum(np.sum(K**2, 1) / q) - np.sum(np.sum(K, 0) ** 2))


def start(args):
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


def logreg_sketch_code(A, m, rng, h_max=2):
    n, d = A.shape
    N = max(int(m / (h_max + 2)), 1)
    b = (n / N) ** (1.0 / h_max)
    pl = b ** np.arange(h_max + 1, dtype=float)
    alpha = pl.sum()
    pl /= alpha
    lev = rng.choice(h_max + 1, n, p=pl)
    SA = np.zeros((N * (h_max + 1), d))
    np.add.at(SA, lev * N + rng.integers(0, N, n), A)
    w_sk = np.repeat(1.0 / pl, N)
    k = max(m - N * (h_max + 1), 1)
    U = rng.choice(n, k, replace=False)
    rows = np.vstack([SA, A[U]])
    w = np.concatenate([w_sk, np.full(k, n / k)])
    return rows, w * (n / w.sum()), N


def _keep(t, N, K, L):
    blocks = t[:L].reshape(-1, N)
    top = np.argpartition(blocks, K - 1, axis=1)[:, :K] + (np.arange(blocks.shape[0]) * N)[:, None]
    return np.concatenate([top.ravel(), np.arange(L, len(t))])


def fit_logreg_sketch_code(A, m, rng, lam, clip=None, h_max=2):
    rows, w, N = logreg_sketch_code(A, m, rng, h_max)
    L = N * (h_max + 1)
    K = max(int(N * clip), 1) if clip else None

    def f(x):
        t = rows @ x
        I = _keep(t, N, K, L) if (clip and K < N) else slice(None)
        tI, wI = t[I], w[I]
        return wI @ np.logaddexp(0, -tI) + 0.5 * lam * x @ x, -(rows[I].T @ (wI * sig(-tI))) + lam * x

    return so.minimize(f, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B").x


GAP_FLOOR = 0.01
for k, (lab, col) in {"c_unif": ("uniform", PALETTE["unif"]),
                      "c_strat_unif": ("stratified uniform", PALETTE["strat_unif"]),
                      "c_lev": ("leverage", PALETTE["lev"]), "c_imp": ("importance, Sec. 3 (oracle)", PALETTE["is"]),
                      "c_imp_proxy": ("importance, Sec. 3 (proxy)", "0.6"),
                      "c_tgt_oracle": ("targeted (oracle)", PALETTE["oracle"]),
                      "c_tgt_proxy": ("targeted (proxy)", PALETTE["proxy"]),
                      "c_strat_tgt_oracle": ("stratified targeted (oracle)", "indigo"),
                      "c_strat_tgt_proxy": ("stratified targeted (proxy)", PALETTE["strat_proxy"]),
                      "logreg_code": ("LogReg-sketch (Munteanu et al. 2021)", PALETTE["logreg"]),
                      "logreg_code_ky": ("LogReg-sketch, clipped", "darkolivegreen")}.items():
    LAB[k], COL[k] = lab, col
S5C_METH = ["c_unif", "c_lev", "c_tgt_oracle", "c_tgt_proxy", "c_strat_tgt_proxy", "logreg_code", "newton_cs"]
S5C_ALL = S5C_METH + ["c_imp", "c_imp_proxy", "c_strat_unif", "c_strat_tgt_oracle", "logreg_code_ky", "newton_gauss"]


def directions(p, x, target="all"):
    pr = sig(p.Z @ x)
    dd = pr * (1 - pr)
    cols = []
    for k in ["IF", "DP", "TPR", "FPR"] if target == "all" else [target]:
        if k == "IF":
            cols.append(x / (x @ x))
            continue
        r0, r1 = p.sets[k]
        u = (dd[r0] @ p.Z[r0]) / r0.sum() - (dd[r1] @ p.Z[r1]) / r1.sum()
        if target == "all":
            u = u / max(abs(pr[r0].mean() - pr[r1].mean()), GAP_FLOOR)
        cols.append(u)
    return np.stack(cols, 1)


def scores(p, x, H, target="all"):
    """kappa_i scores (Section 5)."""
    K = ((p.y > 0) - sig(p.Z @ x))[:, None] * (p.Z @ np.linalg.solve(H, directions(p, x, target)))
    return np.linalg.norm(K, axis=1), K


def _mix(a, mu):
    return (1 - mu) * a / a.sum() + mu / len(a)


def _draw(q, m, rng):
    Kc = rng.multinomial(m, q)
    I = np.nonzero(Kc)[0]
    return I, Kc[I] / (m * q[I])


def _pilot(p, m_p, rng):
    I, w = _draw(np.full(p.n, 1.0 / p.n), m_p, rng)
    xp = fit(p.A[I], w, p.lam)
    s = sig(p.A[I] @ xp)
    Hh = (p.Z[I] * (w * s * (1 - s))[:, None]).T @ p.Z[I] + p.lam * np.eye(p.d)
    return xp, Hh, I, w


def imp_probs(p, x, H):
    """Section 3 importance probabilities at (x, H); p.q_imp is this at the exact solution."""
    s = sig(-(p.A @ x))
    L = np.linalg.cholesky(np.linalg.inv(H))
    a = np.linalg.norm((p.A * s[:, None]) @ L, axis=1)
    return 0.5 * a / a.sum() + 0.5 / p.n


def _strat(p, a, K, m, mu, rng):
    groups = [np.nonzero(p.g == j)[0] for j in (0, 1)]
    qs, beta = [], []
    for idx in groups:
        q = _mix(a[idx] + 1e-300, mu)
        qs.append(q)
        beta.append(max(np.sum(np.sum(K[idx] ** 2, 1) / q) - np.sum(np.sum(K[idx], 0) ** 2), 0.0))
    sb = np.sqrt(beta)
    if sb.sum() > 0:
        m0 = round(m * sb[0] / sb.sum())
    else:
        m0 = round(m * len(groups[0]) / p.n)
    m0 = int(np.clip(m0, 2, m - 2))
    I, W = [], []
    for idx, q, mg in zip(groups, qs, (m0, m - m0)):
        Ig, wg = _draw(q, mg, rng)
        I.append(idx[Ig])
        W.append(wg)
    return np.concatenate(I), np.concatenate(W)


def run_coreset(p, meth, m, rng, T=None):
    if meth.startswith("newton"):
        return run(p, meth, m, rng) if T is None else run(p, meth, m, rng, T=T)
    if meth in ("logreg_code", "logreg_code_ky"):
        return fit_logreg_sketch_code(p.A, m, rng, p.lam, clip=0.25 if meth.endswith("ky") else None)
    target, mu = getattr(p, "target", "all"), getattr(p, "mu", 0.5 if meth.endswith("proxy") else 0.2)
    if meth == "c_unif":
        I, w = _draw(np.full(p.n, 1.0 / p.n), m, rng)
    elif meth == "c_lev":
        I, w = _draw(p.q_lev, m, rng)
    elif meth == "c_imp":
        I, w = _draw(p.q_imp, m, rng)
    elif meth == "c_strat_unif":
        a, K = scores(p, p.xh, p.H, target)
        I, w = _strat(p, np.ones(p.n), K, m, 1.0, rng)
    else:
        proxy = meth.endswith("proxy")
        mm = m
        if proxy:
            m_p = getattr(p, "m_p", None) or m // 2
            mm = m - m_p
            xp, Hh, Ip, wp = _pilot(p, m_p, rng)
            if meth != "c_imp_proxy":
                a, K = scores(p, xp, Hh, target)
        else:
            a, K = scores(p, p.xh, p.H, target)
        if meth == "c_imp_proxy":
            I, w = _draw(imp_probs(p, xp, Hh), mm, rng)
        elif meth.startswith("c_strat"):
            I, w = _strat(p, a, K, mm, mu, rng)
        else:
            I, w = _draw(_mix(a, mu), mm, rng)
        if proxy and getattr(p, "reuse_pilot", True):
            f = m_p / m
            I, w = np.concatenate([Ip, I]), np.concatenate([f * wp, (1 - f) * w])
    return fit(p.A[I], w, p.lam)
