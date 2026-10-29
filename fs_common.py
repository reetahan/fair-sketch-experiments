
import io, os, time, urllib.request, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

USETEX = False


def set_style(latex=False):
    """SciencePlots style (pip install SciencePlots). latex=True needs a working LaTeX install."""
    global USETEX
    try:
        import scienceplots  # noqa: F401  (registers the styles)
        plt.style.use(["science"] if latex else ["science", "no-latex"])
        USETEX = bool(latex)
    except ImportError:
        print("SciencePlots not installed (pip install SciencePlots); using the matplotlib default style")
        USETEX = False


def _plain(v, _pos=None):
    """Plain decimal tick label, never scientific notation."""
    if v == 0:
        return "0"
    return np.format_float_positional(float(v), trim="-", precision=6)


def style_axes(axes, xticks=None):
    """Gridlines on, tick labels in plain decimals (no scientific notation). If xticks is given (e.g. the swept values)
    they become the x ticks."""
    from matplotlib.ticker import FuncFormatter, NullFormatter, NullLocator, LogLocator, ScalarFormatter
    for ax in np.ravel(axes):
        ax.grid(True, which="major", alpha=0.35, lw=0.6)
        ax.set_axisbelow(True)
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale in ("log", "symlog"):
                axis.set_major_formatter(FuncFormatter(_plain))
                if scale == "log":
                    axis.set_minor_locator(LogLocator(base=10, subs=(2.0, 5.0)))
                    axis.set_minor_formatter(FuncFormatter(_plain))
                else:
                    axis.set_minor_formatter(NullFormatter())
            else:
                sf = ScalarFormatter(useOffset=False)
                sf.set_scientific(False)
                axis.set_major_formatter(sf)
        if xticks is not None:
            ax.set_xticks(list(xticks))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(_plain))
        if ax.get_yscale() == "log":
            ax.grid(True, which="minor", axis="y", alpha=0.15, lw=0.4)


def _xt(df, xcol):
    v = np.sort(np.unique(df[xcol].values))
    return v if len(v) <= 8 else None


def tx(s):
    """Escape characters that break LaTeX text when usetex is on."""
    if not USETEX:
        return s
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        s = s.replace(a, b)
    return s

FAMS = ["gauss", "cs", "unif", "is"]
SFAMS = ["gauss", "cs", "unif"]         
METRICS = ["IF", "GFE", "MSE"]
FAM_LABEL = {"gauss": "Gaussian", "cs": "CountSketch", "unif": "Uniform", "is": "Importance"}
COLOR = {"gauss": "tab:blue", "cs": "tab:orange", "unif": "tab:green", "is": "tab:red"}
LAW_URL = "https://raw.githubusercontent.com/damtharvey/law-school-dataset/main/law_dataset.csv"


class Problem:

    def __init__(self, A, b, g, name=""):
        self.name = name
        self.A, self.b, self.g = A, b, np.asarray(g).astype(int)
        self.n, self.d = A.shape
        self.k = self.d + 1
        self.idx = [np.where(self.g == 0)[0], np.where(self.g == 1)[0]]
        self.n0, self.n1 = len(self.idx[0]), len(self.idx[1])
        self.ng = [self.n0, self.n1]
        self.Q, self.R = np.linalg.qr(np.column_stack([A, b]))     # [A b] = Q R
        self.Ra, self.rb = self.R[:, :self.d], self.R[:, self.d]
        self.G = np.linalg.inv(A.T @ A)                             # (A^T A)^{-1}
        self.xhat = self.G @ (A.T @ b)
        self.r = b - A @ self.xhat
        sv = np.linalg.svd(A, compute_uv=False)
        self.sig1, self.sigd = sv[0], sv[-1]
        self.cond = self.sig1 / self.sigd
        self.P = A @ self.G                                         # rows p_i = (A^T A)^{-1} a_i ; also Pi
        A0, A1 = A[self.idx[0]], A[self.idx[1]]
        self.v = A0.mean(0) - A1.mean(0)
        self.w = A0.T @ self.r[self.idx[0]]
        self.gram = [A0.T @ A0, A1.T @ A1]
        self.Atb = [A0.T @ b[self.idx[0]], A1.T @ b[self.idx[1]]]
        self.btb = [b[self.idx[0]] @ b[self.idx[0]], b[self.idx[1]] @ b[self.idx[1]]]
        self.rg = [self.r[self.idx[0]], self.r[self.idx[1]]]
        # targets C for the three metrics (C^T E r is the linearised quantity)
        self.C = {"IF": self.P, "GFE": (self.P @ self.v)[:, None], "MSE": (self.P @ self.w)[:, None]}
        # importance-sampling probabilities (oracle: use the exact residuals), mixed with leverage for safety
        lev = (self.A * self.P).sum(1)
        qopt = np.abs(self.r) * np.linalg.norm(self.P, axis=1)
        self.q_is = 0.5 * qopt / qopt.sum() + 0.5 * lev / lev.sum()
        self.cdf_is = np.cumsum(self.q_is)
        self.cdf_is /= self.cdf_is[-1]

    def mse_g(self, x, g):
        return (self.btb[g] - 2 * x @ self.Atb[g] + x @ self.gram[g] @ x) / self.ng[g]

    def gap(self, x):
        return self.mse_g(x, 0) - self.mse_g(x, 1)

    def ols_values(self):
        return {"IF": np.linalg.norm(self.xhat), "GFE": abs(self.v @ self.xhat), "MSE": abs(self.gap(self.xhat))}

    def solve(self, SQ):
        """Sketched solution from SQ = S Q (m x k):  S[A b] = SQ R."""
        xt, *_ = np.linalg.lstsq(SQ @ self.Ra, SQ @ self.rb, rcond=None)
        return xt

    def solve_checked(self, SQ, tol=1e-6):
        ok = bool(np.linalg.eigvalsh(SQ.T @ SQ)[0] > tol)
        return self.solve(SQ), ok

    def eps(self, SQ):
        """Empirical embedding parameter of the sketch for col([A b])."""
        return float(np.abs(np.linalg.eigvalsh(SQ.T @ SQ - np.eye(self.k))).max())

    def signed(self, xt):
        """Signed changes vs OLS (positive = less fair): IF, GFE, MSE parity."""
        xh = self.xhat
        return np.array([np.linalg.norm(xt) / np.linalg.norm(xh) - 1.0,
                         abs(self.v @ xt) - abs(self.v @ xh),
                         abs(self.gap(xt)) - abs(self.gap(xh))])

    def distortions(self, xt):
        """(IF, GFE, MSE) distortions, and the squared target quantities (c^T Delta)^2 used by the theory."""
        xh = self.xhat
        dist = np.array([
            abs(np.linalg.norm(xt) / np.linalg.norm(xh) - 1.0),
            abs(abs(self.v @ xt) - abs(self.v @ xh)),
            abs(abs(self.gap(xt)) - abs(self.gap(xh))),
        ])
        D = xt - xh
        J = np.array([D @ D, (self.v @ D) ** 2, (self.w @ D) ** 2])
        return dist, J


def sketch_Q(Q, fam, m, rng, cdf=None, q=None):
    n, k = Q.shape
    if fam == "gauss":
        return rng.standard_normal((m, k)) / np.sqrt(m)
    if fam == "cs":
        h = rng.integers(0, m, size=n)
        s = rng.integers(0, 2, size=n) * 2.0 - 1.0
        out = np.empty((m, k))
        for j in range(k):
            out[:, j] = np.bincount(h, weights=s * Q[:, j], minlength=m)
        return out
    if fam == "unif":
        idx = rng.integers(0, n, size=m)
        return Q[idx] * np.sqrt(n / m)
    if fam == "is":
        idx = np.searchsorted(cdf, rng.random(m))
        idx = np.minimum(idx, n - 1)
        return Q[idx] / np.sqrt(m * q[idx])[:, None]
    raise ValueError(fam)


def vanilla_SQ(prob, fam, m, rng):
    return sketch_Q(prob.Q, fam, m, rng, cdf=prob.cdf_is, q=prob.q_is)


def strat_SQ(prob, fam, m0, m1, rng):
    """Block-diagonal stratified sketch applied to Q: independent sketch of each group's rows."""
    blocks = []
    for g, mg in ((0, m0), (1, m1)):
        Qg = prob.Q[prob.idx[g]]
        if fam == "gauss":
            Sg = Qg.T @ Qg
            lam, V = np.linalg.eigh(Sg)
            Wg = V * np.sqrt(np.clip(lam, 0, None))
            blocks.append(rng.standard_normal((mg, prob.k)) / np.sqrt(mg) @ Wg.T)
        else:
            blocks.append(sketch_Q(Qg, fam, mg, rng))
    return np.vstack(blocks)


def sigma_theory(prob, fam, m):
    r, P, G = prob.r, prob.P, prob.G
    rr = r @ r
    if fam == "gauss":
        return rr * G / m
    if fam == "cs":
        return (rr * G - 2 * (P * (r ** 2)[:, None]).T @ P) / m
    if fam == "unif":
        return prob.n / m * (P * (r ** 2)[:, None]).T @ P
    if fam == "is":
        return (P * (r ** 2 / prob.q_is)[:, None]).T @ P / m
    raise ValueError(fam)


def bounds(prob, fam, m, eps, eta=0.1):
    """Dict metric -> dict(pred, prop1_sqrt, prop1_tight, thm1) for embedding parameter eps."""
    S = sigma_theory(prob, fam, m)
    T, V, W = np.trace(S), prob.v @ S @ prob.v, prob.w @ S @ prob.w
    xh = prob.xhat
    pred = {"IF": np.sqrt(max(xh @ S @ xh, 0)) / np.linalg.norm(xh) ** 2,   # first order: x^T Delta / ||x||^2
            "GFE": np.sqrt(max(V, 0)),
            "MSE": 2 * (1 / prob.n0 + 1 / prob.n1) * np.sqrt(max(W, 0))}
    out = {met: dict(pred=pred[met]) for met in METRICS}
    r0, r1 = np.linalg.norm(prob.rg[0]), np.linalg.norm(prob.rg[1])
    rn = np.linalg.norm(prob.r)
    n0, n1 = prob.n0, prob.n1

    def prop1(gam):
        return {"IF": gam / (prob.sigd * np.linalg.norm(xh)),
                "GFE": np.linalg.norm(prob.v) * gam / prob.sigd,
                "MSE": 2 * (r0 / n0 + r1 / n1) * gam + (1 / n0 + 1 / n1) * gam ** 2}

    if eps < 1:
        p_tight = prop1(eps / (1 - eps) * rn)
    else:
        p_tight = {met: np.inf for met in METRICS}
    p_sqrt = prop1(2 * np.sqrt(eps) * rn) if eps <= 0.5 else {met: np.inf for met in METRICS}
    if eps < 1:
        c = prob.cond * eps / (1 - eps)
        t_if = (1 + c) / np.linalg.norm(xh) * np.sqrt(T / eta)
        t_gfe = np.sqrt(2 * V / eta) + c * np.linalg.norm(prob.v) * np.sqrt(2 * T / eta)
        Qg = [np.trace(prob.gram[g] @ S) for g in (0, 1)]
        t_mse = (2 * (1 / n0 + 1 / n1) * (np.sqrt(4 * W / eta) + c * np.linalg.norm(prob.w) * np.sqrt(4 * T / eta))
                 + 8 / eta * sum((Qg[g] + prob.sig1 ** 2 * c ** 2 * T) / prob.ng[g] for g in (0, 1)))
        thm = {"IF": t_if, "GFE": t_gfe, "MSE": t_mse}
        a = eps / (1 - eps)
        Z = np.trace(prob.gram[0] @ S) + np.trace(prob.gram[1] @ S)       
        vGv, wGw = prob.v @ prob.G @ prob.v, prob.w @ prob.G @ prob.w
        k_if = (np.sqrt(T / eta) + a * np.sqrt(Z / eta) / prob.sigd) / np.linalg.norm(xh)
        k_gfe = np.sqrt(2 * V / eta) + a * np.sqrt(vGv) * np.sqrt(2 * Z / eta)
        k_mse = (2 * (1 / n0 + 1 / n1) * (np.sqrt(4 * W / eta) + a * np.sqrt(wGw) * np.sqrt(4 * Z / eta))
                 + sum((8 * Qg[g] / eta + 8 * a * a * Z / eta) / prob.ng[g] for g in (0, 1)))
        thm_k = {"IF": k_if, "GFE": k_gfe, "MSE": k_mse}
    else:
        thm = {met: np.inf for met in METRICS}
        thm_k = {met: np.inf for met in METRICS}
    for met in METRICS:
        out[met].update(prop1_sqrt=p_sqrt[met], prop1_tight=p_tight[met], thm1=thm[met], thm1_kfree=thm_k[met])
    return out


# ----------------------------------------------------------------------------------------------
# Experiment 1: vanilla sketches vs bounds
# ----------------------------------------------------------------------------------------------
def vanilla_table(prob, m, reps, rng, delta=0.1, eta=0.1, fams=FAMS):
    rows = []
    ols = prob.ols_values()
    for fam in fams:
        D = np.zeros((reps, 3))
        JJ = np.zeros((reps, 3))
        E = np.zeros(reps)
        OK = np.zeros(reps, bool)
        for t in range(reps):
            SQ = vanilla_SQ(prob, fam, m, rng)
            xt, OK[t] = prob.solve_checked(SQ)
            D[t], JJ[t] = prob.distortions(xt)
            E[t] = prob.eps(SQ)
        eps_q = float(np.quantile(E, 1 - delta))          # over all runs, failed ones included
        frac_failed = 1.0 - OK.mean()
        if OK.sum() < 2:
            D, JJ = np.full((2, 3), np.nan), np.full((2, 3), np.nan)
        else:
            D, JJ = D[OK], JJ[OK]
        B = bounds(prob, fam, m, eps_q, eta)
        for j, met in enumerate(METRICS):
            rows.append(dict(dataset=prob.name, m=m, family=fam, metric=met, ols_value=ols[met],
                             emp_mean=D[:, j].mean(), emp_rms=np.sqrt((D[:, j] ** 2).mean()),
                             emp_q90=np.quantile(D[:, j], 0.9), pred_rms=B[met]["pred"],
                             prop1_sqrt=B[met]["prop1_sqrt"], prop1_tight=B[met]["prop1_tight"],
                             thm1=B[met]["thm1"], thm1_kfree=B[met]["thm1_kfree"], eps_q=eps_q, reps=reps, frac_failed=frac_failed,
                             # sanity check of Lemma 2/4: E (c^T Delta)^2 should be close to V_F / m (first order)
                             J_emp=JJ[:, j].mean(), J_theory=vanilla_value(prob, prob.C[met], fam) / m))
    return pd.DataFrame(rows)


def coefs(prob, C, y, fam):
    """Group coefficients beta_g (Definition 2) and pieces, for target C and vector y (=r or pilot residual)."""
    out = []
    for g in (0, 1):
        ii = prob.idx[g]
        Cg, yg = C[ii], y[ii]
        rho = yg @ yg
        tau = (Cg ** 2).sum()
        zeta = ((Cg.T @ yg) ** 2).sum()
        s = ((Cg ** 2).sum(1) * yg ** 2).sum()
        if fam == "gauss":
            beta = rho * tau + zeta
        elif fam == "cs":
            beta = rho * tau + zeta - 2 * s
        else:
            beta = prob.ng[g] * s - zeta
        out.append(dict(beta=max(beta, 0.0), rho=rho, tau=tau, zeta=zeta, s=s))
    return out


def vanilla_value(prob, C, fam):
    """V_F: E||C^T E r||^2 = V_F / m for the unsplit sketch (true residual)."""
    rho = prob.r @ prob.r
    tau = (C ** 2).sum()
    s = ((C ** 2).sum(1) * prob.r ** 2).sum()
    if fam == "gauss":
        return rho * tau
    if fam == "cs":
        return rho * tau - 2 * s
    if fam == "unif":
        return prob.n * s
    if fam == "is":
        return ((C ** 2).sum(1) * prob.r ** 2 / prob.q_is).sum()
    raise ValueError(fam)


def alloc(b0, b1, m, n0, n1):
    """Algorithm 1 lines 7-12."""
    if b0 <= 0 and b1 <= 0:
        mh = float(np.clip(m * n0 / (n0 + n1), 1, m - 1))
        return int(round(mh)), m - int(round(mh))
    t0 = np.sqrt(b0) / (np.sqrt(b0) + np.sqrt(b1))
    mh = float(np.clip(m * t0, 1, m - 1))
    cands = {int(np.clip(np.floor(mh), 1, m - 1)), int(np.clip(np.ceil(mh), 1, m - 1))}
    best = min(cands, key=lambda mu: b0 / mu + b1 / (m - mu))
    return best, m - best


def target_C(prob, metric, rt=None):
    """Target matrix; with rt (pilot residual) this is the plug-in C-hat of Algorithm 2."""
    if metric == "IF":
        return prob.P
    if metric == "GFE":
        return (prob.P @ prob.v)[:, None]
    w = prob.w if rt is None else prob.A[prob.idx[0]].T @ rt[prob.idx[0]]
    return (prob.P @ w)[:, None]


def strat_experiment(prob, m, m_pilot, reps, rng, fams=SFAMS, pilot_fam="gauss", metrics=METRICS):
    """OLS -> vanilla -> oracle -> proxy. One row per method."""
    rows = []
    for metric in metrics:
        mi = METRICS.index(metric)
        C = prob.C[metric]
        sbase = {"IF": 1.0, "GFE": prob.ols_values()["GFE"], "MSE": prob.ols_values()["MSE"]}[metric]
        for fam in fams + ["is"]:
            vanilla_only = fam == "is"
            Jvan = vanilla_value(prob, C, fam) / m
            acc = {}
            fails = {}

            def add(method, dist, Jhat, Jan, extra=None, ok=True):
                if not ok:
                    fails[method] = fails.get(method, 0) + 1
                    return
                acc.setdefault(method, []).append((dist, Jhat, Jan, extra or {}))

            if not vanilla_only:
                bt = coefs(prob, C, prob.r, fam)
                m0o, m1o = alloc(bt[0]["beta"], bt[1]["beta"], m, prob.n0, prob.n1)
                Jor = bt[0]["beta"] / m0o + bt[1]["beta"] / m1o
            for t in range(reps):
                SQ = vanilla_SQ(prob, fam, m, rng)
                xt, ok = prob.solve_checked(SQ)
                d, J = prob.distortions(xt)
                add("vanilla", d[mi], J[mi], Jvan, dict(sgn=prob.signed(xt)[mi] / sbase), ok=ok)
                if vanilla_only:
                    continue
                SQ = strat_SQ(prob, fam, m0o, m1o, rng)
                xt, ok = prob.solve_checked(SQ)
                d, J = prob.distortions(xt)
                add("oracle", d[mi], J[mi], Jor, dict(sgn=prob.signed(xt)[mi] / sbase), ok=ok)
                # pilot (vanilla sketch of family pilot_fam), then plug-in allocation
                SQp = vanilla_SQ(prob, pilot_fam, m_pilot, rng)
                xp, okp = prob.solve_checked(SQp)
                if not okp:
                    fails["proxy"] = fails.get("proxy", 0) + 1
                    continue
                rt = prob.b - prob.A @ xp
                Ch = target_C(prob, metric, rt) if metric == "MSE" else C
                bh = coefs(prob, Ch, rt, fam)
                kap = max(abs(np.sqrt(bh[g]["beta"]) - np.sqrt(bt[g]["beta"])) / np.sqrt(bt[g]["beta"])
                          for g in (0, 1))
                for method, mm in (("proxy", m),):
                    a0, a1 = alloc(bh[0]["beta"], bh[1]["beta"], mm, prob.n0, prob.n1)
                    SQ = strat_SQ(prob, fam, a0, a1, rng)
                    xt, ok = prob.solve_checked(SQ)
                    d, J = prob.distortions(xt)
                    Jan = bt[0]["beta"] / a0 + bt[1]["beta"] / a1       # true objective at the chosen allocation
                    add(method, d[mi], J[mi], Jan, dict(kappa=kap, share0=a0 / mm, sgn=prob.signed(xt)[mi] / sbase), ok=ok)
            Jvan_ref = Jvan
            for method, lst in acc.items():
                if len(lst) < 2:
                    continue
                dist = np.array([x[0] for x in lst])
                Jhat = np.array([x[1] for x in lst])
                sg = np.array([x[3]["sgn"] for x in lst])
                Jan = float(np.mean([x[2] for x in lst]))
                row = dict(dataset=prob.name, m=m, m_pilot=m_pilot, metric=metric, family=fam, method=method,
                           J_analytic=Jan, ols_value=prob.ols_values()[metric],
                           J_over_vanilla=Jan / Jvan_ref,
                           Jhat_emp=Jhat.mean(), emp_mean=dist.mean(), emp_rms=np.sqrt((dist ** 2).mean()),
                           reps=reps, frac_failed=fails.get(method, 0) / reps,
                           mean_rel=sg.mean(), mean_abs_rel=np.abs(sg).mean(), q10_rel=np.quantile(sg, 0.1),
                           q90_rel=np.quantile(sg, 0.9), frac_worse=(sg > 0).mean())
                if not vanilla_only:
                    row["R_oracle_gain"] = Jvan / Jor        # R = V/O of Theorem 6 (>1: oracle helps)
                    row["J_over_oracle"] = Jan / Jor if Jor > 0 else np.nan
                    row["oracle_share0"] = m0o / m
                    row["true_share0_sqrtbeta"] = np.sqrt(bt[0]["beta"]) / (np.sqrt(bt[0]["beta"]) + np.sqrt(bt[1]["beta"]))
                if method.startswith("proxy"):
                    kk = np.array([x[3]["kappa"] for x in lst])
                    row["kappa_hat"] = kk.mean()
                    row["kappa_hat_max"] = kk.max()
                    ratio = np.array([x[2] for x in lst]) / Jor
                    row["Jprox_over_Jor_mean"] = ratio.mean()
                    ok = kk < 1
                    # Theorem 5(4) bound t*(1-t*) 4k^2/(1-k^2) <= k^2/(1-k^2), evaluated at the realised kappa
                    row["thm5_bound_mean"] = float(np.mean((kk[ok] ** 2) / (1 - kk[ok] ** 2))) if ok.any() else np.inf
                    row["share_kappa_lt1"] = ok.mean()
                rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------------
def make_synth(n=20000, d=10, frac1=0.2, cond=10.0, gap=1.0, gap_dir="top", noise0=1.0, noise1=1.0,
               mis=0.5, lev=0.0, lev1=0.0, spikes=0, spike_out=30.0, spike_val=1.0, seed=0, name="synthetic"):
    """Group 1 is the minority (fraction frac1). cond ~ condition number of A. gap = ||v||.
    mis > 0 gives group 1 a different true coefficient vector (so w = A_0^T r_0 != 0).
    spikes = k > 0 builds a COHERENT design: the last coordinate is carried only by k 'spike' rows of the minority group (value spike_val,
    zero on every other coordinate), whose labels carry outliers of size spike_out times the bulk noise std, with alternating signs (zero mean, so the
    fitted weak-direction coefficient is unchanged and each spike residual is about spike_out, not random). The spike rows then have leverage
    about 1/k and carry a large share of the residual, the case where CountSketch is provably less noisy than Gaussian (Lemma 2). cond is then not controllable."""
    rng = np.random.default_rng(seed)
    g = np.zeros(n, int)
    g[rng.choice(n, int(round(frac1 * n)), replace=False)] = 1
    scales = np.logspace(0, -np.log10(cond), d)
    X = rng.standard_normal((n, d)) * scales
    shift = np.zeros(d)
    shift[0 if gap_dir == "top" else d - 1] = gap
    X = X + np.outer(g, shift)
    if lev > 0 or lev1 > 0:
        # heavy-tailed row scales (lognormal): lev for all rows, lev1 extra for the minority group. Normalised to mean square 1.
        z = rng.standard_normal(n)
        sig = lev + lev1 * g
        a = np.exp(sig * z)
        a = a / np.sqrt((a ** 2).mean())
        X = X * a[:, None]
    beta = rng.standard_normal(d) / np.sqrt(d) / scales
    dev = mis * rng.standard_normal(d) / np.sqrt(d) / scales
    noise = np.where(g == 0, noise0, noise1) * rng.standard_normal(n)
    if spikes > 0:
        sp = rng.choice(np.where(g == 1)[0], int(spikes), replace=False)
        keep = np.ones(n, bool)
        keep[sp] = False
        X[:, d - 1] = np.where(keep, 0.0, spike_val)
        X[sp, :d - 1] = 0.0
        off = spike_out * noise1 * np.where(np.arange(len(sp)) % 2 == 0, 1.0, -1.0)
        if len(sp) % 2 == 1:
            off[-1] = 0.0
        noise[sp] += off
    b = X @ beta + g * (X @ dev) + noise
    return Problem(X, b, g, name)


def cs_gauss_trace_ratio(prob):
    """tr Sigma_CS / tr Sigma_Gauss for the linearised error (Lemma 2); 1 - 2 sum_i r_i^2 ||(A^T A)^{-1} a_i||^2 / (||r||^2 tr (A^T A)^{-1}).
    It is at least 1 - 2 max_i leverage_i, so CountSketch only beats Gaussian noticeably on coherent data."""
    r2 = prob.r ** 2
    tg = r2.sum() * np.trace(prob.G)
    return float(1.0 - 2.0 * np.sum(r2 * (prob.P ** 2).sum(1)) / tg), float(((prob.A * prob.P).sum(1)).max())


def _standardise(df_num):
    out = df_num.copy()
    for c in out.columns:
        s = out[c].std()
        if s > 0:
            out[c] = (out[c] - out[c].mean()) / s
    return out


def load_law(group="race", include_sensitive=False, tier_onehot=True, path=None):
    """Law school data (damtharvey/law-school-dataset). Target: zfygpa (first-year GPA, standardised).
    Features: intercept, lsat, ugpa, fam_inc, fulltime, tier.  (zgpa/decile*/pass_bar are outcomes: excluded.)
    group='race': racetxt==1 -> g=1 (ASSUMED to be the majority/white group: check the dataset's README),
                  racetxt==0 -> g=0.   group='sex': male -> g=1.
    NOTE: the sensitive attribute is excluded from the features unless include_sensitive=True."""
    df = pd.read_csv(path or LAW_URL).dropna()
    g = (df["racetxt"] == 1).astype(int).values if group == "race" else (df["male"] == 1).astype(int).values
    num = df[["lsat", "ugpa", "fam_inc", "fulltime"]].astype(float)
    if include_sensitive:
        num["male"] = df["male"].astype(float)
        num["racetxt"] = df["racetxt"].astype(float)
    if tier_onehot:
        T = pd.get_dummies(df["tier"].astype(int), prefix="tier", drop_first=True).astype(float)
        num = pd.concat([num, T], axis=1)
    else:
        num["tier"] = df["tier"].astype(float)
    X = _standardise(num)
    X.insert(0, "intercept", 1.0)
    return Problem(X.values, df["zfygpa"].values.astype(float), g, f"law-{group}")


def load_acs(state="CA", year=2018, n_max=200000, group="sex", seed=0):
    """ACS PUMS person file, read straight from the Census zip into memory (no file written).
    Target: log(PINCP). Filters follow folktables' ACSIncome. group='sex': female->g=1; 'race': RAC1P!=1 -> g=1."""
    url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/1-Year/csv_p{state.lower()}.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=300).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    cols = ["AGEP", "COW", "SCHL", "MAR", "RELP", "WKHP", "SEX", "RAC1P", "PINCP", "PWGTP"]
    df = pd.read_csv(z.open(name), usecols=cols)
    df = df[(df.AGEP > 16) & (df.PINCP > 100) & (df.WKHP > 0) & (df.PWGTP >= 1)].dropna()
    if len(df) > n_max:
        df = df.sample(n_max, random_state=seed)
    g = (df["SEX"] == 2).astype(int).values if group == "sex" else (df["RAC1P"] != 1).astype(int).values
    num = pd.DataFrame({"AGEP": df.AGEP.astype(float), "AGEP2": df.AGEP.astype(float) ** 2,
                        "SCHL": df.SCHL.astype(float), "WKHP": df.WKHP.astype(float)})
    cat = pd.get_dummies(df[["COW", "MAR", "RELP"]].astype(int).astype(str), drop_first=True).astype(float)
    X = _standardise(pd.concat([num.reset_index(drop=True), cat.reset_index(drop=True)], axis=1))
    X.insert(0, "intercept", 1.0)
    return Problem(X.values, np.log(df["PINCP"].values.astype(float)), g, f"acs-{state}{year}-{group}")


def alter(prob, kind, val, seed=0):
    """Synthetic alterations of a real dataset, for ablations."""
    rng = np.random.default_rng(seed)
    A, b, g = prob.A.copy(), prob.b.copy(), prob.g.copy()
    if kind == "scale_col":            # shrink the last non-intercept column: raises the condition number
        A[:, -1] *= 1.0 / val
    elif kind == "subsample0":         # keep fraction val of group 0
        keep = np.ones(len(g), bool)
        i0 = np.where(g == 0)[0]
        keep[i0] = False
        keep[rng.choice(i0, int(val * len(i0)), replace=False)] = True
        A, b, g = A[keep], b[keep], g[keep]
    elif kind == "noise1":             # add label noise (sd = val * sd(b)) to group 1
        b[g == 1] += val * b.std() * rng.standard_normal((g == 1).sum())
    else:
        raise ValueError(kind)
    return Problem(A, b, g, f"{prob.name}|{kind}={val}")


ALTER_LABEL = {"scale_col": "last-column shrink factor", "subsample0": "group-0 fraction kept", "noise1": "group-1 extra noise"}


def fmt(x):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if np.isinf(x):
        return r"$\infty$"
    return f"{x:.3g}"


def write_booktabs(df, cols, headers, path, caption="", label=""):
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{" + "l" * 2 + "r" * (len(cols) - 2) + "}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    last = None
    for _, row in df.iterrows():
        if last is not None and row[cols[0]] != last:
            lines.append(r"\midrule")
        last = row[cols[0]]
        lines.append(" & ".join(fmt(row[c]) if c not in ("family", "metric", "method") else
                                FAM_LABEL.get(row[c], str(row[c])) for c in cols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        lines.append(r"\caption{" + caption + "}")
    if label:
        lines.append(r"\label{" + label + "}")
    lines.append(r"\end{table}")
    open(path, "w").write("\n".join(lines))


def save_vanilla_table(df, out, tag):
    df = df.copy()
    df["metric"] = pd.Categorical(df["metric"], METRICS, ordered=True)
    df["family"] = pd.Categorical(df["family"], FAMS, ordered=True)
    df = df.sort_values(["metric", "family"])
    df["ratio_thm1"] = df["thm1"] / df["emp_mean"]
    df["ratio_thm1_kfree"] = df["thm1_kfree"] / df["emp_mean"]
    df.to_csv(f"{out}/table_{tag}.csv", index=False)
    write_booktabs(df, ["metric", "family", "ols_value", "emp_mean", "prop1_tight", "prop1_sqrt", "thm1_kfree", "ratio_thm1_kfree"],
                   ["Metric", "Sketch", "OLS value", "Empirical", "Prop.~1 (tight)", "Prop.~1", "Thm.~1", "Thm.~1 / emp."],
                   f"{out}/table_{tag}.tex", caption=f"Distortion under sketching ({tag}). Empirical = mean over repetitions.",
                   label=f"tab:{tag}")


def save_strat_table(df, out, tag):
    df = df.copy()
    df["metric"] = pd.Categorical(df["metric"], METRICS, ordered=True)
    df["family"] = pd.Categorical(df["family"], SFAMS + ["is"], ordered=True)
    df["method"] = pd.Categorical(df["method"], ["vanilla", "oracle", "proxy"], ordered=True)
    df = df.sort_values(["metric", "family", "method"])
    df["frac_worse"] = (100 * df["frac_worse"]).round(1)
    df.to_csv(f"{out}/strat_{tag}.csv", index=False)
    write_booktabs(df, ["metric", "family", "method", "ols_value", "mean_abs_rel", "mean_rel", "frac_worse", "J_over_vanilla", "R_oracle_gain", "kappa_hat"],
                   ["Metric", "Sketch", "Method", "OLS value", "Mean $|$change$|$ (rel.)", "Mean change (rel.)", "\\% worse",
                    "$J/J_{\\rm van}$", "$R$", "$\\hat\\kappa$"],
                   f"{out}/strat_{tag}.tex",
                   caption=f"Unsketched (OLS), sketched (vanilla), oracle and proxy ({tag}). Changes are relative to OLS; positive = less fair.",
                   label=f"tab:strat_{tag}")


def plot_vanilla_sweep(df, xcol, xlabel, path, logx=True):
    """Empirical mean distortion (solid) and Theorem 1 bound (dashed) per sketch family. No title: context goes in the caption."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, met in zip(axes, METRICS):
        for fam in FAMS:
            d = df[(df.metric == met) & (df.family == fam)].sort_values(xcol)
            c = COLOR[fam]
            ax.plot(d[xcol], d["emp_mean"], "o-", color=c, label=FAM_LABEL[fam])
            ax.plot(d[xcol], d["thm1_kfree"].replace(np.inf, np.nan), "--", color=c, alpha=0.8)
        ax.set_xscale("log" if logx else "linear")
        ax.set_yscale("log")
        ax.set_xlabel(tx(xlabel))
        ax.set_ylabel(f"{met}: distortion")
    axes[0].plot([], [], "k--", label="Thm 1 bound")
    axes[0].legend(fontsize=8)
    style_axes(axes, _xt(df, xcol))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_strat_sweep(df, xcol, xlabel, path, logx=True, title="", ycol="J_over_vanilla"):
    """ycol='J_over_vanilla': theory objective relative to the vanilla sketch (1 = vanilla).
    ycol='emp_mean': empirical mean distortion of the metric (OLS itself is 0 by definition), with the vanilla
    sketch of the same family and the importance-sampling vanilla sketch as references."""
    absolute = ycol != "J_over_vanilla"
    fig, axes = plt.subplots(len(SFAMS), 3, figsize=(14, 9), sharex=True)
    sty = {"oracle": ("tab:green", "o-"), "proxy": ("tab:purple", "s-")}
    if absolute:
        sty = {"vanilla": ("k", "x-"), **sty}
    for i, fam in enumerate(SFAMS):
        for j, met in enumerate(METRICS):
            ax = axes[i, j]
            if not absolute:
                ax.axhline(1.0, color="k", lw=0.8)
            for method, (c, st) in sty.items():
                d = df[(df.metric == met) & (df.family == fam) & (df.method == method)].sort_values(xcol)
                ax.plot(d[xcol], d[ycol], st, color=c, label=method)
            if absolute:
                d = df[(df.metric == met) & (df.family == "is") & (df.method == "vanilla")].sort_values(xcol)
                ax.plot(d[xcol], d[ycol], ":", color="tab:red", label="importance (vanilla)")
                ax.set_yscale("log")
            ax.set_xscale("log" if logx else "linear")
            if i == 0:
                ax.set_title(met)
            if j == 0:
                ax.set_ylabel(tx(f"{FAM_LABEL[fam]}\n{ycol}"))
            if i == len(SFAMS) - 1:
                ax.set_xlabel(tx(xlabel))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(tx(title + ("   (OLS = 0 by definition)" if absolute else "")))
    style_axes(axes, _xt(df, xcol))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Drivers
# ----------------------------------------------------------------------------------------------
def run_sweep_vanilla(name, xs, builder, m, reps, rng, out, xlabel, logx=True, extra_m=None):
    frames = []
    for x in xs:
        prob = builder(x)
        mm = extra_m(x) if extra_m else m
        d = vanilla_table(prob, mm, reps, rng)
        d["x"] = x
        frames.append(d)
        print(f"  {name} x={x}: done", flush=True)
    df = pd.concat(frames)
    df.to_csv(f"{out}/sweep_{name}.csv", index=False)
    plot_vanilla_sweep(df, "x", xlabel, f"{out}/sweep_{name}.png", logx=logx, title=name)
    return df


def run_sweep_strat(name, xs, builder, mfun, mpfun, reps, rng, out, xlabel, logx=True, pilot_fam="gauss"):
    frames = []
    for x in xs:
        prob = builder(x)
        d = strat_experiment(prob, mfun(x), mpfun(x), reps, rng, pilot_fam=pilot_fam)
        d["x"] = x
        frames.append(d)
        print(f"  {name} x={x}: done", flush=True)
    df = pd.concat(frames)
    df.to_csv(f"{out}/strat_sweep_{name}.csv", index=False)
    plot_strat_sweep(df, "x", xlabel, f"{out}/strat_sweep_{name}.png", logx=logx, title=name)
    plot_strat_sweep(df, "x", xlabel, f"{out}/strat_sweep_{name}_distortion.png", logx=logx, title=name, ycol="emp_mean")
    return df




def add_common_args(ap):
    ap.add_argument("--out", default="results")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--quick", action="store_true", help="few reps, smaller n, fewer sweep points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=500, help="default sketch size")
    ap.add_argument("--m_pilot", type=int, default=100)
    ap.add_argument("--pilot_fam", default="gauss", choices=FAMS)
    ap.add_argument("--latex", action="store_true", help="render plot text with LaTeX (needs LaTeX installed)")
    return ap


def add_synth_args(ap):
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--d", type=int, default=10)
    return ap


def add_real_args(ap, acs_n=200000, hmda_n=200000):
    """Real-data options shared by every `real` driver. By default ALL THREE are run: LSAC, ACS California 2018, HMDA California 2018."""
    ap.add_argument("--law_path", default=None, help="local csv instead of the GitHub url")
    ap.add_argument("--no_law", action="store_true", help="skip LSAC")
    ap.add_argument("--no_acs", action="store_true", help="skip ACS")
    ap.add_argument("--acs_state", default="CA", help="ACS state (downloaded into memory)")
    ap.add_argument("--acs_year", type=int, default=2018)
    ap.add_argument("--acs_n", type=int, default=acs_n, help="max ACS rows (random subsample)")
    ap.add_argument("--acs_group", default="sex", choices=["sex", "race"])
    ap.add_argument("--no_hmda", action="store_true", help="skip HMDA")
    ap.add_argument("--hmda_state", default="CA")
    ap.add_argument("--hmda_year", type=int, default=2018)
    ap.add_argument("--hmda_n", type=int, default=hmda_n, help="max HMDA rows (random subsample)")
    ap.add_argument("--hmda_group", default="race", choices=["race", "ethnicity", "sex"], help="race: Black vs White applicants")
    ap.add_argument("--hmda_path", default=None, help="local CSV from https://ffiec.cfpb.gov/data-browser/ instead of the API")
    ap.add_argument("--diab", action="store_true", help="add Diabetes 130-US Hospitals")
    ap.add_argument("--diab_path", default=None, help="local CSV of the Fairlearn frame; default fetches from OpenML")
    ap.add_argument("--diab_n", type=int, default=50000)
    ap.add_argument("--enem_path", default=None, help="MICRODADOS_ENEM_2020.csv from INEP; ENEM is skipped if unset")
    ap.add_argument("--enem_group", default="race", choices=["race", "sex"])
    ap.add_argument("--enem_score", default="NU_NOTA_CH", choices=["NU_NOTA_CH", "NU_NOTA_LC", "NU_NOTA_MT", "NU_NOTA_CN"])
    ap.add_argument("--enem_n", type=int, default=50000)
    ap.add_argument("--allow_missing", action="store_true", help="warn and continue if a dataset cannot be loaded (default: stop)")
    return ap


def load_real(args, law_fn, acs_fn, hmda_fn, extra=()):
    """Load LSAC, ACS and HMDA (unless --no_law / --no_acs / --no_hmda), then any extra datasets, given as
    (name, skip, fn(args), how_to_skip) tuples. A failed load stops the run unless --allow_missing."""
    out = []
    for name, skip, fn, how in (("LSAC (law school)", args.no_law, law_fn, "pass --no_law"),
                                (f"ACS {args.acs_state} {args.acs_year}", args.no_acs, acs_fn, "pass --no_acs"),
                                (f"HMDA {args.hmda_state} {args.hmda_year}", args.no_hmda, hmda_fn, "pass --no_hmda"),
                                *extra):
        if skip:
            continue
        print(f"loading {name} ...", flush=True)
        try:
            out.append(fn(args))
        except Exception as e:
            msg = f"could not load {name} ({type(e).__name__}: {e})"
            if not args.allow_missing:
                raise SystemExit(f"ERROR: {msg}. Fix the connection or path, or {how} to skip it, or --allow_missing to continue without it.")
            print(f"WARNING: {msg}; continuing without it.", flush=True)
    return out


def finish_args(args):
    if args.quick:
        args.reps = min(args.reps, 30)
        if hasattr(args, "n"):
            args.n = min(args.n, 8000)
    os.makedirs(args.out, exist_ok=True)
    set_style(args.latex)
    return args


def load_hmda_reg(args):
    import lib_hmda_data
    A, y, g, name = lib_hmda_data.load(args.hmda_state, args.hmda_year, args.hmda_group, args.hmda_n, args.seed, args.hmda_path, "regression")
    return Problem(A, y, g, name)


def load_diab_reg(args):
    import lib_extra_data
    return Problem(*lib_extra_data.load_diabetes(args.diab_n, args.seed, args.diab_path, "regression"))


def load_enem_reg(args):
    import lib_extra_data
    return Problem(*lib_extra_data.load_enem(args.enem_path, args.enem_group, args.enem_score, args.enem_n, args.seed, "regression"))


def real_problems(args):
    """Regression versions of the real datasets (target: zfygpa for LSAC, log income for ACS, interest rate for HMDA,
    length of stay for Diabetes (--diab), exam score for ENEM (--enem_path))."""
    extra = [("Diabetes 130-US Hospitals", not args.diab, load_diab_reg, "drop --diab"),
             (f"ENEM 2020 ({args.enem_path})", not args.enem_path, load_enem_reg, "drop --enem_path")]
    return load_real(args, lambda a: load_law("race", path=a.law_path),
                     lambda a: load_acs(a.acs_state, a.acs_year, a.acs_n, a.acs_group, a.seed), load_hmda_reg, extra)


def synth_builder(args):
    def synth(**kw):
        p = dict(n=args.n, d=args.d)
        p.update(kw)
        return make_synth(seed=args.seed, **p)
    return synth


def sweep_pts(args, full, quick):
    return quick if args.quick else full


# ----------------------------------------------------------------------------------------------
# "Harm" experiments: empirical effect of sketching on the three fairness metrics, independent of theory.
# Signed distortions: positive = the fairness gap got WORSE than under OLS.
#   IF  : ||x~||/||x^|| - 1        GFE : |v^T x~| - |v^T x^|        MSE : |Gap(x~)| - |Gap(x^)|
# Relative versions divide GFE and MSE by their OLS value (IF is already relative).
# ----------------------------------------------------------------------------------------------
def harm_runs(prob, fam, m, reps, rng):
    S = np.zeros((reps, 3))
    X = np.zeros(reps)
    OK = np.zeros(reps, bool)
    xh = prob.xhat
    base_gram = prob.gram[0] + prob.gram[1]
    rr = float(prob.r @ prob.r)
    for t in range(reps):
        xt, OK[t] = prob.solve_checked(vanilla_SQ(prob, fam, m, rng))
        S[t] = [np.linalg.norm(xt) / np.linalg.norm(xh) - 1.0,
                abs(prob.v @ xt) - abs(prob.v @ xh),
                abs(prob.gap(xt)) - abs(prob.gap(xh))]
        D = xt - xh
        X[t] = float(D @ base_gram @ D) / rr          # relative excess squared loss ||A D||^2 / ||r||^2
    return S, X, OK


def harm_base(prob):
    ols = prob.ols_values()
    return {"IF": 1.0, "GFE": ols["GFE"], "MSE": ols["MSE"]}


def harm_table(prob, m, reps, rng, fams=FAMS):
    rows = []
    base = harm_base(prob)
    ols = prob.ols_values()
    for fam in fams:
        S, X, OK = harm_runs(prob, fam, m, reps, rng)
        ff = 1.0 - OK.mean()
        if OK.sum() < 3:
            continue
        S, X = S[OK], X[OK]
        nok = len(X)
        for j, met in enumerate(METRICS):
            s = S[:, j]
            rel = s / base[met]
            rows.append(dict(dataset=prob.name, m=m, family=fam, metric=met, ols_value=ols[met], reps=reps,
                             mean_signed=s.mean(), se_signed=s.std(ddof=1) / np.sqrt(nok), frac_failed=ff,
                             frac_worse=(s > 0).mean(), mean_abs=np.abs(s).mean(),
                             mean_rel=rel.mean(), mean_abs_rel=np.abs(rel).mean(), q10_rel=np.quantile(rel, 0.1), q90_rel=np.quantile(rel, 0.9),
                             excess_loss_rel=X.mean()))
    return pd.DataFrame(rows)


def save_harm_table(df, out, tag):
    df = df.copy()
    df["metric"] = pd.Categorical(df["metric"], METRICS, ordered=True)
    df["family"] = pd.Categorical(df["family"], FAMS, ordered=True)
    df = df.sort_values(["metric", "family"])
    df["frac_worse"] = (100 * df["frac_worse"]).round(1)
    df.to_csv(f"{out}/harm_{tag}.csv", index=False)
    write_booktabs(df, ["metric", "family", "ols_value", "mean_signed", "mean_rel", "frac_worse", "q10_rel", "q90_rel", "excess_loss_rel"],
                   ["Metric", "Sketch", "OLS value", "Mean change", "Mean rel.\\ change", "\\% runs worse", "10th pct (rel.)",
                    "90th pct (rel.)", "Excess loss (rel.)"],
                   f"{out}/harm_{tag}.tex",
                   caption=f"Effect of sketching on fairness ({tag}). Positive change = larger disparity than OLS.", label=f"tab:harm_{tag}")


def plot_harm_sweep(df, xcol, xlabel, path, logx=True, title=""):
    """Panels 1-3: signed relative change (mean and 10th-90th percentile band), failed sketches excluded.
    Panel 4: fraction of sketches that failed (numerically rank-deficient, e.g. uniform sampling missing a rare category)."""
    from matplotlib.ticker import NullFormatter
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, met in zip(axes[:3], METRICS):
        ax.axhline(0.0, color="k", lw=0.8)
        for fam in FAMS:
            d = df[(df.metric == met) & (df.family == fam)].sort_values(xcol)
            if d.empty:
                continue
            ax.plot(d[xcol], d["mean_rel"], "o-", color=COLOR[fam], label=FAM_LABEL[fam])
            ax.fill_between(d[xcol], d["q10_rel"], d["q90_rel"], color=COLOR[fam], alpha=0.12)
        ax.set_xscale("log" if logx else "linear")
        dm = df[df.metric == met]
        yl = robust_lim(np.concatenate([dm["q10_rel"].values, dm["q90_rel"].values, dm["mean_rel"].values]), q=(0, 100), include=0.0)
        if yl:
            ax.set_ylim(*yl)
        ax.set_xlabel(tx(xlabel))
        ax.set_title(met + (" (change in norm, relative)" if met == "IF" else " (relative to OLS value)"))
    axes[0].set_ylabel("signed change (positive = less fair)")
    axes[0].legend(fontsize=8)
    ax = axes[3]
    for fam in FAMS:
        d = df[(df.metric == "IF") & (df.family == fam)].sort_values(xcol)
        if not d.empty:
            ax.plot(d[xcol], d["frac_failed"], "o-", color=COLOR[fam], label=FAM_LABEL[fam])
    ax.set_xscale("log" if logx else "linear")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel(tx(xlabel))
    ax.set_title("fraction of failed sketches")
    fig.suptitle(tx(title + "   (line: mean over successful runs, band: 10th to 90th percentile)"))
    style_axes(axes, _xt(df, xcol))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run_sweep_harm(name, xs, builder, m, reps, rng, out, xlabel, logx=True, extra_m=None):
    frames = []
    for x in xs:
        prob = builder(x)
        d = harm_table(prob, extra_m(x) if extra_m else m, reps, rng)
        d["x"] = x
        frames.append(d)
        print(f"  harm {name} x={x}: done", flush=True)
    df = pd.concat(frames)
    df.to_csv(f"{out}/harm_sweep_{name}.csv", index=False)
    plot_harm_sweep(df, "x", xlabel, f"{out}/harm_sweep_{name}.png", logx=logx, title=name)
    return df


def save_method_table(df, path_stem, metric_order, method_order, method_label, caption, label, echo=True):
    """Table of metrics x methods (used by classification): raw CSV, booktabs .tex with readable method names and percent worse."""
    d = df.copy()
    d["method"] = pd.Categorical(d["method"], method_order, ordered=True)
    d["metric"] = pd.Categorical(d["metric"], metric_order, ordered=True)
    d = d.sort_values(["metric", "method"])
    d.to_csv(f"{path_stem}.csv", index=False)
    t = d.copy()
    t["method"] = t["method"].astype(str).map(method_label)
    t["frac_worse"] = (100 * t["frac_worse"]).round(1)
    cols = ["metric", "method", "mean_abs_rel", "mean_rel", "frac_worse", "loss_ratio"]
    write_booktabs(t, cols, ["Metric", "Method", "Mean $|$change$|$ (rel.)", "Mean change (rel.)", "\\% worse", "$f(\\tilde x)/f(\\hat x)$"],
                   f"{path_stem}.tex", caption=caption, label=label)
    if echo:
        print(t[cols].round(3).to_string(index=False), flush=True)


def trim_x(df, xcol, logx, series_col, keys, tol=0.05, frac=1 / 3):
    """Keep x values up to the point where the region in which every plotted series is below `tol` (mean |relative change|,
    all metrics) fills at most `frac` of the axis (log or linear spacing). Everything counts as converged from x* on if it
    stays below tol for all larger x. Nothing is cut if the last point has not converged, or if every point has (no transition)."""
    xs = np.array(sorted(df[xcol].unique()), float)
    d = df[df[series_col].isin(keys)]
    ok = np.array([(d[d[xcol] == x]["mean_abs_rel"] <= tol).all() for x in xs])
    if not ok[-1] or ok.all():
        return df
    first = next(i for i in range(len(xs)) if ok[i:].all())
    f = np.log if logx else (lambda v: v)
    lo, xstar = f(xs[0]), f(xs[first])
    keep = xs[f(xs) - lo <= (xstar - lo) / (1 - frac) + 1e-12]       # (end - x*) <= frac * (end - lo)
    return df[df[xcol].isin(keep)]


def sd_ylab(met):
    return f"{met}: $|$rel. change$|$", f"{met}: rel. change"      # math-mode bars: the SciencePlots serif font has no "|"


def plot_sd(df, xcol, xlabel, path, series, series_col, metrics, logx=True, band=False, zero_label=None,
            zero_color="k", min_base=None, mask_metrics=(), floor=None, tight_size=False, trim=True, log_ratio=20.0):
    """Generic size/direction figure shared by regression and classification. No title: context goes in the caption.
    Row 1 (size): mean |relative change| vs the unsketched/exact solution.  Row 2 (direction): mean signed relative change
    (positive = less fair), optionally with a 10th-90th percentile band, and a zero line for the unsketched/exact solution.
    The size row is linear from 0, unless its values in that panel span more than log_ratio (largest / smallest), then log.
    When a panel is linear and nothing in the direction row is negative, both rows share one y range so heights compare directly.
      series        list of dict(key, label, color[, style]) ; the rows of df with df[series_col] == key
      min_base      with mask_metrics: omit points whose exact gap |base| is below this (relative change is meaningless there)
      floor         on a log size row, omit points with size <= floor (exact zeros cannot sit on a log axis)
      tight_size    on a log size row, scale it to the plotted data
      trim          if every plotted series converges, cut the x axis so the converged part is at most 1/3 of it (trim_x)"""
    if trim:
        vis = df
        if min_base is not None and "base" in df:
            vis = df[~(df.metric.isin(mask_metrics) & (df["base"].abs() < min_base))]
        df = df[df[xcol].isin(trim_x(vis, xcol, logx, series_col, [s["key"] for s in series])[xcol].unique())]
    fig, axes = plt.subplots(2, len(metrics), figsize=(3.6 * len(metrics) + 2.0, 6.0), sharex="col")
    for j, met in enumerate(metrics):
        dm = df[df.metric == met]
        if min_base is not None and met in mask_metrics and "base" in dm:
            kept = dm[dm["base"].abs() >= min_base]
            if kept.empty and not dm.empty:
                axes[0, j].text(0.5, 0.5, f"exact gap below {min_base}:\nrelative change undefined", ha="center", va="center",
                                transform=axes[0, j].transAxes, fontsize=9)
            dm = kept
        curves = [(sr, dm[dm[series_col] == sr["key"]].sort_values(xcol)) for sr in series]
        curves = [(sr, d) for sr, d in curves if not d.empty]
        sizes = np.concatenate([d["mean_abs_rel"].values for _, d in curves]) if curves else np.array([])
        nz = sizes[sizes > (floor or 0)]
        logy = len(nz) > 0 and nz.max() > log_ratio * nz.min()     # linear unless the size row spans more than log_ratio
        ypos, dvals = [], []
        for sr, d in curves:
            st = sr.get("style", "o-")
            pos = (d["mean_abs_rel"] > floor).values if (logy and floor is not None) else np.ones(len(d), bool)
            axes[0, j].plot(d[xcol][pos], d["mean_abs_rel"][pos], st, color=sr["color"], label=sr["label"])
            axes[1, j].plot(d[xcol], d["mean_rel"], st, color=sr["color"])
            ypos += list(d["mean_abs_rel"][pos])
            dvals += list(d["mean_rel"])
            if band:
                axes[1, j].fill_between(d[xcol], d["q10_rel"], d["q90_rel"], color=sr["color"], alpha=0.11)
                dvals += list(d["q10_rel"]) + list(d["q90_rel"])
        axes[1, j].axhline(0.0, color=zero_color, lw=1.2 if zero_label else 0.8, label=zero_label)
        if logy:
            axes[0, j].set_yscale("log")
            if tight_size:
                axes[0, j].set_ylim(min(ypos) / 1.6, max(ypos) * 1.6)
        elif ypos:
            axes[0, j].set_ylim(0, max(ypos) * 1.08)
        if not logy and ypos and dvals and min(dvals) >= 0:              # both rows non-negative and linear: same y range
            top = max(max(ypos), max(dvals)) * 1.08
            axes[0, j].set_ylim(0, top)
            axes[1, j].set_ylim(0, top)
        elif dvals:
            axes[1, j].set_ylim(*robust_lim(np.array(dvals), q=(0, 100), include=0.0))
        axes[1, j].set_xlabel(tx(xlabel))
        for i in range(2):
            axes[i, j].set_xscale("log" if logx else "linear")
        a, b = sd_ylab(met)
        axes[0, j].set_ylabel(a)
        axes[1, j].set_ylabel(b)
    axes[0, 0].legend(fontsize=7)
    if zero_label:
        axes[1, 0].legend(fontsize=7, loc="upper right")
    style_axes(axes, _xt(df, xcol))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_size_direction(df, xcol, xlabel, path, logx=True):
    """Regression, vanilla sketches: one line per sketch family. Rank-deficient sketches are excluded and noted in the legend."""
    first = df[df.metric == METRICS[0]]
    series = []
    for fam in FAMS:
        d = first[first.family == fam]
        if d.empty:
            continue
        ff = d["frac_failed"].max()
        series.append(dict(key=fam, color=COLOR[fam], label=FAM_LABEL[fam] + (f" (up to {100 * ff:.0f}% failed)" if ff > 0 else "")))
    plot_sd(df, xcol, xlabel, path, series, "family", METRICS, logx=logx, band=True)


def run_sweep_sd(name, xs, builder, m, reps, rng, out, xlabel, logx=True, extra_m=None):
    frames = []
    for x in xs:
        prob = builder(x)
        d = harm_table(prob, extra_m(x) if extra_m else m, reps, rng)
        d["x"] = x
        frames.append(d)
        print(f"  {name} x={x}: done", flush=True)
    df = pd.concat(frames)
    df.to_csv(f"{out}/{name}.csv", index=False)
    plot_size_direction(df, "x", xlabel, f"{out}/{name}.png", logx=logx)
    return df


def plot_methods_sd(df, xcol, xlabel, path, fam, logx=True):
    """Regression, Section 5: unsketched (OLS, the zero line) vs vanilla sketch vs stratified oracle vs stratified proxy, for one family."""
    series = [dict(key="vanilla", color="k", style="x-", label="sketched (vanilla)"),
              dict(key="oracle", color="tab:green", style="o-", label="oracle"),
              dict(key="proxy", color="tab:purple", style="s-", label="proxy")]
    plot_sd(df[df.family == fam], xcol, xlabel, path, series, "method", METRICS, logx=logx,
            band=True, zero_label="unsketched (OLS)", zero_color="tab:red")


def run_sweep_methods(name, xs, builder, mfun, mpfun, reps, rng, out, xlabel, logx=True, pilot_fam="gauss", fams=SFAMS):
    frames = []
    for x in xs:
        prob = builder(x)
        d = strat_experiment(prob, mfun(x), mpfun(x), reps, rng, pilot_fam=pilot_fam, fams=list(fams))
        d["x"] = x
        frames.append(d)
        print(f"  {name} x={x}: done", flush=True)
    df = pd.concat(frames)
    df.to_csv(f"{out}/{name}.csv", index=False)
    for fam in fams:
        plot_methods_sd(df, "x", xlabel, f"{out}/{name}_{fam}.png", fam, logx=logx)
    return df


def beta_table(prob):
    """Analytic (no sketching) group quantities and oracle gain R = V/O for each metric and family."""
    rows = []
    for met in METRICS:
        C = prob.C[met]
        for fam in SFAMS:
            bt = coefs(prob, C, prob.r, fam)
            b0, b1 = bt[0]["beta"], bt[1]["beta"]
            V = vanilla_value(prob, C, fam)
            O = (np.sqrt(b0) + np.sqrt(b1)) ** 2
            rows.append(dict(metric=met, family=fam, rho0=bt[0]["rho"], rho1=bt[1]["rho"], tau0=bt[0]["tau"], tau1=bt[1]["tau"],
                             zeta0=bt[0]["zeta"], zeta1=bt[1]["zeta"], beta0=b0, beta1=b1,
                             omega=np.sqrt(b1 / b0) if b0 > 0 else np.nan, V=V, O=O, R=V / O if O > 0 else np.nan))
    return pd.DataFrame(rows)


def plot_beta_gain(df, path):
    """Oracle gain R = V/O (analytic) against group asymmetry omega = sqrt(beta_1/beta_0); R > 1 means the oracle beats the plain sketch."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, met in zip(axes, METRICS):
        for fam in SFAMS:
            d = df[(df.metric == met) & (df.family == fam)]
            ax.scatter(d["omega"], d["R"], s=22, alpha=0.7, color=COLOR[fam], label=FAM_LABEL[fam])
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\omega = \sqrt{\beta_1 / \beta_0}$")
        ax.set_ylabel(f"{met}: oracle gain $R = V/O$")
    axes[0].legend(fontsize=8)
    style_axes(axes)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def robust_lim(vals, q=(1, 99), pad=0.08, include=None):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo, hi = np.percentile(v, q)
    if include is not None:
        lo, hi = min(lo, include), max(hi, include)
    span = (hi - lo) or abs(hi) or 1.0
    return lo - pad * span, hi + pad * span


def harm_scatter(prob, m, reps, rng, path, fams=FAMS):
    """One point per sketch draw: relative excess loss (accuracy cost) vs signed relative fairness change.
    Rank-deficient sketches are dropped (and counted in the legend); axes are fitted to the 1st-99th percentile."""
    base = harm_base(prob)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    allx = {j: [] for j in range(3)}
    ally = {j: [] for j in range(3)}
    for fam in fams:
        S, X, OK = harm_runs(prob, fam, m, reps, rng)
        lab = FAM_LABEL[fam] + (f" ({int((~OK).sum())} failed)" if (~OK).any() else "")
        for j, (ax, met) in enumerate(zip(axes, METRICS)):
            ax.scatter(X[OK], S[OK, j] / base[met], s=8, alpha=0.5, color=COLOR[fam], label=lab if j == 0 else None)
            allx[j].append(X[OK])
            ally[j].append(S[OK, j] / base[met])
    for j, (ax, met) in enumerate(zip(axes, METRICS)):
        ax.axhline(0.0, color="k", lw=0.8)
        xs, ys = np.concatenate(allx[j]), np.concatenate(ally[j])
        xs = xs[xs > 0]
        ax.set_xscale("log")
        if xs.size:
            lo, hi = np.percentile(np.log10(xs), [1, 99])
            pad = 0.1 * ((hi - lo) or 1)
            ax.set_xlim(10 ** (lo - pad), 10 ** (hi + pad))
        yl = robust_lim(ys, include=0.0)
        if yl:
            ax.set_ylim(*yl)
        ax.set_xlabel("relative excess loss")
        ax.set_title(met)
    axes[0].set_ylabel("signed relative change (positive = less fair)")
    axes[0].legend(fontsize=8)
    fig.suptitle(tx(f"{prob.name}, m={m}: accuracy cost vs fairness change, one point per sketch draw"))
    style_axes(axes)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)