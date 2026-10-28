import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
 
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
 
warnings.filterwarnings("ignore")
 
KINDS = ("gauss", "cs", "unif")
LABEL = {"gauss": "Gaussian", "cs": "CountSketch", "unif": "Uniform"}
COL = {"gauss": "#0072B2", "cs": "#D55E00", "unif": "#009E73"}
METS = ("IF", "GFE", "MSE")
ETAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
M_LIST = [60, 120, 240, 480]
M_REF = 240
KAPPAS = [1.5, 5, 20, 100, 500]
 
 
# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def make_ctx(n=1000, d=4, kappa=10.0, n0frac=0.25, seed=0, noise_ratio=1.0,
             shift=1.0, misspec=0.3):
    """Synthetic regression with two groups.
    kappa: target condition number (realised value is stored in ctx['kappa']).
    n0frac: size of group 0 / n.   noise_ratio: noise sd of group 0 / group 1.
    shift: group-0 mean shift in the first feature.  misspec: amplitude of an
    unmodelled sin() term (keeps the residual r nonzero)."""
    rng = np.random.default_rng(seed)
    n0 = max(int(round(n0frac * n)), d + 2)
    n1 = n - n0
    G = rng.normal(size=(n, d))
    G[:n0, 0] += shift
    s = np.geomspace(1.0, 1.0 / kappa, d)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    A = (G * s) @ Q.T
    coef = rng.normal(size=d)
    scale = 0.5 * (0.5 + np.abs(G[:, min(1, d - 1)]))
    scale[:n0] *= noise_ratio
    b = G @ coef + scale * rng.normal(size=n) + misspec * np.sin(2 * G[:, 0])
    I0, I1 = np.arange(n0), np.arange(n0, n)
    xh = np.linalg.lstsq(A, b, rcond=None)[0]
    r = b - A @ xh
    sig = np.linalg.svd(A, compute_uv=False)
    Ni = np.linalg.inv(A.T @ A)
    Ni = (Ni + Ni.T) / 2
    Pi = A @ Ni
    Pr = Pi * r[:, None]
    Ab = np.c_[A, b]
    U, _ = np.linalg.qr(Ab)
    return dict(
        n=n, d=d, n0=n0, n1=n1, A=A, b=b, Ab=Ab, U=U, I0=I0, I1=I1, xh=xh,
        nxh=float(np.linalg.norm(xh)), r=r, sig=sig, kappa=float(sig[0] / sig[-1]),
        Ni=Ni, Pi=Pi, Gm=Pr.T @ Pr,            # Gm = sum_i r_i^2 pi_i pi_i^T
        v=A[I0].mean(0) - A[I1].mean(0), w=A[I0].T @ r[I0],
        N0=A[I0].T @ A[I0], N1=A[I1].T @ A[I1])
 
 
def parity(c, x):
    e = c["A"] @ x - c["b"]
    n0 = c["n0"]
    return e[:n0] @ e[:n0] / n0 - e[n0:] @ e[n0:] / c["n1"]
 
 
# --------------------------------------------------------------------------
# sketches
# --------------------------------------------------------------------------
def draw_S(kind, m, n, rng):
    """Dense m x n sketch with E[S^T S] = I."""
    if kind == "gauss":
        return rng.normal(size=(m, n)) / np.sqrt(m)
    S = np.zeros((m, n))
    if kind == "cs":
        S[rng.integers(0, m, n), np.arange(n)] = rng.choice([-1.0, 1.0], n)
    else:  # uniform row sampling with replacement
        S[np.arange(m), rng.integers(0, n, m)] = np.sqrt(n / m)
    return S
 
 
def draw_Sstr(kind, m0, m1, c, rng):
    """Group-stratified sketch: block diagonal, one independent sketch per group."""
    n, n0 = c["n"], c["n0"]
    S = np.zeros((m0 + m1, n))
    S[:m0, :n0] = draw_S(kind, m0, n0, rng)
    S[m0:, n0:] = draw_S(kind, m1, n - n0, rng)
    return S
 
 
# --------------------------------------------------------------------------
# theory: Lemma 2 covariance, Proposition 1, Theorem 1
# --------------------------------------------------------------------------
def Sigma(c, kind, m):
    r, Ni = c["r"], c["Ni"]
    if kind == "gauss":
        return (r @ r) * Ni / m
    if kind == "cs":
        return ((r @ r) * Ni - 2 * c["Gm"]) / m
    return c["n"] * c["Gm"] / m
 
 
def moments(c, kind, m):
    Sg = Sigma(c, kind, m)
    v, w, xh = c["v"], c["w"], c["xh"]
    return dict(T=max(np.trace(Sg), 0.0), V=max(v @ Sg @ v, 0.0), W=max(w @ Sg @ w, 0.0),
                X=max(xh @ Sg @ xh, 0.0), Q0=np.trace(c["N0"] @ Sg), Q1=np.trace(c["N1"] @ Sg))
 
 
def prop_bounds(c, eps):
    """Proposition 1 (vectorised in eps; nan where eps > 1/2)."""
    eps = np.asarray(eps, float)
    ok = eps <= 0.5
    e = np.where(ok, eps, np.nan)
    se = np.sqrt(e)
    r = c["r"]
    rn, r0, r1 = np.linalg.norm(r), np.linalg.norm(r[c["I0"]]), np.linalg.norm(r[c["I1"]])
    sd = c["sig"][-1]
    n0, n1 = c["n0"], c["n1"]
    return dict(IF=2 * se * rn / (sd * c["nxh"]),
                GFE=2 * se * np.linalg.norm(c["v"]) * rn / sd,
                MSE=4 * se * rn * (r0 / n0 + r1 / n1) + 4 * e * rn ** 2 * (1 / n0 + 1 / n1))
 
 
def thm_bounds(c, M, eps, eta):
    """Theorem 1 (kappa-dependent version), holds w.p. >= 1 - delta - eta."""
    nan = float("nan")
    if not eps < 1:
        return dict(IF=nan, GFE=nan, MSE=nan, c=nan)
    cc = c["kappa"] * eps / (1 - eps)
    n0, n1 = c["n0"], c["n1"]
    lam = 2 * (1 / n0 + 1 / n1)
    s1 = c["sig"][0]
    T, V, W = M["T"], M["V"], M["W"]
    IF = (1 + cc) * np.sqrt(T / eta) / c["nxh"]
    GFE = np.sqrt(2 * V / eta) + cc * np.linalg.norm(c["v"]) * np.sqrt(2 * T / eta)
    MSE = (lam * (np.sqrt(4 * W / eta) + cc * np.linalg.norm(c["w"]) * np.sqrt(4 * T / eta))
           + (8 / eta) * (M["Q0"] / n0 + M["Q1"] / n1 + s1 ** 2 * cc ** 2 * T * (1 / n0 + 1 / n1)))
    return dict(IF=IF, GFE=GFE, MSE=MSE, c=cc)
 
 
# --------------------------------------------------------------------------
# vanilla-sketch simulation
# --------------------------------------------------------------------------
def simulate(c, kind, m, T, rng):
    n, d = c["n"], c["d"]
    Ab, U, Pi, r, xh, v = c["Ab"], c["U"], c["Pi"], c["r"], c["xh"], c["v"]
    I = np.eye(d + 1)
    gx, px = abs(v @ xh), abs(parity(c, xh))
    out = {k: np.empty(T) for k in ("IF", "GFE", "MSE", "eps")}
    D, Dp = np.empty((T, d)), np.empty((T, d))
    for t in range(T):
        S = draw_S(kind, m, n, rng)
        SAb = S @ Ab
        xt = np.linalg.lstsq(SAb[:, :d], SAb[:, d], rcond=None)[0]
        D[t] = xt - xh
        Dp[t] = Pi.T @ (S.T @ (S @ r) - r)
        SU = S @ U
        out["eps"][t] = np.abs(np.linalg.eigvalsh(SU.T @ SU - I)).max()
        out["IF"][t] = abs(np.linalg.norm(xt) / c["nxh"] - 1)
        out["GFE"][t] = abs(abs(v @ xt) - gx)
        out["MSE"][t] = abs(abs(parity(c, xt)) - px)
    out["D"], out["Dp"] = D, Dp
    return out
 
 
def run_cfg(p):
    c = make_ctx(**p["data"])
    kind, m, T = p["kind"], p["m"], p["T"]
    rng = np.random.default_rng([p["data"]["seed"], m, KINDS.index(kind), 7])
    sim = simulate(c, kind, m, T, rng)
    M = moments(c, kind, m)
    epsq = float(np.quantile(sim["eps"], 1 - p["delta"]))
    res = dict(kind=kind, m=m, tag=p.get("tag"), seed=p["data"]["seed"],
               kappa=c["kappa"], eps_q=epsq)
    thm = thm_bounds(c, M, epsq, p["eta"])
    prp = prop_bounds(c, epsq)
    pred = dict(IF=np.sqrt(M["X"]) / c["nxh"] ** 2, GFE=np.sqrt(M["V"]),
                MSE=2 * (1 / c["n0"] + 1 / c["n1"]) * np.sqrt(M["W"]))
    res["c"] = float(thm["c"])
    for k in METS:
        q = np.quantile(sim[k], [0.05, 0.5, 0.95])
        res[k + "_q05"], res[k + "_q50"], res[k + "_q95"] = map(float, q)
        res["thm_" + k], res["prop_" + k], res["pred_" + k] = float(thm[k]), float(prp[k]), float(pred[k])
        cov, bds, qs = [], [], []
        for eta in ETAS:
            bd = thm_bounds(c, M, epsq, eta)[k]
            cov.append(float(np.mean(sim[k] <= bd)) if np.isfinite(bd) else float("nan"))
            bds.append(float(bd))
            qs.append(float(np.quantile(sim[k], min(0.999, 1 - eta - p["delta"]))))
        res["cov_" + k], res["thmeta_" + k], res["qeta_" + k] = cov, bds, qs
    # moment validation (Lemma 2): simulated vs predicted
    Dp, D = sim["Dp"], sim["D"]
    res.update(mT_th=float(M["T"]), mT_sim=float(np.mean((Dp ** 2).sum(1))),
               mV_th=float(M["V"]), mV_sim=float(np.mean((Dp @ c["v"]) ** 2)),
               mW_th=float(M["W"]), mW_sim=float(np.mean((Dp @ c["w"]) ** 2)),
               mTD_sim=float(np.mean((D ** 2).sum(1))))
    # deterministic claims, checked per draw with that draw's own eps
    eps = sim["eps"]
    pb = prop_bounds(c, eps)
    nv = 0
    for k in METS:
        ok = eps <= 0.5
        nv += int(np.sum(sim[k][ok] > pb[k][ok] * (1 + 1e-9)))
    res["viol_prop"] = nv
    ok = eps < 1
    cc = c["kappa"] * eps[ok] / (1 - eps[ok])
    lhs = np.linalg.norm(D[ok] - Dp[ok], axis=1)
    rhs = cc * np.linalg.norm(Dp[ok], axis=1)
    res["viol_lem1"] = int(np.sum(lhs > rhs * (1 + 1e-9)))
    res["n_draws"] = int(T)
    return res
 
 
# --------------------------------------------------------------------------
# stratified objective (oracle) and proxy
# --------------------------------------------------------------------------
def stats(C, r, I):
    """rho, tau, zeta, s for the rows I (a group, or all rows for the vanilla sketch)."""
    Cg, rg = C[I], r[I]
    cr = Cg.T @ rg
    return dict(rho=float(rg @ rg), tau=float((Cg ** 2).sum()), zeta=float(cr @ cr),
                s=float(((Cg ** 2).sum(1) * rg ** 2).sum()), ng=len(I))
 
 
def beta(kind, g):
    """Per-group coefficient: objective = beta_0/m_0 + beta_1/m_1 (vanilla: beta/m)."""
    if kind == "gauss":
        return g["rho"] * g["tau"] + g["zeta"]
    if kind == "cs":
        return g["rho"] * g["tau"] + g["zeta"] - 2 * g["s"]
    return g["ng"] * g["s"] - g["zeta"]
 
 
def pos(x):
    return max(float(x), 1e-12)
 
 
def mc_obj(S, C, r):
    vec = (S @ C).T @ (S @ r) - C.T @ r
    return float(vec @ vec)
 
 
def run_phase_cell(p):
    c = make_ctx(**p["data"])
    C = c["Pi"] if p["target"] == "IF" else (c["Pi"] @ c["v"])[:, None]
    r, n, m, T = c["r"], c["n"], p["m"], p["T"]
    gs = [stats(C, r, c["I0"]), stats(C, r, c["I1"])]
    gv = stats(C, r, np.arange(n))
    rows = []
    for kind in KINDS:
        b0, b1 = pos(beta(kind, gs[0])), pos(beta(kind, gs[1]))
        Vv = pos(beta(kind, gv))
        t = np.sqrt(b0) / (np.sqrt(b0) + np.sqrt(b1))
        m0 = int(np.clip(round(m * t), 1, m - 1))
        m1 = m - m0
        V_pred, O_pred = Vv / m, b0 / m0 + b1 / m1
        rng = np.random.default_rng([p["data"]["seed"], m, KINDS.index(kind), 11])
        Vs = [mc_obj(draw_S(kind, m, n, rng), C, r) for _ in range(T)]
        Os = [mc_obj(draw_Sstr(kind, m0, m1, c, rng), C, r) for _ in range(T)]
        V_meas, O_meas = float(np.mean(Vs)), float(np.mean(Os))
        rv = np.std(Vs, ddof=1) / np.sqrt(T) / V_meas
        ro = np.std(Os, ddof=1) / np.sqrt(T) / max(O_meas, 1e-300)
        rows.append(dict(kind=kind, n0frac=p["data"]["n0frac"], noise_ratio=p["data"]["noise_ratio"],
                         m0=m0, m1=m1, V_pred=V_pred, V_meas=V_meas, O_pred=O_pred, O_meas=O_meas,
                         R_pred=V_pred / O_pred, R_meas=V_meas / max(O_meas, 1e-300),
                         R_se=float(V_meas / max(O_meas, 1e-300) * np.sqrt(rv ** 2 + ro ** 2)), T=T))
    return rows
 
 
def exb(t, k):
    """Excess bound t*(1-t*) 4 k^2 / (1-k^2), valid for k < 1."""
    return 4 * t * (1 - t) * k ** 2 / (1 - k ** 2) if k < 1 else float("nan")
 
 
def run_proxy(p):
    c = make_ctx(**p["data"])
    n, C, r = c["n"], c["Pi"], c["r"]
    I0, I1 = c["I0"], c["I1"]
    rng = np.random.default_rng([p["data"]["seed"], 99])
    gt = [stats(C, r, I0), stats(C, r, I1)]
    gv = stats(C, r, np.arange(n))
    maxrow = [float((C[I] ** 2).sum(1).max()) for I in (I0, I1)]
    rn = np.linalg.norm(r)
    I5 = np.eye(c["d"] + 1)
    rows = []
    for mp in p["mps"]:
        for _ in range(p["reps"]):
            S = draw_S("cs", mp, n, rng)              # pilot is always CountSketch
            xp = np.linalg.lstsq(S @ c["A"], S @ c["b"], rcond=None)[0]
            rt = c["b"] - c["A"] @ xp
            SU = S @ c["U"]
            ep = np.abs(np.linalg.eigvalsh(SU.T @ SU - I5)).max()
            gam = 2 * np.sqrt(ep) * rn if ep <= 0.5 else float("nan")
            gh = [stats(C, rt, I0), stats(C, rt, I1)]
            for kind in KINDS:
                b = [pos(beta(kind, g)) for g in gt]
                bh = [pos(beta(kind, g)) for g in gh]
                ts = np.sqrt(b[0]) / (np.sqrt(b[0]) + np.sqrt(b[1]))
                th = np.sqrt(bh[0]) / (np.sqrt(bh[0]) + np.sqrt(bh[1]))
                f = lambda t: b[0] / t + b[1] / (1 - t)
                fs, Vv = f(ts), pos(beta(kind, gv))
                kr = max(abs(np.sqrt(bh[g]) - np.sqrt(b[g])) / np.sqrt(b[g]) for g in (0, 1))
                if kind == "unif":
                    err = [np.sqrt(gt[g]["ng"] * maxrow[g]) * gam for g in (0, 1)]
                else:
                    err = [np.sqrt(2 * gt[g]["tau"]) * gam for g in (0, 1)]
                ka = max(err[g] / np.sqrt(b[g]) for g in (0, 1))
                rows.append(dict(kind=kind, mp=mp, seed=p["data"]["seed"], exc=f(th) / fs - 1,
                                 ratio=f(th) / Vv, orc=fs / Vv, kr=float(kr), ka=float(ka),
                                 br=exb(ts, kr), ba=exb(ts, ka)))
    return rows
 
 
# --------------------------------------------------------------------------
# plotting (every figure carries its own title, legend and explanatory caption)
# --------------------------------------------------------------------------
from matplotlib.ticker import LogFormatter, NullFormatter
 
plt.rcParams.update({"font.size": 11, "axes.titlesize": 11.5, "axes.labelsize": 11,
                     "legend.fontsize": 9.5, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
MARK = {"gauss": "o", "cs": "s", "unif": "^"}
NAME = {"IF": r"IF $=\,|\,\|\tilde x\|/\|\hat x\|-1\,|$",
        "GFE": r"GFE $=\,||v^\top\tilde x|-|v^\top\hat x||$",
        "MSE": r"MSE gap $=\,||P(\tilde x)|-|P(\hat x)||$"}
 
 
def med(rows, f):
    return float(np.nanmedian([r[f] for r in rows]))
 
 
def logfmt(ax, axes="xy"):
    for a in axes:
        axis = ax.xaxis if a == "x" else ax.yaxis
        axis.set_major_formatter(LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
        axis.set_minor_formatter(LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.5)))
 
 
def xticks_at(ax, vals):
    ax.set_xticks(vals)
    ax.set_xticklabels([f"{v:g}" for v in vals])
    ax.xaxis.set_minor_formatter(NullFormatter())
 
 
def kind_handles():
    return [Line2D([], [], color=COL[k], marker=MARK[k], lw=2, label=LABEL[k]) for k in KINDS]
 
 
def finish(fig, out, name, title, cap, handles=None, ncol=4):
    h = fig.get_figheight()
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1 - 0.12 / h)
    fig.tight_layout(rect=(0, (1.75 if handles else 1.0) / h, 1, 1 - 0.45 / h))
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.85 / h),
                   ncol=ncol, frameon=False)
    fig.text(0.012, 0.08 / h, cap, ha="left", va="bottom", fontsize=9.5, wrap=True, color="#333333")
    path = os.path.join(out, name)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("wrote", path)
 
 
def fig_moments(res, out):
    fig, axs = plt.subplots(1, 4, figsize=(17, 5.6))
    specs = [("mT_th", "mT_sim", r"$E\|\Delta'\|^2$  vs  tr $\Sigma$"),
             ("mV_th", "mV_sim", r"$E(v^\top\Delta')^2$  vs  $v^\top\Sigma v$"),
             ("mW_th", "mW_sim", r"$E(w^\top\Delta')^2$  vs  $w^\top\Sigma w$"),
             ("mT_th", "mTD_sim", r"actual $E\|\Delta\|^2$  vs  tr $\Sigma$")]
    for ax, (th, sm, ttl) in zip(axs, specs):
        allv = []
        for kind in KINDS:
            rows = [r for r in res if r["kind"] == kind]
            x, y = [r[th] for r in rows], [r[sm] for r in rows]
            ax.scatter(x, y, s=24, color=COL[kind], marker=MARK[kind], alpha=0.75,
                       label=LABEL[kind], edgecolor="none")
            allv += x + y
        lo, hi = min(allv) * 0.7, max(allv) * 1.4
        ax.plot([lo, hi], [lo, hi], "k-", lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xscale("log"); ax.set_yscale("log"); logfmt(ax)
        ratio = np.array([r[sm] / r[th] for r in res])
        ax.text(0.04, 0.96, f"simulation / formula\nmedian {np.median(ratio):.2f}\n"
                f"5-95%: {np.quantile(ratio, .05):.2f} to {np.quantile(ratio, .95):.2f}",
                transform=ax.transAxes, va="top", fontsize=9.5)
        ax.set_xlabel("formula (Lemma 2)"); ax.set_ylabel("simulation (mean over sketch draws)")
        ax.set_title(ttl)
    axs[0].legend(loc="lower right")
    cap = ("Each dot is one configuration (dataset, sketch type, and one value of m or kappa); the simulated second moment is "
           "averaged over the sketch draws. Dots on the black diagonal mean the formula matches the simulation. Panel 4 uses the "
           "actual error Delta instead of its linearisation Delta', so it also tests the approximation Delta ~ Delta'. "
           "The visible clusters are different m and kappa values.")
    finish(fig, out, "fig1_moments.png", "Check of Lemma 2: predicted vs simulated second moments of the sketching error", cap)
 
 
def fig_msweep(res, out, kappa_main):
    ms = sorted({r["m"] for r in res})
    grp = {}
    for r in res:
        grp.setdefault((r["kind"], r["m"]), []).append(r)
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for j, met in enumerate(METS):
        top, bot = axs[0, j], axs[1, j]
        for kind in KINDS:
            line = lambda f: np.array([med(grp[(kind, m)], f) for m in ms])
            q50 = line(met + "_q50")
            top.plot(ms, q50, color=COL[kind], lw=2.4)
            top.plot(ms, line("pred_" + met), color=COL[kind], ls=":", lw=1.8)
            top.plot(ms, line("thm_" + met), color=COL[kind], ls="--", lw=1.8)
            top.plot(ms, line("prop_" + met), color=COL[kind], ls="-.", lw=1.8)
            bot.plot(ms, line("thm_" + met) / q50, color=COL[kind], ls="--", marker="o", ms=5)
            bot.plot(ms, line("prop_" + met) / q50, color=COL[kind], ls="-.", marker="s", ms=5)
        top.set_title(NAME[met]); top.set_ylabel("error")
        bot.axhline(1, color="k", lw=1)
        bot.set_ylabel("bound / simulated median"); bot.set_xlabel("sketch size m")
        for ax in (top, bot):
            ax.set_xscale("log"); ax.set_yscale("log"); xticks_at(ax, ms); logfmt(ax, "y")
    hs = [Line2D([], [], color="k", lw=2.4, label="simulated median"),
          Line2D([], [], color="k", ls=":", lw=1.8, label="Lemma 2 first-order prediction"),
          Line2D([], [], color="k", ls="--", lw=1.8, label="Theorem 1 bound"),
          Line2D([], [], color="k", ls="-.", lw=1.8, label="Proposition 1 bound")] + kind_handles()
    cap = ("Top: error of each fairness metric against sketch size m. Bottom: each bound divided by the simulated median error "
           "(1 would be tight). Proposition 1 needs the embedding distortion eps <= 1/2, so it only appears at larger m. "
           "The three sketch types give nearly the same curves, so the lines overlap. Median over datasets; "
           "Theorem 1 uses the empirical 95th-percentile eps, eta = 0.1.")
    finish(fig, out, "fig2_m_sweep.png",
           f"Metric error vs sketch size m, and slack of the bounds (kappa about {kappa_main:g})", cap, hs, ncol=4)
 
 
def fig_kappa(res, out, m):
    tags = sorted({r["tag"] for r in res})
    fig, axs = plt.subplots(1, 4, figsize=(17, 5.6))
    for j, met in enumerate(METS):
        ax = axs[j]
        ref = None
        for kind in KINDS:
            xs, yt, yp = [], [], []
            for tg in tags:
                rows = [r for r in res if r["kind"] == kind and r["tag"] == tg]
                xs.append(med(rows, "kappa"))
                yt.append(float(np.nanmedian([r["thm_" + met] / r[met + "_q50"] for r in rows])))
                yp.append(float(np.nanmedian([r["prop_" + met] / r[met + "_q50"] for r in rows])))
            ax.plot(xs, yt, "-", marker=MARK[kind], color=COL[kind], ms=6)
            ax.plot(xs, yp, "--", marker=MARK[kind], color=COL[kind], ms=4, alpha=0.7)
            if kind == "gauss":
                ref = (np.array(xs), yt[0])
        pw = (1, 2, 4)[j]
        ax.plot(ref[0], 2 * ref[1] * (ref[0] / ref[0][0]) ** pw, "k:", lw=1.3, label=rf"$\propto\kappa^{pw}$ (guide)")
        ax.set_xscale("log"); ax.set_yscale("log"); logfmt(ax)
        ax.set_xlabel(r"condition number $\kappa(A)$"); ax.set_ylabel("bound / simulated median")
        ax.set_title(NAME[met]); ax.legend(loc="upper left")
    ax = axs[3]
    for kind in KINDS:
        xs = [med([r for r in res if r["kind"] == kind and r["tag"] == tg], "kappa") for tg in tags]
        cs = [med([r for r in res if r["kind"] == kind and r["tag"] == tg], "c") for tg in tags]
        ax.plot(xs, cs, "-", marker=MARK[kind], color=COL[kind], ms=6)
    ax.plot(xs, np.array(xs) * 0.6, "k:", lw=1.3, label=r"$\propto\kappa$ (guide)")
    ax.set_xscale("log"); ax.set_yscale("log"); logfmt(ax)
    ax.set_xlabel(r"condition number $\kappa(A)$"); ax.set_ylabel(r"$c=\kappa\,\epsilon/(1-\epsilon)$")
    ax.set_title("Lemma 1 constant inside Theorem 1"); ax.legend(loc="upper left")
    hs = [Line2D([], [], color="k", lw=2, label="Theorem 1"),
          Line2D([], [], color="k", ls="--", lw=2, label="Proposition 1")] + kind_handles()
    cap = ("Each point is one condition number (median over datasets, sketch size m = %d). Panels 1-3: bound divided by the simulated "
           "median error, so 1 would be tight; solid = Theorem 1 (contains kappa through c), dashed = Proposition 1 (kappa enters only "
           "through sigma_min). Dotted: a power of kappa for reference. Panel 4: the constant c that Theorem 1 uses. "
           "The sketch types nearly coincide." % m)
    finish(fig, out, "fig3_kappa_sweep.png", "How the looseness of the bounds grows with the condition number", cap, hs, ncol=5)
 
 
def fig_slack(res, out, delta):
    rows_all = [r for r in res if r["m"] == M_REF]
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.6))
    mincov = 1.0
    for ax, met in zip(axs, METS):
        for kind in KINDS:
            rows = [r for r in rows_all if r["kind"] == kind]
            slack = np.nanmedian([np.array(r["thmeta_" + met]) / np.array(r["qeta_" + met]) for r in rows], axis=0)
            ax.plot(ETAS, slack, "-", marker=MARK[kind], color=COL[kind], ms=6)
            mincov = min(mincov, float(np.nanmin([r["cov_" + met] for r in rows])))
        ax.axhline(1, color="k", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log"); xticks_at(ax, ETAS); logfmt(ax, "y")
        ax.set_xlabel(r"$\eta$  (bound holds with probability $\geq 1-\eta-\delta$)")
        ax.set_ylabel("Theorem 1 bound / simulated quantile"); ax.set_title(NAME[met])
    cap = (f"y = Theorem 1 bound at confidence 1-eta-delta (delta = {delta}) divided by the simulated (1-eta-delta) quantile of the error, "
           f"median over datasets, m = {M_REF}. Values below the black line (1) would be violations. Lowest empirical coverage over all "
           f"cells and eta: {mincov:.3f}. The slack grows as eta shrinks because the bound scales like 1/sqrt(eta) "
           "(1/eta for the squared terms).")
    finish(fig, out, "fig4_slack.png", "How loose Theorem 1 is at each confidence level", cap, kind_handles(), ncol=3)
 
 
def fig_phase(rows, out, m):
    fr = sorted({r["n0frac"] for r in rows})
    nr = sorted({r["noise_ratio"] for r in rows})
    allR = [r["R_pred"] for r in rows] + [r["R_meas"] for r in rows]
    norm = TwoSlopeNorm(vmin=min(0.85, min(allR)), vcenter=1.0, vmax=max(1.15, max(allR)))
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for j, kind in enumerate(KINDS):
        for i, (key, nm) in enumerate((("R_pred", "formula"), ("R_meas", "Monte Carlo"))):
            ax = axs[i, j]
            Z = np.full((len(nr), len(fr)), np.nan)
            for r in rows:
                if r["kind"] == kind:
                    Z[nr.index(r["noise_ratio"]), fr.index(r["n0frac"])] = r[key]
            im = ax.imshow(Z, origin="lower", cmap="RdBu", norm=norm, aspect="auto")
            for a in range(len(nr)):
                for b in range(len(fr)):
                    ax.text(b, a, f"{Z[a, b]:.2f}", ha="center", va="center", fontsize=9.5)
            ax.set_xticks(range(len(fr))); ax.set_xticklabels([f"{x:g}" for x in fr])
            ax.set_yticks(range(len(nr))); ax.set_yticklabels([f"{x:g}" for x in nr])
            ax.grid(False)
            ax.set_title(f"{LABEL[kind]}: {nm}")
            if i == 1: ax.set_xlabel(r"group-0 share of rows, $n_0/n$")
            if j == 0: ax.set_ylabel("noise sd, group 0 / group 1")
            fig.colorbar(im, ax=ax, shrink=0.85, label="R")
    se = float(np.median([r["R_se"] for r in rows]))
    cap = ("R = error of the vanilla sketch / error of the oracle-allocated stratified sketch, where error is E||C^T E r||^2 with C = Pi "
           "(the IF target). R > 1 (blue): stratification with the oracle allocation helps; R < 1 (red): it hurts; white = tie. "
           f"Top row: closed-form formulas. Bottom row: Monte Carlo with {rows[0]['T']} draws per sketch; a measured R has a typical "
           f"standard error of about {se:.2f}, so differences of that size between rows are noise. m = {m}.")
    finish(fig, out, "fig5_phase.png", "When does oracle stratification beat a vanilla sketch? (group imbalance vs group noise)", cap)
 
 
def fig_objectives(rows, out):
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2))
    allv = []
    for kind in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        axs[0].scatter([r["V_pred"] for r in rs], [r["V_meas"] for r in rs], s=26, marker="o", color=COL[kind],
                       alpha=0.75, edgecolor="none")
        axs[0].scatter([r["O_pred"] for r in rs], [r["O_meas"] for r in rs], s=30, marker="^", color=COL[kind],
                       alpha=0.75, edgecolor="none")
        axs[1].errorbar([r["R_pred"] for r in rs], [r["R_meas"] for r in rs], yerr=[2 * r["R_se"] for r in rs],
                        fmt=MARK[kind], ms=4, color=COL[kind], alpha=0.7, elinewidth=0.8)
        allv += [r[k] for r in rs for k in ("V_pred", "V_meas", "O_pred", "O_meas")]
    lo, hi = min(allv) * 0.8, max(allv) * 1.25
    axs[0].plot([lo, hi], [lo, hi], "k-", lw=1)
    axs[0].set_xlim(lo, hi); axs[0].set_ylim(lo, hi)
    axs[0].set_xscale("log"); axs[0].set_yscale("log"); logfmt(axs[0])
    axs[0].set_xlabel("formula"); axs[0].set_ylabel("Monte Carlo mean")
    axs[0].set_title(r"objective $E\|C^\top E r\|^2$")
    Rall = [r["R_pred"] for r in rows] + [r["R_meas"] for r in rows]
    lo, hi = min(Rall) - 0.1, max(Rall) + 0.1
    axs[1].plot([lo, hi], [lo, hi], "k-", lw=1)
    axs[1].axhline(1, color="gray", ls=":"); axs[1].axvline(1, color="gray", ls=":")
    axs[1].set_xlim(lo, hi); axs[1].set_ylim(lo, hi)
    axs[1].set_xlabel("R from formula"); axs[1].set_ylabel("R from Monte Carlo")
    axs[1].set_title("ratio R = vanilla / oracle-stratified")
    frac = np.mean([abs(r["R_meas"] - r["R_pred"]) <= 2 * r["R_se"] for r in rows])
    axs[1].text(0.04, 0.96, f"{100 * frac:.0f}% of points within\n2 standard errors of the diagonal",
                transform=axs[1].transAxes, va="top", fontsize=9.5)
    hs = kind_handles() + [Line2D([], [], color="gray", marker="o", ls="", label="vanilla sketch"),
                           Line2D([], [], color="gray", marker="^", ls="", label="oracle stratified")]
    cap = ("One point per grid cell of the previous figure. Left: predicted vs Monte Carlo value of the sketching objective for vanilla "
           "sketches (circles) and oracle-allocated stratified sketches (triangles). Right: the ratio R of the two, with error bars of "
           "2 standard errors from the Monte Carlo; points on the diagonal mean the formula is right.")
    finish(fig, out, "fig6_objectives.png", "Check of the stratified-objective formulas", cap, hs, ncol=5)
 
 
def qband(x):
    x = np.asarray(x, float)
    if x.size == 0 or not np.isfinite(x).any():
        return (np.nan,) * 3
    return tuple(np.nanquantile(x, [0.05, 0.5, 0.95]))
 
 
def fig_proxy(rows, out, note):
    mps = sorted({r["mp"] for r in rows})
    fin = float(np.mean([np.isfinite(r["ba"]) for r in rows]))
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.5))
    for j, kind in enumerate(KINDS):
        rk = [r for r in rows if r["kind"] == kind]
        by = {mp: [r for r in rk if r["mp"] == mp] for mp in mps}
        ax = axs[0, j]
        q = np.array([qband([r["exc"] for r in by[mp]]) for mp in mps])
        ax.fill_between(mps, q[:, 0], q[:, 2], color=COL[kind], alpha=0.2, lw=0)
        ax.plot(mps, q[:, 1], color=COL[kind], lw=2.4)
        ax.plot(mps, [np.nanmedian([r["br"] for r in by[mp]]) for mp in mps], "k--", lw=1.8)
        if fin > 0:
            ax.plot(mps, [np.nanmedian([r["ba"] for r in by[mp]]) for mp in mps], color="gray", ls="-.", lw=1.8)
        ax.set_xscale("log"); ax.set_yscale("log"); xticks_at(ax, mps); logfmt(ax, "y")
        ax.set_title(LABEL[kind]); ax.set_xlabel("pilot size")
        if j == 0: ax.set_ylabel("excess over oracle: f(t_hat)/f(t*) - 1")
        ax = axs[1, j]
        gains = [[(1 - r["ratio"]) / (1 - r["orc"]) for r in by[mp] if r["orc"] < 1] for mp in mps]
        q = np.array([qband(g) for g in gains])
        ax.fill_between(mps, q[:, 0], q[:, 2], color=COL[kind], alpha=0.2, lw=0)
        ax.plot(mps, q[:, 1], color=COL[kind], lw=2.4)
        ax.axhline(1, color="k", lw=1); ax.axhline(0, color="gray", lw=1)
        ax.set_xscale("log"); xticks_at(ax, mps); ax.set_ylim(-0.25, 1.15)
        pb = float(np.mean([r["ratio"] < 1 for r in rk]))
        ax.text(0.97, 0.05, f"proxy beats vanilla in {100 * pb:.0f}% of runs", transform=ax.transAxes,
                ha="right", fontsize=9.5)
        ax.set_xlabel("pilot size")
        if j == 0: ax.set_ylabel("share of oracle's gain kept")
    hs = [Line2D([], [], color="k", lw=2.4, label="realised median (band: 5-95%)"),
          Line2D([], [], color="k", ls="--", lw=1.8, label="Theorem F bound from realised plug-in error")]
    if fin > 0:
        hs.append(Line2D([], [], color="gray", ls="-.", lw=1.8, label="a-priori Theorem F bound"))
    apr = (f"The a-priori bound (needs pilot embedding eps <= 1/2 and kappa < 1) is finite in {100 * fin:.0f}% of runs."
           if fin > 0 else "The a-priori bound (needs pilot embedding eps <= 1/2 and kappa < 1) was vacuous in every run, so it is not drawn.")
    cap = ("Top: how far the proxy allocation's error is above the oracle's (log scale), for a pilot CountSketch of the given size, against "
           "the Theorem F bound computed from the realised plug-in error. " + apr + " Bottom: share of the oracle's gain over the vanilla sketch "
           "that the proxy keeps, (1 - proxy/vanilla) / (1 - oracle/vanilla): 1 = as good as the oracle, 0 = no better than vanilla. "
           "The pilot's own cost is not charged. Final sketch uses the continuous split; no rounding.")
    finish(fig, out, "fig7_proxy.png", "Proxy allocation: cost of estimating the residual from a pilot sketch" + note, cap, hs, ncol=3)
 
 
# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def pmap(fn, items, procs):
    if procs <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(procs) as ex:
        return list(ex.map(fn, items, chunksize=1))
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fairsketch_out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--nds", type=int, default=None, help="datasets (seeds) per configuration")
    ap.add_argument("--T", type=int, default=None, help="sketch draws per configuration")
    ap.add_argument("--only", default="msweep,kappa,phase,proxy")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--kappa-main", type=float, default=10.0)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.05)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nds = a.nds or (2 if a.quick else 5)
    T = a.T or (100 if a.quick else 300)
    Tph = 200 if a.quick else 1000
    reps = 30 if a.quick else 100
    only = set(a.only.split(","))
    allres = {}
    t0 = time.time()
 
    def cfgs(kappas, ms, tagf):
        out = []
        for kp in kappas:
            for m in ms:
                for kind in KINDS:
                    for s in range(nds):
                        out.append(dict(kind=kind, m=m, T=T, eta=a.eta, delta=a.delta, tag=tagf(kp, m),
                                        data=dict(n=a.n, d=a.d, kappa=kp, seed=s)))
        return out
 
    ms_res, kp_res = [], []
    if "msweep" in only:
        ms_res = pmap(run_cfg, cfgs([a.kappa_main], M_LIST, lambda kp, m: m), a.procs)
        allres["msweep"] = ms_res
        fig_msweep(ms_res, a.out, a.kappa_main)
        fig_slack(ms_res, a.out, a.delta)
        print(f"[msweep] {time.time() - t0:.0f}s")
    if "kappa" in only:
        kp_res = pmap(run_cfg, cfgs(KAPPAS, [M_REF], lambda kp, m: kp), a.procs)
        allres["kappa"] = kp_res
        fig_kappa(kp_res, a.out, M_REF)
        print(f"[kappa] {time.time() - t0:.0f}s")
    if ms_res or kp_res:
        fig_moments(ms_res + kp_res, a.out)
        both = ms_res + kp_res
        print("Deterministic checks over all sketch draws:")
        print("  Prop 1 violations :", sum(r["viol_prop"] for r in both), "(expected 0)")
        print("  Lemma 1 violations:", sum(r["viol_lem1"] for r in both), "(expected 0)")
        print("  draws checked     :", sum(r["n_draws"] for r in both))
        rr = [r["mT_sim"] / r["mT_th"] for r in both]
        print(f"  Lemma 2 sim/theory for tr Sigma: median {np.median(rr):.3f}, "
              f"range [{min(rr):.3f}, {max(rr):.3f}]")
 
    if "phase" in only:
        fr = [0.1, 0.25, 0.5] if a.quick else [0.05, 0.1, 0.2, 0.35, 0.5]
        nrs = [0.25, 1, 4] if a.quick else [0.25, 0.5, 1, 2, 4]
        m_ph = 200
        items = [dict(m=m_ph, T=Tph, target="IF",
                      data=dict(n=min(a.n, 600), d=a.d, kappa=a.kappa_main, seed=1, n0frac=f, noise_ratio=nr))
                 for f in fr for nr in nrs]
        rows = [r for rs in pmap(run_phase_cell, items, a.procs) for r in rs]
        allres["phase"] = rows
        fig_phase(rows, a.out, m_ph)
        fig_objectives(rows, a.out)
        print(f"[phase] {time.time() - t0:.0f}s")
 
    if "proxy" in only:
        mps = [20, 40, 80, 160, 320, 640]
        items = [dict(mps=mps, reps=reps,
                      data=dict(n=a.n, d=a.d, kappa=a.kappa_main, seed=s, n0frac=0.2, noise_ratio=3.0))
                 for s in range(nds)]
        rows = [r for rs in pmap(run_proxy, items, a.procs) for r in rs]
        allres["proxy"] = rows
        fig_proxy(rows, a.out, f" (n0/n=0.2, noise ratio 3, kappa~{a.kappa_main:g})")
        v1 = sum(1 for r in rows if r["kr"] < 1 and r["exc"] > r["br"] * (1 + 1e-9))
        v2 = sum(1 for r in rows if np.isfinite(r["ka"]) and r["kr"] > r["ka"] * (1 + 1e-9))
        print("Proxy checks: excess > realised-kappa bound:", v1, "| realised kappa > a-priori kappa:", v2,
              "(both expected 0)")
        for kind in KINDS:
            rk = [r for r in rows if r["kind"] == kind]
            for mp in mps:
                x = [r for r in rk if r["mp"] == mp]
                print(f"  {kind:5s} m_pilot={mp:4d}  P(proxy beats vanilla)="
                      f"{np.mean([r['ratio'] < 1 for r in x]):.2f}  median excess={np.median([r['exc'] for r in x]):.3f}")
        print(f"[proxy] {time.time() - t0:.0f}s")
 
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(allres, f, default=float)
    print("done; raw numbers in", os.path.join(a.out, "results.json"))
 
 
if __name__ == "__main__":
    main()