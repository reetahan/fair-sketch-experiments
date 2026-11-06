"""Section 5 classification: one-shot targeted coresets (oracle, proxy, stratified)."""

import numpy as np
import lib_classification as cl
from lib_classification import sig, LAB, COL, fit
from lib_logreg_sketch import fit_logreg_sketch_code
from fs_common import PALETTE

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
        return cl.run(p, meth, m, rng) if T is None else cl.run(p, meth, m, rng, T=T)
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
