"""Section 5, classification: the targeted coreset of the paper (Algorithms "oracle" and "proxy"), one-shot.
A sketch keeps a weighted sample of rows and the model is refit once on it; there are no Newton iterations.

Score of point i for a direction u (paper, Section 5):  kappa_i = (ytil_i - p_i) z_i^T H^{-1} u,  ytil = (1+y)/2, p_i = sigma(z_i^T x).
Directions (paper, Table "cls-targets"): IF: x/||x||^2;  DP: c_0 - c_1 with c_g = mean over G_g of sigma'(z_i^T x) z_i;
TPR / FPR: the same over the label subgroups.
target = "all" (default) uses the four directions at once (Theorem "targeted sampling" (d): q_i ~ ||kappa_i||), with each rate-gap
direction divided by max(|gap|, GAP_FLOOR) so that every coordinate is a relative change, like the plotted metrics.
target = "IF" | "DP" | "TPR" | "FPR" uses that one direction.
Probabilities are mixed with uniform: q = (1-mu) |kappa|/sum|kappa| + mu/n.

Methods (run_coreset):
  c_unif            uniform sample, m rows
  c_strat_unif      uniform within each group, Neyman split of the uniform beta_g
  c_imp             Section 3 importance sampling, p.q_imp (needs x^)
  c_tgt_oracle      targeted, scores at x^ (needs x^)
  c_tgt_proxy       targeted, scores from a uniform pilot of m_p rows; main sample has m - m_p rows (the pilot is charged to the budget).
                    The pilot rows are kept in the final fit (p.reuse_pilot, default True): weights f*w_pilot + (1-f)*w_main with
                    f = m_p/m, so E[W] = I and m E[(u'Delta')^2] = f V(uniform) + (1-f) E V(q_hat) to leading order.
  c_strat_tgt_oracle / c_strat_tgt_proxy   the same probabilities within each group, split m_g ~ sqrt(beta_g) (Theorem "stratifying never hurts")
  logreg_code       LogReg-sketch, authors' code version (lib_logreg_sketch); logreg_code_ky: with the Ky Fan clipped loss
  newton_cs / newton_gauss   Newton sketch reference, p.T steps (lib_classification.run)
Problem attributes read if present: p.target, p.mu, p.m_p (pilot rows; default m // 4), p.reuse_pilot (default True)."""
import numpy as np
import lib_classification as cl
from lib_classification import sig, LAB, COL, fit
from lib_logreg_sketch import fit_logreg_sketch_code

GAP_FLOOR = 0.01
for k, (lab, col) in {"c_unif": ("uniform", "tab:blue"), "c_strat_unif": ("stratified uniform", "tab:cyan"),
                      "c_imp": ("importance, Sec. 3 (oracle)", "tab:gray"),
                      "c_tgt_oracle": ("targeted (oracle)", "tab:red"), "c_tgt_proxy": ("targeted (proxy)", "tab:pink"),
                      "c_strat_tgt_oracle": ("stratified targeted (oracle)", "darkred"),
                      "c_strat_tgt_proxy": ("stratified targeted (proxy)", "tab:purple"),
                      "logreg_code": ("LogReg-sketch (Munteanu et al. 2021)", "tab:olive"),
                      "logreg_code_ky": ("LogReg-sketch, clipped", "tab:brown")}.items():
    LAB[k], COL[k] = lab, col
S5C_METH = ["c_unif", "c_imp", "c_tgt_oracle", "c_tgt_proxy", "c_strat_tgt_proxy", "logreg_code", "newton_cs"]
S5C_ALL = S5C_METH + ["c_strat_unif", "c_strat_tgt_oracle", "logreg_code_ky", "newton_gauss"]


def directions(p, x, target="all"):
    """d x r matrix of directions at x (columns in the order of the metrics used)."""
    pr = sig(p.Z @ x)
    dd = pr * (1 - pr)
    cols = []
    for k in (["IF", "DP", "TPR", "FPR"] if target == "all" else [target]):
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
    """n-vector ||kappa_i|| (one direction: |kappa_i|) and the n x r score matrix kappa."""
    K = ((p.y > 0) - sig(p.Z @ x))[:, None] * (p.Z @ np.linalg.solve(H, directions(p, x, target)))
    return np.linalg.norm(K, axis=1), K


def _mix(a, mu):
    return (1 - mu) * a / a.sum() + mu / len(a)


def _draw(q, m, rng):
    """Importance sketch with m draws: distinct indices and weights K_i/(m q_i)."""
    Kc = rng.multinomial(m, q)
    I = np.nonzero(Kc)[0]
    return I, Kc[I] / (m * q[I])


def _pilot(p, m_p, rng):
    """Uniform pilot: its fit x_p and the pilot Hessian (pilot rows only)."""
    I, w = _draw(np.full(p.n, 1.0 / p.n), m_p, rng)
    xp = fit(p.A[I], w, p.lam)
    s = sig(p.A[I] @ xp)
    Hh = (p.Z[I] * (w * s * (1 - s))[:, None]).T @ p.Z[I] + p.lam * np.eye(p.d)
    return xp, Hh, I, w


def _strat(p, a, K, m, mu, rng):
    """Stratified importance sketch: within group g, q^(g) = mix of a restricted to G_g (a = ||kappa_i||, or ones with mu = 1 for
    stratified uniform); split m_g ~ sqrt(beta_g),
    beta_g = sum_{G_g} ||kappa_i||^2 / q^(g)_i - ||sum_{G_g} kappa_i||^2. Returns (indices, weights)."""
    groups = [np.nonzero(p.g == j)[0] for j in (0, 1)]
    qs, beta = [], []
    for idx in groups:
        q = _mix(a[idx] + 1e-300, mu)
        qs.append(q)
        beta.append(max(np.sum(np.sum(K[idx] ** 2, 1) / q) - np.sum(np.sum(K[idx], 0) ** 2), 0.0))
    sb = np.sqrt(beta)
    m0 = int(np.clip(round(m * sb[0] / sb.sum()) if sb.sum() > 0 else round(m * len(groups[0]) / p.n), 2, m - 2))
    I, W = [], []
    for idx, q, mg in zip(groups, qs, (m0, m - m0)):
        Ig, wg = _draw(q, mg, rng)
        I.append(idx[Ig])
        W.append(wg)
    return np.concatenate(I), np.concatenate(W)


def run_coreset(p, meth, m, rng, T=None):
    """One draw of a Section 5 classification method. Returns the coefficient vector."""
    if meth.startswith("newton"):
        return cl.run(p, meth, m, rng) if T is None else cl.run(p, meth, m, rng, T=T)
    if meth in ("logreg_code", "logreg_code_ky"):
        return fit_logreg_sketch_code(p.A, m, rng, p.lam, clip=0.25 if meth.endswith("ky") else None)
    target, mu = getattr(p, "target", "all"), getattr(p, "mu", 0.2)
    if meth == "c_unif":
        I, w = _draw(np.full(p.n, 1.0 / p.n), m, rng)
    elif meth == "c_imp":
        I, w = _draw(p.q_imp, m, rng)
    elif meth == "c_strat_unif":
        a, K = scores(p, p.xh, p.H, target)
        I, w = _strat(p, np.ones(p.n), K, m, 1.0, rng)        # mu = 1: uniform within groups; beta_g uses the oracle scores
    else:
        proxy = meth.endswith("proxy")
        mm = m
        if proxy:
            m_p = getattr(p, "m_p", None) or m // 4
            mm = m - m_p
            xp, Hh, Ip, wp = _pilot(p, m_p, rng)
            a, K = scores(p, xp, Hh, target)
        else:
            a, K = scores(p, p.xh, p.H, target)
        if meth.startswith("c_strat"):
            I, w = _strat(p, a, K, mm, mu, rng)
        else:
            I, w = _draw(_mix(a, mu), mm, rng)
        if proxy and getattr(p, "reuse_pilot", True):
            f = m_p / m
            I, w = np.concatenate([Ip, I]), np.concatenate([f * wp, (1 - f) * w])   # repeated indices just add in fit()
    return fit(p.A[I], w, p.lam)
