
import numpy as np
from lib_classification import sig, LAB, COL

for k, (lab, col) in {"vanilla_cs": ("vanilla: Newton-Hessian, CountSketch", "tab:orange"),
                      "vanilla_gauss": ("vanilla: Newton-Hessian, Gaussian", "tab:green"),
                      "oracle": ("oracle (stratified IS)", "tab:red"), "proxy": ("proxy (stratified IS)", "tab:purple")}.items():
    LAB[k], COL[k] = lab, col
S5_METH = ["vanilla_cs", "vanilla_gauss", "oracle", "proxy"]
DEFAULT_T = 3          # Newton steps used for the Section 5 comparison (the iteration converges to the exact model, so the comparison is transient)


def _cs(B, m, rng):
    SB = np.zeros((m, B.shape[1]))
    np.add.at(SB, rng.integers(0, m, B.shape[0]), B * rng.choice([-1.0, 1.0], B.shape[0])[:, None])
    return SB.T @ SB


def _gauss(B, m, rng):
    SB = rng.standard_normal((m, B.shape[0])) @ B / np.sqrt(m)
    return SB.T @ SB


def _strat(B, u, h, Hinv, groups, m, gamma, rng):
    d = B.shape[1]
    qs, taus = [], []
    for idx in groups:
        sc = np.abs(u[idx]) * np.sqrt(h[idx]) + 1e-300
        q = (1 - gamma) * sc / sc.sum() + gamma * (h[idx] + 1e-300) / (h[idx] + 1e-300).sum()
        q /= q.sum()
        v = B[idx].T @ u[idx]
        taus.append(max(np.sum(u[idx] ** 2 * h[idx] / q) - v @ Hinv @ v, 1e-30))
        qs.append(q)
    w = np.sqrt(taus)
    mg = np.maximum(2, np.round(m * w / w.sum())).astype(int)
    mg[np.argmax(mg)] += m - mg.sum()
    Hs = np.zeros((d, d))
    for idx, q, k in zip(groups, qs, mg):
        s = rng.choice(len(idx), size=int(k), p=q)
        Bi = B[idx[s]]
        Hs += (Bi / (k * q[s])[:, None]).T @ Bi
    return Hs


def run_newton(p, meth, m, rng, T=None, gamma=0.2):
    T = T if T is not None else getattr(p, "T", DEFAULT_T)
    n, d, lam = p.n, p.d, p.lam
    groups = [np.where(p.g == 0)[0], np.where(p.g == 1)[0]]
    x = np.zeros(d)
    Ht_inv = None
    for _ in range(T):
        s = sig(-(p.A @ x))
        gr = -(p.A.T @ s) + lam * x
        B = p.A * np.sqrt(s * (1 - s))[:, None]
        if meth == "vanilla_cs" or (meth == "proxy" and Ht_inv is None):
            Hs = _cs(B, m, rng)
        elif meth == "vanilla_gauss":
            Hs = _gauss(B, m, rng)
        else:
            if meth == "oracle":
                Hi = np.linalg.inv(B.T @ B + lam * np.eye(d))
            else:
                Hi = Ht_inv
            u = B @ (Hi @ gr)
            h = ((B @ Hi) * B).sum(1)
            Hs = _strat(B, u, h, Hi, groups, m, gamma, rng)
        Ht = Hs + lam * np.eye(d)
        Ht_inv = np.linalg.inv(Ht)
        x = x - Ht_inv @ gr
    return x