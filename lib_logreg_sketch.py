"""LogReg-sketch (Munteanu, Omlor, Woodruff 2021), following the authors' code."""

import numpy as np
import scipy.optimize as so

sig = lambda t: 0.5 * (1 + np.tanh(0.5 * t))


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
