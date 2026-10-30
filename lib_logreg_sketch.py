"""LogReg-sketch of Munteanu, Omlor and Woodruff (ICML 2021), as implemented in the authors' code
(github.com/cxan96/oblivious-sketching-varreglogreg, sketching/sketch.py, experiments.py, optimizer.py), not as in the paper's text.

Differences from the paper version in lib_classification.logreg_sketch:
  * levels l = 0..h_max with base b = (n/N)^(1/h_max) and N = floor(m/(h_max+2)) buckets per level;
  * row j goes to level l with probability b^l/alpha, alpha = sum_l b^l, and to a uniform bucket at that level;
  * bucket rows are UNWEIGHTED sums of the (label-folded) data rows; the level weight alpha/b^l multiplies the LOSS term;
  * one extra block of k = m - N(h_max+1) rows drawn uniformly WITHOUT replacement, loss weight n/k;
  * all weights are normalised to sum to one (authors) -- here to sum to n, so that a ridge lam/2||x||^2 means the same as in the
    full objective sum_i ln(1+exp(-a_i^T x)) + lam/2||x||^2 (the authors use no ridge);
  * the optional clipped ("Ky Fan") loss keeps, in every level, only the kyfan*N buckets with the largest loss; the uniform block is
    always kept. The fit is L-BFGS from x = 0, as in the authors' optimizer.

The authors draw the level by an integer x ~ randint(0, alpha) and l = floor(log(x(b-1)+1)/log b); with a continuous x this is exactly
Pr(l) = b^l/alpha, which is what is used here."""
import numpy as np
import scipy.optimize as so

sig = lambda t: 0.5 * (1 + np.tanh(0.5 * t))


def logreg_sketch_code(A, m, rng, h_max=2):
    """A: n x d label-folded rows (a_i = y_i z_i). Returns (rows, weights, N): rows has N(h_max+1) bucket rows (level-major order)
    followed by the uniform block; weights sum to n."""
    n, d = A.shape
    N = max(int(m / (h_max + 2)), 1)
    b = (n / N) ** (1.0 / h_max)
    pl = b ** np.arange(h_max + 1, dtype=float)
    alpha = pl.sum()
    pl /= alpha
    lev = rng.choice(h_max + 1, n, p=pl)
    SA = np.zeros((N * (h_max + 1), d))
    np.add.at(SA, lev * N + rng.integers(0, N, n), A)
    w_sk = np.repeat(1.0 / pl, N)                        # = alpha / b^l for every bucket of level l
    k = max(m - N * (h_max + 1), 1)
    U = rng.choice(n, k, replace=False)
    rows = np.vstack([SA, A[U]])
    w = np.concatenate([w_sk, np.full(k, n / k)])
    return rows, w * (n / w.sum()), N


def _keep(t, N, K, L):
    """Indices kept by the clipped loss: in each of the L/N level blocks of the first L rows, the K rows with the largest loss
    (smallest margin t); every row after L is kept. Same selection as the authors' only_keep_k."""
    blocks = t[:L].reshape(-1, N)
    top = np.argpartition(blocks, K - 1, axis=1)[:, :K] + (np.arange(blocks.shape[0]) * N)[:, None]
    return np.concatenate([top.ravel(), np.arange(L, len(t))])


def fit_logreg_sketch_code(A, m, rng, lam, clip=None, h_max=2):
    """Fit on the authors' LogReg-sketch. clip=None: plain weighted loss; clip=0.25: the authors' Ky Fan loss with 25% kept per level."""
    rows, w, N = logreg_sketch_code(A, m, rng, h_max)
    L = N * (h_max + 1)
    K = max(int(N * clip), 1) if clip else None

    def f(x):
        t = rows @ x
        I = _keep(t, N, K, L) if (clip and K < N) else slice(None)
        tI, wI = t[I], w[I]
        return wI @ np.logaddexp(0, -tI) + 0.5 * lam * x @ x, -(rows[I].T @ (wI * sig(-tI))) + lam * x

    return so.minimize(f, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B").x
