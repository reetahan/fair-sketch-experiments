#!/usr/bin/env python3
"""Section 5 classification: where the targeted proxy loses the oracle's gain, and two attempted fixes."""
import os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lib_classification as cl
from lib_classification import sig, fit
from lib_classification import _draw, _mix, _pilot, imp_probs, run_coreset, directions
from lib_classification import fit_logreg_sketch_code

M_FACTORS = [10, 40, 160]
REFS = {"unif": "c_unif", "lev": "c_lev", "imp_base": "c_imp_proxy", "oracle": "c_tgt_oracle", "tgt_base": "c_tgt_proxy"}
SHRINK = ["tgt_floor03", "tgt_floor10", "tgt_adaptive", "tgt_exactgap"]
EXACT = ["tgt_exactres", "tgt_exactH", "tgt_exactdir", "tgt_exactall"]
PILOTS = {"naive": "naive_cs", "logreg": "logreg", "newton": "newton_cs"}
ALPHA = ["tgt_alpha", "tgt_alphatrue", "imp_alpha", "imp_alphatrue", "exact_alpha"]
METHODS = list(REFS) + SHRINK + EXACT + ALPHA + [f"{r}_{k}" for r in ("tgt", "imp") for k in PILOTS]


def shrunk_directions(p, x, denom):
    pr = sig(p.Z @ x)
    dd = pr * (1 - pr)
    cols = [x / (x @ x)]
    for k in ("DP", "TPR", "FPR"):
        r0, r1 = p.sets[k]
        u = (dd[r0] @ p.Z[r0]) / r0.sum() - (dd[r1] @ p.Z[r1]) / r1.sum()
        cols.append(u / denom(k, abs(pr[r0].mean() - pr[r1].mean())))
    return np.stack(cols, 1)


def kappa(p, x_res, H, U):
    return ((p.y > 0) - sig(p.Z @ x_res))[:, None] * (p.Z @ np.linalg.solve(H, U))


def targeted_probs(p, x_res, H, U):
    return _mix(np.linalg.norm(kappa(p, x_res, H, U), axis=1), p.mu)


def alpha_run(p, meth, m, rng):
    m_p = p.m_p
    xp, Hh, Ip, wp = _pilot(p, m_p, rng)
    Ke = kappa(p, xp, Hh, directions(p, xp))
    if meth.startswith("imp"):
        q = imp_probs(p, xp, Hh)
    elif meth.startswith("exact"):
        q = _mix(np.linalg.norm(p.K_exact, axis=1), p.mu)
    else:
        q = _mix(np.linalg.norm(Ke, axis=1), p.mu)
    K = Ke if meth in ("tgt_alpha", "imp_alpha") else p.K_exact
    f = m_p / m
    vu, vq = cl.objective(K, np.full(len(q), 1.0 / len(q))), max(cl.objective(K, q), 1e-300)
    alpha = (f / vu) / (f / vu + (1 - f) / vq)
    p.alphas.setdefault(meth, []).append(alpha)
    I, w = _draw(q, m - m_p, rng)
    return fit(p.A[np.concatenate([Ip, I])], np.concatenate([alpha * wp, (1 - alpha) * w]), p.lam)


def sketched_hessian(p, x, m_p, rng):
    s = sig(-(p.A @ x))
    B = p.A * np.sqrt(s * (1 - s))[:, None]
    SB = np.zeros((m_p, p.d))
    np.add.at(SB, rng.integers(0, m_p, p.n), B * rng.choice([-1.0, 1.0], p.n)[:, None])
    return SB.T @ SB + p.lam * np.eye(p.d)


def all_rows_pilot(p, kind, m_p, rng):
    if kind == "logreg":
        x = fit_logreg_sketch_code(p.A, m_p, rng, p.lam)
    else:
        x = cl.run(p, PILOTS[kind], m_p, rng, T=2)
    return x, sketched_hessian(p, x, m_p, rng)


def exact_gaps(p):
    pr = sig(p.Z @ p.xh)
    return {k: abs(pr[r0].mean() - pr[r1].mean()) for k, (r0, r1) in p.sets.items()}


def runner(p, meth, m, rng):
    if meth in REFS:
        return run_coreset(p, REFS[meth], m, rng)
    if meth in ALPHA:
        return alpha_run(p, meth, m, rng)
    m_p = p.m_p
    if meth.endswith(tuple(PILOTS)):
        rule, kind = meth.split("_")
        xp, Hh = all_rows_pilot(p, kind, m_p, rng)
        q = imp_probs(p, xp, Hh) if rule == "imp" else targeted_probs(p, xp, Hh, directions(p, xp))
        I, w = _draw(q, m - m_p, rng)
        return fit(p.A[I], w, p.lam)
    xp, Hh, Ip, wp = _pilot(p, m_p, rng)
    x_res, H, U = xp, Hh, None
    if meth == "tgt_exactres":
        x_res = p.xh
    elif meth == "tgt_exactH":
        H = p.H
    elif meth == "tgt_exactdir":
        U = directions(p, p.xh)
    elif meth == "tgt_exactall":
        x_res, H, U = p.xh, p.H, directions(p, p.xh)
    elif meth == "tgt_floor03":
        U = shrunk_directions(p, xp, lambda k, g: max(g, 0.03))
    elif meth == "tgt_floor10":
        U = shrunk_directions(p, xp, lambda k, g: max(g, 0.10))
    elif meth == "tgt_adaptive":
        U = shrunk_directions(p, xp, lambda k, g: np.sqrt(g * g + 1.0 / m_p))
    elif meth == "tgt_exactgap":
        ex = p.exact_gaps
        U = shrunk_directions(p, xp, lambda k, g: max(ex[k], cl.GAP_FLOOR))
    U = directions(p, xp) if U is None else U
    I, w = _draw(targeted_probs(p, x_res, H, U), m - m_p, rng)
    f = m_p / m
    return fit(p.A[np.concatenate([Ip, I])], np.concatenate([f * wp, (1 - f) * w]), p.lam)


def sweep(p, reps, rng, fracs, factors, meths):
    p.exact_gaps = exact_gaps(p)
    p.mu = 0.5
    p.K_exact = kappa(p, p.xh, p.H, directions(p, p.xh))
    frames = []
    for m in [k * p.d for k in factors if k * p.d < p.n // 4]:
        for f in fracs:
            p.m_p = max(int(round(f * m)), 1)
            p.alphas = {}
            d = cl.experiment(p, m, reps, rng, meths=meths, runner=runner).assign(f=f, mu=p.mu)
            d["alpha_mean"] = d.method.map({k: float(np.mean(v)) for k, v in p.alphas.items()})
            frames.append(d)
            print(f"  m={m} f={f}: done", flush=True)
    return pd.concat(frames)


def main():
    ap = cl.ext_args("results/fix_s5c")
    ap.add_argument("--fracs", default="0.25,0.5")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    rng = cl.start(a)
    fracs = [float(v) for v in a.fracs.split(",")]
    factors = [10, 40] if a.quick else M_FACTORS
    probs = cl.ext_problems(a)
    t0 = time.time()
    for p in probs:
        print(f"[{p.name}] n={p.n} d={p.d}", flush=True)
        name = "synth" if a.synthetic else p.name
        sweep(p, a.reps, rng, fracs, factors, a.methods.split(",")).to_csv(f"{a.out}/s5c_fix_{name}.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s; outputs in {a.out}/")


if __name__ == "__main__":
    main()
