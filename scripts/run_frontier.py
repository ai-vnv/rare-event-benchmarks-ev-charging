"""Robustness and cost of the Surabaya siting solutions under each estimator.

Feeders have the headroom of the one-day-in-ten-years Gaussian design (s = 3.46).  Each
estimator supplies the 11 x 11 matrix of joint feeder outage probabilities from the same
budget of 20,000 simulator runs; the siting MILP is solved for a sweep of the risk
weight, and every design is re-evaluated under the reference matrix.
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per process
import sys
import time
import warnings

import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cascade.features import feature_map  # noqa: E402
from cascade.model import D, NF, Cascade  # noqa: E402
from cascade.pipeline import fit_gbm, learned_proposal, outer_threshold, pilot, widen  # noqa: E402
from cascade.sampling import Mixture, build_reference  # noqa: E402
from cascade.siting import Case  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))   # set RESULTS_DIR for a scratch run
S_DESIGN = 3.46
N1 = N2 = 10_000
LAMS = [10.0 ** e for e in (7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5)]
SEEDS = 8


def join(q1, q2):
    return Mixture(np.vstack([q1.means, q2.means]), np.concatenate([q1.w, q2.w]) / 2,
                   np.concatenate([q1.taus, q2.taus]))


def joint(w, A, B=None):
    """Weighted second-moment matrix: mean_n w_n A_nj B_nk."""
    B = A if B is None else B
    return (A * w[:, None]).T @ B / len(w)


def cv_matrix(w, Dn, H, Gam):
    """Entry-wise control-variate estimate of E[w d_j d_k] with control w h_j h_k."""
    Q = np.zeros((NF, NF))
    for j in range(NF):
        for k in range(j, NF):
            y = w * Dn[:, j] * Dn[:, k]; c = w * H[:, j] * H[:, k]
            vc = c.var()
            beta = np.cov(y, c)[0, 1] / vc if vc > 0 else 1.0
            Q[j, k] = Q[k, j] = max(y.mean() - beta * (c.mean() - Gam[j, k]), 0.0)
    return Q


def outer_set(F1, y, active, M):
    gbm = fit_gbm(F1, y)
    kap = outer_threshold(F1, y, active)
    return lambda Z: gbm.decision_function(Z @ M) >= kap


def estimates(c, rng):
    """Joint outage matrices from MC, learned CV, learned IS and IS+CV at 20,000 runs each."""
    M = feature_map(c.rho)
    out = {}
    X = rng.standard_normal((N1 + N2, D)); Dn = c.simulate(X).astype(float)
    out["mc"] = joint(np.ones(len(X)), Dn)
    # pilot aimed at days with two or more feeders down
    X1, _, active = pilot(c, rng, N1, M, event=lambda Z: c.simulate(Z).sum(1) >= 2)
    D1 = c.simulate(X1); k1 = D1.sum(1); F1 = X1 @ M
    q = join(widen(learned_proposal(outer_set(F1, k1 >= 1, active, M), rng)),
             widen(learned_proposal(outer_set(F1, k1 >= 2, active, M), rng)))
    hs = [fit_gbm(F1, D1[:, j])
          if 0 < D1[:, j].sum() < len(D1) else None for j in range(NF)]

    def H(Z):
        Fz = Z @ M
        return np.column_stack([h.predict(Fz) if h is not None else np.zeros(len(Z)) for h in hs]).astype(float)

    Xg = q.sample(200_000, rng); Gam = joint(np.exp(q.logw(Xg)), H(Xg))    # control means, no simulator
    X = rng.standard_normal((N2, D))
    out["lcv"] = cv_matrix(np.ones(N2), c.simulate(X).astype(float), H(X), Gam)
    X = q.sample(N2, rng); w = np.exp(q.logw(X)); Dn = c.simulate(X).astype(float)
    out["lis"] = joint(w, Dn)
    out["is_cv"] = cv_matrix(w, Dn, H(X), Gam)
    return out


def main():
    rng = np.random.default_rng(2026)
    c = Cascade(s=S_DESIGN); cs = Case()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    q = join(build_reference(lambda Z: c.simulate(Z).any(1), rng),
             build_reference(lambda Z: c.simulate(Z).sum(1) >= 2, rng))
    Qs = []
    for _ in range(8):
        X = q.sample(500_000, rng); Qs.append(joint(np.exp(q.logw(X)), c.simulate(X).astype(float)))
    Qref = np.mean(Qs, 0); Qse = np.std(Qs, 0, ddof=1) / np.sqrt(len(Qs))
    p = np.diag(Qref)
    Qind = np.outer(p, p); np.fill_diagonal(Qind, p)
    print("reference", time.time() - t0, "max rel se", (Qse / Qref).max(), flush=True)
    res = {"s": S_DESIGN, "lams": LAMS, "Qref": Qref.tolist(), "Qref_rel_se": (Qse / Qref).tolist(),
           "ties": c.A.tolist(), "names": cs.names, "designs": []}

    def record(name, seed, Q):
        for lam in LAMS:
            d = cs.solve(Q, lam)
            res["designs"].append({"est": name, "seed": seed, "lam": lam, "x": d["x"], "u": d["u"],
                                   "cost": d["cost"], "risk_est": cs.risk(d, Q), "risk_true": cs.risk(d, Qref),
                                   "tied_backup": float(np.mean([c.A[j, k] for _, j, k in d["assign"]])),
                                   "no_backup": float(np.mean([j == k for _, j, k in d["assign"]]))})
        json.dump(res, open(os.path.join(OUT, "frontier.json"), "w"))

    record("reference", -1, Qref)
    record("independent", -1, Qind)
    res["Q"] = {}
    for seed in range(SEEDS):
        t = time.time()
        est = estimates(c, np.random.default_rng(500 + seed))
        res["Q"][str(seed)] = {k: v.tolist() for k, v in est.items()}
        for name, Q in est.items():
            record(name, seed, Q)
        print("seed", seed, time.time() - t, flush=True)


if __name__ == "__main__":
    main()
