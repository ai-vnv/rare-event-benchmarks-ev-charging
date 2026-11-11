"""One replication of the benchmark: shared pilot stage, then every estimator."""
from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtr

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from .features import feature_map
from .model import D
from .sampling import Mixture, build_reference, greedy_dominating, mixture_at

Z95 = 1.959964


def explore(c, rng, n1=10_000, rounds=8, kmax=150, event=None):
    """Pilot stage: tempered search, then rounds of sampling around low-norm failures."""
    nb = n1 // (rounds + 1)
    event = c.event if event is None else event
    taus = np.repeat([1.3, 1.6, 1.9, 2.2], nb // 4)
    X = taus[:, None] * rng.standard_normal((len(taus), D))
    Xs, Ys = [X], [event(X)]
    for _ in range(rounds):
        F = np.vstack(Xs)[np.concatenate(Ys)]
        if len(F) == 0:
            Xn = 2.5 * rng.standard_normal((nb, D))
        else:
            pts = greedy_dominating(F, delta=8.0, kmax=kmax)
            Xn = mixture_at(0.92 * pts, nominal=0.05, temper=0.3).sample(nb, rng)
        Xs.append(Xn); Ys.append(event(Xn))
    return np.vstack(Xs), np.concatenate(Ys)


def score(c, X):
    """Continuous score for cross-entropy: tripped feeders plus the next feeder's loading."""
    down = c.simulate(X)
    L = c.loads(X) / c.z
    nxt = np.where(down, -np.inf, L).max(1)
    return down.sum(1) + np.clip(np.where(np.isfinite(nxt), nxt, 1.0), 0, 0.999)


def cross_entropy(c, rng, n1=10_000, iters=5, rho=0.1):
    """Multilevel cross-entropy with one mean-shifted Gaussian."""
    nb = n1 // iters
    mu = np.zeros(D)
    for _ in range(iters):
        X = mu + rng.standard_normal((nb, D))
        S = score(c, X)
        lw = -X @ mu + 0.5 * mu @ mu
        lvl = min(np.quantile(S, 1 - rho), float(c.m))
        el = S >= lvl
        w = np.exp(lw[el] - lw[el].max())
        mu = (w[:, None] * X[el]).sum(0) / w.sum()
    return Mixture(mu[None], [1.0])


def mean_se(v):
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def cv_fit(y, cc, gamma):
    """Control-variate estimate with fitted coefficient; falls back to beta = 1."""
    vc = cc.var(ddof=1)
    beta = np.cov(y, cc)[0, 1] / vc if vc > 0 else 1.0
    r = y - beta * (cc - gamma)
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r))), float(beta)


def learned_proposal(g, rng, light=False):
    """Mixture at sampled dominating points of a learned set; surrogate calls only."""
    if light:
        return build_reference(g, rng, n_explore=20_000, rounds=1, n_round=20_000, cap=3000,
                               refine=2, kmax=60, grid_n=12)
    return build_reference(g, rng, n_explore=40_000, rounds=2, n_round=40_000, cap=4000,
                           refine=4, kmax=100, grid_n=15)


def fit_gbm(F, y):
    return HistGradientBoostingClassifier(max_iter=60, random_state=0).fit(F, y)


def pilot(c, rng, n1, M, event=None):
    """Pilot stage of n1 runs: exploration, then two active rounds that sample the
    simulator around the dominating points of the current learned set."""
    event = c.event if event is None else event
    na = n1 // 5
    X1, Y1 = explore(c, rng, n1 - 2 * na, rounds=5, event=event)
    active = np.zeros(len(X1), dtype=bool)
    for _ in range(2):
        if Y1.any() and not Y1.all():
            g0 = fit_gbm(X1 @ M, Y1)
            q0 = learned_proposal(lambda Z: g0.decision_function(Z @ M) >= -1.0, rng, light=True)
        else:
            q0 = None
        Xa = q0.sample(na, rng) if q0 is not None else 2.5 * rng.standard_normal((na, D))
        X1 = np.vstack([X1, Xa]); Y1 = np.concatenate([Y1, event(Xa)])
        active = np.concatenate([active, np.ones(na, dtype=bool)])
    return X1, Y1, active


def widen(q, shrink=0.9, tau=1.3):
    """Defensive version of a learned proposal: components pulled inward and widened."""
    return mixture_at(shrink * q.means[:-1], nominal=0.05, temper=0.5, tau=tau)


def outer_threshold(F1, Y1, active):
    """Threshold with 1% out-of-fold false negatives among the cascades of the active rounds."""
    oof = cross_val_predict(HistGradientBoostingClassifier(max_iter=60, random_state=0), F1, Y1,
                            cv=3, method="decision_function")
    sel = Y1 & active if (Y1 & active).sum() >= 20 else Y1
    return float(np.clip(np.quantile(oof[sel], 0.01), -6.0, 0.0))


def surrogate_mass(g, q, rng, n=200_000):
    X = q.sample(n, rng)
    return mean_se(np.exp(q.logw(X)) * g(X))


def replicate(c, rng, n1=10_000, n2=10_000, ref=None):
    """Run all estimators once.  `ref` = (X, logw, Y, theta) from the reference proposal
    gives the exact per-run variance of each estimator."""
    out, tm = {}, {}
    N = n1 + n2
    theta = ref[3] if ref is not None else np.nan
    M = feature_map(c.rho)

    # ---- crude Monte Carlo and the Gaussian-proxy control variate (budget N)
    X = rng.standard_normal((N, D)); Y = c.event(X).astype(float)
    out["mc"] = mean_se(Y)
    gp = float(ndtr(-c.s))
    out["proxy_cv"] = cv_fit(Y, (X[:, 0] > c.s).astype(float), gp)[:2]

    # ---- shared pilot stage and the learned sets
    t0 = time.time()
    X1, Y1, active = pilot(c, rng, n1, M)
    out["n_fail_pilot"] = int(Y1.sum()); out["n_fail_active"] = int((Y1 & active).sum())
    F1 = X1 @ M
    gbm = fit_gbm(F1, Y1)
    kap = outer_threshold(F1, Y1, active)
    lin = LogisticRegression(max_iter=500).fit(F1, Y1)
    wl = M @ lin.coef_[0]; bl = float(lin.intercept_[0])
    sets = {
        "strong": lambda Z: gbm.decision_function(Z @ M) >= 0.0,
        "outer": lambda Z: gbm.decision_function(Z @ M) >= kap,
        "weak": lambda Z: Z @ wl + bl >= 0.0,
    }
    tm["train"] = time.time() - t0; t0 = time.time()
    q_in = learned_proposal(sets["strong"], rng)
    q_o1 = learned_proposal(sets["outer"], rng)
    q_out = widen(q_o1)
    gam, gam_se = surrogate_mass(sets["strong"], q_in, rng)
    gam_w = float(ndtr(bl / np.linalg.norm(wl)))            # exact mean of the linear set
    ub, ub_se = surrogate_mass(sets["outer"], q_o1, rng)
    out["gamma"], out["gamma_weak"], out["ub"], out["kappa"] = (gam, gam_se), gam_w, (ub, ub_se), kap
    out["K"] = (len(q_in.w) - 1, len(q_out.w) - 1)
    tm["proposal"] = time.time() - t0; t0 = time.time()

    # ---- learned control variates under nominal sampling (n2 runs)
    X = rng.standard_normal((n2, D)); Y = c.event(X).astype(float)
    out["lcv_strong"] = cv_fit(Y, sets["strong"](X).astype(float), gam)[:2]
    out["lcv_weak"] = cv_fit(Y, sets["weak"](X).astype(float), gam_w)[:2]

    # ---- cross-entropy importance sampling (own n1 pilot, n2 runs)
    qce = cross_entropy(c, rng, n1)
    X = qce.sample(n2, rng)
    out["ce_is"] = mean_se(np.exp(qce.logw(X)) * c.event(X))

    # ---- learned IS on the learned set, on its outer approximation, and with the control
    X = q_in.sample(n2, rng)
    out["lis"] = mean_se(np.exp(q_in.logw(X)) * c.event(X))
    X = q_out.sample(n2, rng); L = np.exp(q_out.logw(X)); Y = c.event(X).astype(float)
    C = sets["strong"](X).astype(float)
    out["lis_outer"] = mean_se(L * Y)
    out["is_cv"] = cv_fit(L * Y, L * C, gam)[:2]
    tm["estimate"] = time.time() - t0
    out["time"] = tm

    # ---- exact per-run variances from the reference sample
    if ref is not None:
        Xr, lwr, Yr, _ = ref
        wr = np.exp(lwr) * Yr                       # nominal mass on the cascade set
        v = {"mc": theta * (1 - theta)}

        def rho2(C, g):
            rec = float((wr * C).sum() / wr.sum())
            return rec, rec * theta / g, (rec * theta - theta * g) ** 2 / (theta * (1 - theta) * g * (1 - g))

        Cs = sets["strong"](Xr)
        rec, prec, r2 = rho2(Cs, gam)
        out["strong"] = {"recall": rec, "precision": prec, "rho2": r2}
        v["lcv_strong"] = v["mc"] * (1 - r2)
        rec_w, prec_w, r2w = rho2(sets["weak"](Xr), gam_w)
        out["weak"] = {"recall": rec_w, "precision": prec_w, "rho2": r2w}
        v["lcv_weak"] = v["mc"] * (1 - r2w)
        v["proxy_cv"] = v["mc"] * (1 - rho2(Xr[:, 0] > c.s, gp)[2])
        out["outer_recall"] = rho2(sets["outer"](Xr), ub)[0]
        for name, prop in (("lis", q_in), ("lis_outer", q_out), ("ce_is", qce)):
            v[name] = float((wr * np.exp(prop.logw(Xr))).mean() - theta ** 2)
        # combined estimator: Var_q[L (Y - b C)] at the best fixed b
        Lr = np.exp(q_out.logw(Xr))
        Xe = q_out.sample(60_000, rng); Le = np.exp(q_out.logw(Xe)); Ye = c.event(Xe)
        Ce = sets["strong"](Xe)
        a_yc = float((wr * Lr * Cs).mean())                   # E_p[L Y C]
        a_cc = a_yc + float((Le * Le * Ce * ~Ye).mean())      # E_p[L C]
        b = (a_yc - theta * gam) / (a_cc - gam ** 2)
        v["is_cv"] = v["lis_outer"] - b * b * (a_cc - gam ** 2)
        out["exact_var"] = {k: float(val) for k, val in v.items()}
        out["n_req"] = {k: float((Z95 / 0.1) ** 2 * val / theta ** 2) for k, val in v.items()}
    return out
