"""Gaussian mixture proposals, importance weights, and the reference estimator."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .model import D

LOG2PI = np.log(2 * np.pi)


class Mixture:
    """Mixture of isotropic Gaussians N(m_k, tau_k^2 I) in D dimensions."""

    def __init__(self, means, weights, taus=None):
        self.means = np.atleast_2d(np.asarray(means, dtype=float))
        self.w = np.asarray(weights, dtype=float)
        self.w = self.w / self.w.sum()
        self.taus = np.ones(len(self.w)) if taus is None else np.asarray(taus, dtype=float)

    def sample(self, n, rng):
        k = rng.choice(len(self.w), size=n, p=self.w)
        return self.means[k] + self.taus[k, None] * rng.standard_normal((n, D))

    def logpdf(self, X, chunk=100_000):
        res = np.empty(X.shape[0])
        m2 = (self.means * self.means).sum(1)
        const = np.log(self.w) - D * np.log(self.taus) - 0.5 * D * LOG2PI
        for i in range(0, X.shape[0], chunk):
            Xc = X[i:i + chunk]
            d2 = (Xc * Xc).sum(1)[:, None] - 2 * Xc @ self.means.T + m2[None, :]
            res[i:i + chunk] = logsumexp(const[None, :] - 0.5 * d2 / self.taus[None, :] ** 2, axis=1)
        return res

    def logw(self, X):
        """log of nominal density over proposal density."""
        return -0.5 * (X * X).sum(1) - 0.5 * D * LOG2PI - self.logpdf(X)


def boundary_points(event, X, grid=np.linspace(0.3, 1.0, 29)):
    """Shrink each failure point toward the origin as far as the event still holds."""
    B = X.copy()
    alive = np.ones(len(X), dtype=bool)
    for t in grid[::-1][1:]:
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break
        ok = event(t * X[idx])
        B[idx[ok]] = t * X[idx[ok]]
        alive[idx[~ok]] = False
    return B


def greedy_dominating(B, delta=12.0, kmax=120):
    """Pick low-norm points whose half-spaces {a.x >= |a|^2} cover the candidates."""
    r2 = (B * B).sum(1)
    order = np.argsort(r2)
    B, r2 = B[order], r2[order]
    keep = np.ones(len(B), dtype=bool)
    pts = []
    for i in range(len(B)):
        if not keep[i]:
            continue
        if r2[i] > r2[0] + 2 * delta or len(pts) >= kmax:
            break
        a = B[i]
        pts.append(a)
        keep &= ~(B @ a >= r2[i] - 1e-9)
    return np.array(pts)


def mixture_at(points, nominal=0.05, temper=0.5, tau=1.0):
    r2 = (points * points).sum(1)
    w = np.exp(-0.5 * temper * (r2 - r2.min()))
    w = (1 - nominal) * w / w.sum()
    means = np.vstack([points, np.zeros((1, D))])
    return Mixture(means, np.append(w, nominal), np.append(np.full(len(points), tau), 1.0))


def is_estimate(logw, Y):
    v = np.exp(logw) * Y
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v))


def build_reference(event, rng, n_explore=200_000, rounds=3, n_round=200_000, cap=20000,
                    refine=12, kmax=120, grid_n=29, verbose=False):
    """Oracle mixture proposal built with free use of the simulator."""
    cand = []
    for tau in (1.3, 1.6, 1.9, 2.2):
        X = tau * rng.standard_normal((n_explore // 4, D))
        F = X[event(X)]
        if len(F) > cap // 5:
            F = F[rng.choice(len(F), cap // 5, replace=False)]
        if len(F):
            cand.append(boundary_points(event, F, grid=np.linspace(0.3, 1.0, grid_n)))
    if not cand:
        return None
    cand = np.vstack(cand)
    pts = greedy_dominating(cand, kmax=kmax)
    for r in range(rounds):
        q = mixture_at(pts, tau=1.15)
        X = q.sample(n_round, rng)
        F = X[event(X)]
        if len(F) > cap:
            F = F[rng.choice(len(F), cap, replace=False)]
        cand = np.vstack([cand, boundary_points(event, F, grid=np.linspace(0.6, 1.0, max(grid_n * 2 // 3, 8)))])
        pts = greedy_dominating(cand, kmax=kmax)
        if verbose:
            print("  round", r, "components", len(pts), "min r2", (pts * pts).sum(1).min())
    # add components wherever single samples still carry a large share of the estimate
    for r in range(refine):
        q = mixture_at(pts)
        X = q.sample(n_round, rng)
        Y = event(X)
        w = np.exp(q.logw(X)) * Y
        share = w / w.sum()
        top = np.argsort(share)[::-1][:15]
        top = top[share[top] > 0.002]
        if verbose:
            print("  refine", r, "components", len(pts), "max share", share.max())
        if top.size == 0:
            break
        pts = np.vstack([pts, boundary_points(event, X[top], grid=np.linspace(0.6, 1.0, 21))])
    return mixture_at(pts)
