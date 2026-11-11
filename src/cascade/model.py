"""Cascading feeder-trip model for the 11 Surabaya feeders.

Inputs are a standard normal vector X of dimension 23:
  X[0]      city-wide load factor F (weather),
  X[1:12]   idiosyncratic load noise of each feeder,
  X[12:23]  lightning severity of each feeder (standardised log peak stroke current).
A feeder trips at the start of the day if its load exceeds its rating, or if the peak
stroke current on its corridor exceeds the critical current.  A tripped feeder hands a
fraction of its load to the live feeders it is tied to; a tie that would carry more than
its pick-up limit stays open.  A live feeder trips when its own load plus the load it
picked up exceeds its rating.  The process repeats until no new feeder trips.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
D = 23
NF = 11

# Site-specific log-normal stroke current, Firdaus, Arla and Arief (ICT-PEP 2026)
MU_LNI = 2.840
SIG_LNI = 0.601
I_CRIT_KA = 219.0          # largest stroke recorded on the corridor, 2020-2025


def load_stations():
    return pd.read_csv(os.path.join(DATA, "stations.csv"))


def tie_graph(st, k=2):
    """Symmetric k-nearest-neighbour graph on the feeder coordinates."""
    lat = np.radians(st.lat.values)
    lon = np.radians(st.lon.values)
    x = 6371.0 * lon * np.cos(lat.mean())
    y = 6371.0 * lat
    dist = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    A = np.zeros((NF, NF), dtype=bool)
    for j in range(NF):
        nb = np.argsort(dist[j])[1:k + 1]
        A[j, nb] = True
    A = A | A.T
    return A, dist


@dataclass
class Cascade:
    s: float = 2.0            # feeder headroom in standard deviations: rating = mean + s sd
    rho: float = 0.3          # share of load variance from the city-wide factor
    lam: float = 0.15         # share of a tripped feeder's load that is back-fed
    kappa: float = 0.20       # tie pick-up limit as a share of the smaller rating
    m: int = 3                # cascade event: at least m feeders down
    lightning: bool = True
    transfer: bool = True
    st: pd.DataFrame = field(default_factory=load_stations)

    def __post_init__(self):
        self.mu = self.st.demand_mean.values.astype(float)
        self.sd = self.st.demand_sd.values.astype(float)
        self.A, self.dist = tie_graph(self.st)
        self.z = self.mu + self.s * self.sd
        self.limit = self.kappa * np.minimum(self.z[:, None], self.z[None, :])
        self.nb = [np.flatnonzero(self.A[j]) for j in range(NF)]
        self.xi_crit = (np.log(I_CRIT_KA) - MU_LNI) / SIG_LNI

    def loads(self, X):
        F = X[:, :1]
        eps = X[:, 1:12]
        return self.mu + self.sd * (np.sqrt(self.rho) * F + np.sqrt(1 - self.rho) * eps)

    def simulate(self, X):
        """Return the boolean matrix of tripped feeders, shape (n, 11)."""
        X = np.atleast_2d(X)
        L = self.loads(X)
        down = L > self.z
        if self.lightning:
            down = down | (X[:, 12:23] > self.xi_crit)
        if not self.transfer:
            return down
        act = np.flatnonzero(down.any(1))          # only days with a trip can cascade
        Af = self.A.astype(float)
        while act.size:
            La, da = L[act], down[act]
            live = ~da
            nlive = live.astype(float) @ Af            # live neighbours of each feeder
            share = np.where(da & (nlive > 0), self.lam * La / np.maximum(nlive, 1), 0.0)
            T = np.zeros_like(La)
            for j in range(NF):
                nb = self.nb[j]
                sj = share[:, j:j + 1]
                T[:, nb] += np.where((sj <= self.limit[j, nb]) & live[:, nb], sj, 0.0)
            new = live & (La + T > self.z)
            grew = new.any(1)
            if not grew.any():
                break
            down[act[grew]] = da[grew] | new[grew]
            act = act[grew]
        return down

    def event(self, X):
        return self.simulate(X).sum(1) >= self.m

    def trace(self, x):
        """Follow one day: round in which each feeder trips (-1 if it stays live), the cause of
        each trip, the transfers that were carried and the ties that stayed open."""
        x = np.asarray(x, dtype=float)
        L = self.loads(x[None])[0]
        over = L > self.z
        light = (x[12:23] > self.xi_crit) if self.lightning else np.zeros(NF, dtype=bool)
        rnd = np.where(over | light, 0, -1)
        cause = np.where(light, "lightning", np.where(over, "overload", "")).astype(object)
        carried, blocked = [], []
        for r in range(1, NF + 1):
            down = rnd >= 0
            T = np.zeros(NF); step_c, step_b = [], []
            for j in np.flatnonzero(down):
                live_nb = [k for k in self.nb[j] if not down[k]]
                if not live_nb:
                    continue
                share = self.lam * L[j] / len(live_nb)
                for k in live_nb:
                    if share <= self.limit[j, k]:
                        T[k] += share; step_c.append((int(j), int(k)))
                    else:
                        step_b.append((int(j), int(k)))
            new = (~down) & (L + T > self.z)
            carried, blocked = step_c, step_b
            if not new.any():
                break
            rnd[new] = r; cause[new] = "transfer"
        return {"round": rnd, "cause": cause, "carried": carried, "blocked": blocked,
                "loading": L / self.z, "F": float(x[0])}
