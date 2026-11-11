"""Charging-station siting with a backup station per sub-district.

Each sub-district is assigned a primary and a backup station (possibly the same one).
Its demand is unserved on a day when the feeders of both stations are down, so the risk
of a design is linear in the pairwise joint outage probabilities Q[j, k] (Q[j, j] is the
marginal).  Data and costs are those of the Surabaya model of Akhra, Vanany and Arief.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .model import DATA, NF


class Case:
    def __init__(self):
        st = pd.read_csv(os.path.join(DATA, "stations.csv"))
        dm = pd.read_csv(os.path.join(DATA, "demand.csv"))
        tt = pd.read_csv(os.path.join(DATA, "travel_time.csv"))
        par = json.load(open(os.path.join(DATA, "params.json")))
        self.names = st.station.tolist()
        self.h = st.daily_cost_idr.values
        self.g = par["connector_daily_cost_idr"]
        self.cap = par["connector_minutes_per_day"]
        self.qmax = par["max_connectors_per_station"]
        self.dmax = par["max_travel_time_min"]
        self.existing = [self.names.index(n) for n in par["existing_stations"]]
        self.w = (dm.demand_motorcycle + dm.demand_car).values.astype(float)
        self.t = (par["charge_time_min"]["motorcycle"] * dm.demand_motorcycle
                  + par["charge_time_min"]["car"] * dm.demand_car).values.astype(float)
        d = tt.pivot(index="subdistrict", columns="station", values="minutes_car")
        self.d = d.loc[dm.subdistrict.values, self.names].values
        self.ni = len(dm)
        reach = self.d <= self.dmax
        for i in np.flatnonzero(~reach.any(1)):      # keep the nearest station if none is in range
            reach[i, np.argmin(self.d[i])] = True
        self.triples = [(i, j, k) for i in range(self.ni) for j in range(NF) for k in range(NF)
                        if reach[i, j] and reach[i, k]]

    def solve(self, Q, lam, time_limit=60.0):
        """Minimise daily cost + lam * expected unserved vehicles per day."""
        nz = len(self.triples)
        nv = 2 * NF + nz                      # x, u, z
        c = np.zeros(nv)
        c[:NF] = self.h; c[NF:2 * NF] = self.g
        for n, (i, j, k) in enumerate(self.triples):
            c[2 * NF + n] = lam * self.w[i] * Q[j, k]
        lo, hi = [], []
        A = lil_matrix((self.ni + 2 * self.ni * NF + 2 * NF, nv))
        r = 0
        for i in range(self.ni):                                   # one pair per sub-district
            for n, (ii, j, k) in enumerate(self.triples):
                if ii == i:
                    A[r, 2 * NF + n] = 1
            lo.append(1); hi.append(1); r += 1
        prim = {}; back = {}
        for n, (i, j, k) in enumerate(self.triples):
            prim.setdefault((i, j), []).append(n); back.setdefault((i, k), []).append(n)
        for (i, j), ns in list(prim.items()) + list(back.items()):  # stations must be open
            for n in ns:
                A[r, 2 * NF + n] = 1
            A[r, j] = -1; lo.append(-np.inf); hi.append(0); r += 1
        for j in range(NF):                                        # capacity incl. backup duty
            for n, (i, jj, k) in enumerate(self.triples):
                if jj == j or k == j:
                    A[r, 2 * NF + n] = self.t[i]
            A[r, NF + j] = -self.cap; lo.append(-np.inf); hi.append(0); r += 1
            A[r, NF + j] = 1; A[r, j] = -self.qmax; lo.append(-np.inf); hi.append(0); r += 1
        A = A[:r].tocsr()
        lb = np.zeros(nv); ub = np.ones(nv); ub[NF:2 * NF] = self.qmax
        lb[self.existing] = 1
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(nv),
                   bounds=Bounds(lb, ub), options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
        if res.x is None:
            return None
        x = np.round(res.x[:NF]).astype(int); u = np.round(res.x[NF:2 * NF]).astype(int)
        z = np.round(res.x[2 * NF:]).astype(bool)
        assign = [(i, j, k) for (i, j, k), on in zip(self.triples, z) if on]
        return {"x": x.tolist(), "u": u.tolist(), "assign": assign,
                "cost": float(self.h @ x + self.g * u.sum())}

    def risk(self, design, Q):
        """Expected unserved vehicles per day of a design under joint outage matrix Q."""
        return float(sum(self.w[i] * Q[j, k] for i, j, k in design["assign"]))
