"""Calibrate the feeder headroom s to each target cascade probability.

    python scripts/calibrate_levels.py

For each target in 1e-3, ..., 1e-8, a secant search on log10(theta) against s is run.
Each step estimates theta with an oracle mixture proposal (free use of the simulator).
The result is written to results/levels.json (or to RESULTS_DIR when that is set).
The shipped levels.json was produced this way and is rounded to three decimals.
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cascade.model import Cascade  # noqa: E402
from cascade.sampling import build_reference  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))
TARGETS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
START = [3.0, 3.5, 3.9, 4.3, 4.6, 5.0]


def theta_at(s, rng, n=1_000_000):
    c = Cascade(s=s)
    q = build_reference(c.event, rng, refine=8)
    X = q.sample(n, rng)
    return float(np.mean(np.exp(q.logw(X)) * c.event(X)))


def calibrate(target, s0, rng, tol=0.02, iters=8):
    s_prev, l_prev = s0, np.log10(theta_at(s0, rng))
    s = s0 + 0.1 * np.sign(l_prev - np.log10(target))
    for _ in range(iters):
        l_cur = np.log10(theta_at(s, rng))
        print(f"  s {s:.3f} log10 theta {l_cur:.3f}", flush=True)
        if abs(l_cur - np.log10(target)) < tol:
            break
        slope = (l_cur - l_prev) / (s - s_prev)
        s_prev, l_prev = s, l_cur
        s = s + (np.log10(target) - l_cur) / slope
    return round(float(s), 3)


def main():
    rng = np.random.default_rng(7)
    levels = []
    for t, s0 in zip(TARGETS, START):
        print("target", t, flush=True)
        levels.append({"target": t, "s": calibrate(t, s0, rng)})
    os.makedirs(OUT, exist_ok=True)
    json.dump(levels, open(os.path.join(OUT, "levels.json"), "w"), indent=1)
    print(levels)


if __name__ == "__main__":
    main()
