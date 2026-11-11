"""Rarity sweep: exact per-run variance and realised estimates of every estimator.

    python scripts/run_benchmark.py LEVEL [REPS]

LEVEL indexes results/levels.json (headroom s calibrated to a cascade probability).
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # one thread per process
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cascade.model import Cascade  # noqa: E402
from cascade.pipeline import replicate  # noqa: E402
from cascade.sampling import build_reference  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT = os.environ.get("RESULTS_DIR", os.path.join(ROOT, "results"))   # set RESULTS_DIR for a scratch run


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    return o


def main():
    lev = int(sys.argv[1]); reps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    levels = json.load(open(os.path.join(ROOT, "results", "levels.json")))
    s = levels[lev]["s"]
    rng = np.random.default_rng(1000 + lev)
    c = Cascade(s=s)
    t0 = time.time()
    q = build_reference(c.event, rng, refine=20)
    # reference probability from a large sample, then a smaller sample kept for evaluation
    tot, tot2, n, light = 0.0, 0.0, 0, 0.0
    size = np.zeros(12)
    for _ in range(16):
        X = q.sample(500_000, rng); w = np.exp(q.logw(X)); down = c.simulate(X)
        k = down.sum(1); v = w * (k >= c.m)
        tot += v.sum(); tot2 += (v * v).sum(); n += len(v)
        light += (v * (X[:, 12:] > c.xi_crit).any(1)).sum()
        size += np.bincount(k, weights=w, minlength=12)
    theta = tot / n; se = np.sqrt((tot2 / n - theta ** 2) / n)
    Xr = q.sample(400_000, rng); lwr = q.logw(Xr); Yr = c.event(Xr)
    head = {"s": s, "theta": theta, "theta_re": se / theta, "lightning_share": light / tot,
            "size_given_cascade": (size[c.m:] / size[c.m:].sum()).tolist(),
            "K_ref": len(q.w) - 1, "ref_time": time.time() - t0}
    print(head, flush=True)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"benchmark_level{lev}.json")
    runs = []
    for r in range(reps):
        t = time.time()
        o = replicate(c, rng, ref=(Xr, lwr, Yr, theta))
        o["rep_time"] = time.time() - t
        runs.append(clean(o))
        json.dump({"head": clean(head), "runs": runs}, open(path, "w"), indent=1)
        print(lev, r, f"{o['rep_time']:.0f}s", {k: f"{v:.2e}" for k, v in o["n_req"].items()}, flush=True)


if __name__ == "__main__":
    main()
