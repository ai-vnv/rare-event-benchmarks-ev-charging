"""Rarity-sweep figure, Table I and the summary files behind the numbers in the paper.

    python scripts/make_figures.py

Reads results/benchmark_level*.json and results/frontier.json, and writes
figs/fig_sweep.pdf (with a PNG copy), tables/tab_sweep.tex, results/summary.json and
results/frontier_summary.json.  It also prints every quantity quoted in Section V.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cascade import figstyle  # noqa: E402
from cascade.figstyle import STYLE  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figs"); TAB = os.path.join(ROOT, "tables")
figstyle.apply(base=8)

Z = 1.959964


def load():
    levels = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "benchmark_level*.json"))):
        levels.append(json.load(open(f)))
    return sorted(levels, key=lambda d: -d["head"]["theta"])


def med(v):
    return float(np.median(v))


def summarise(levels):
    rows = []
    for d in levels:
        th = d["head"]["theta"]; R = d["runs"]
        row = {"theta": th, "s": d["head"]["s"], "reps": len(R), "lightning": d["head"]["lightning_share"],
               "theta_re": d["head"]["theta_re"],
               "recall": med([r["strong"]["recall"] for r in R]),
               "precision": med([r["strong"]["precision"] for r in R]),
               "rho2": med([r["strong"]["rho2"] for r in R]),
               "rp": med([r["strong"]["recall"] * r["strong"]["precision"] for r in R]),
               "rho2_gap": float(np.max([abs(r["strong"]["rho2"] - r["strong"]["recall"] * r["strong"]["precision"]) for r in R])),
               "rho2_weak": med([r["weak"]["rho2"] for r in R]),
               "outer_recall": med([r["outer_recall"] for r in R]),
               "ub_ratio": med([r["ub"][0] / th for r in R]),
               "rep_time": med([r["rep_time"] for r in R])}
        for k in STYLE:
            nr = np.array([r["n_req"][k] for r in R])
            est = np.array([r[k][0] for r in R]); se = np.array([r[k][1] for r in R])
            row[k] = {"n_req": med(nr), "n_lo": float(np.quantile(nr, 0.25)), "n_hi": float(np.quantile(nr, 0.75)),
                      "ratio": med(est / th), "r_lo": float(np.quantile(est / th, 0.25)),
                      "r_hi": float(np.quantile(est / th, 0.75)),
                      "cover": float(np.mean(np.abs(est - th) <= Z * se)),
                      "under": float(np.mean(est < 0.5 * th)),
                      "mare": med(np.abs(est / th - 1))}
        rows.append(row)
    return rows


def fig_sweep(rows):
    th = np.array([r["theta"] for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.2), constrained_layout=True)
    for k in ("mc", "lcv_strong", "ce_is", "lis", "lis_outer", "is_cv"):
        lab, col, mk, ls = STYLE[k]
        y = np.array([r[k]["n_req"] for r in rows])
        ax[0].plot(th, y, color=col, marker=mk, ls=ls, label=lab)
        if k != "ce_is":
            ax[0].fill_between(th, [r[k]["n_lo"] for r in rows], [r[k]["n_hi"] for r in rows], color=col, alpha=0.12, lw=0)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].invert_xaxis()
    ax[0].set_ylim(1e3, 1e12)
    ax[0].set_xlabel(r"Cascade probability $\theta$"); ax[0].set_ylabel("Runs for 10% relative error")
    ax[0].set_title("(a)", loc="left")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside upper center", ncol=6, columnspacing=1.2, handlelength=2.6)
    keys = ("lcv_strong", "lis", "lis_outer", "is_cv")
    x = np.arange(len(rows))
    for i, k in enumerate(keys):
        lab, col, mk, ls = STYLE[k]
        m = np.array([r[k]["ratio"] for r in rows])
        lo = np.array([r[k]["r_lo"] for r in rows]); hi = np.array([r[k]["r_hi"] for r in rows])
        ax[1].errorbar(x + (i - 1.5) * 0.17, m, yerr=[m - lo, hi - m], color=col, marker=mk, ls="none", capsize=1.5, lw=0.9, label=lab)
    ax[1].axhline(1.0, color="k", lw=0.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"$\\mathrm{{10^{{{int(round(np.log10(t)))}}}}}$" for t in th])
    ax[1].set_ylim(0, 1.5)
    ax[1].set_xlabel(r"Cascade probability $\theta$"); ax[1].set_ylabel(r"Estimate / truth at 20,000 runs")
    ax[1].set_title("(b)", loc="left")
    for k in ("mc",) + keys:
        lab, col, mk, ls = STYLE[k]
        ax[2].plot(x, [r[k]["cover"] for r in rows], color=col, marker=mk, ls=ls, label=lab)
    ax[2].axhline(0.95, color="k", lw=0.6)
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"$\\mathrm{{10^{{{int(round(np.log10(t)))}}}}}$" for t in th])
    ax[2].set_ylim(-0.03, 1.03)
    ax[2].set_xlabel(r"Cascade probability $\theta$"); ax[2].set_ylabel("Coverage of 95% interval")
    ax[2].set_title("(c)", loc="left")
    fig.savefig(os.path.join(FIG, "fig_sweep.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG, "fig_sweep.png"), dpi=300, transparent=False, facecolor="white")
    plt.close(fig)


def frontier_stats():
    f = os.path.join(ROOT, "results", "frontier.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    D = d["designs"]
    ref = {x["lam"]: x for x in D if x["est"] == "reference"}
    out = {}
    for est in ("reference", "independent", "mc", "lcv", "lis", "is_cv"):
        sel = [x for x in D if x["est"] == est]
        if not sel:
            continue
        reg = [(x["cost"] + x["lam"] * x["risk_true"]) / (ref[x["lam"]]["cost"] + x["lam"] * ref[x["lam"]]["risk_true"]) - 1 for x in sel]
        opt = [x["risk_est"] / x["risk_true"] for x in sel if x["risk_true"] > 0]
        out[est] = {"regret_mean": float(np.mean(reg)), "regret_max": float(np.max(reg)),
                    "regret_median": float(np.median(reg)), "regret_over10": float(np.mean(np.array(reg) > 0.10)),
                    "optimism": float(np.median(opt)), "optimism_lo": float(np.quantile(opt, 0.1)),
                    "risk_ratio": float(np.median([x["risk_true"] / ref[x["lam"]]["risk_true"] for x in sel])),
                    "risk_ratio_max": float(np.max([x["risk_true"] / ref[x["lam"]]["risk_true"] for x in sel])),
                    "stations": float(np.mean([sum(x["x"]) for x in sel])),
                    "tied": float(np.mean([x["tied_backup"] for x in sel])),
                    "same": float(np.mean([x["x"] == ref[x["lam"]]["x"] for x in sel]))}
    return d, out


def sci(x, d=1):
    e = int(np.floor(np.log10(abs(x)))); m = x / 10 ** e
    if round(m, d) >= 10:
        m /= 10; e += 1
    return f"{m:.{d}f}\\times10^{{{e}}}"


def e(x):
    """Compact scientific notation for table cells, e.g. 3.8e8."""
    ex = int(np.floor(np.log10(x))); m = x / 10 ** ex
    if round(m, 1) >= 10:
        m /= 10; ex += 1
    return f"{m:.1f}e{ex}"


def table_sweep(rows):
    EOL = r"\\"
    L = [r"\footnotesize\setlength{\tabcolsep}{2.2pt}", r"\begin{tabular}{@{}lccccccccc@{}}", r"\toprule",
         r"$\theta$ & $r$ & $\pi$ & $\rho^2_{YC}$ & MC & CV & CE & IS & D-IS & IS+CV" + EOL, r"\midrule"]
    for r in rows:
        L.append(f"${sci(r['theta'])}$ & {r['recall']:.2f} & {r['precision']:.2f} & {r['rho2']:.2f} & "
                 + " & ".join(e(r[k]["n_req"]) for k in ("mc", "lcv_strong", "ce_is", "lis", "lis_outer", "is_cv")) + EOL)
    L += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(TAB, "tab_sweep.tex"), "w").write("\n".join(L) + "\n")


def main():
    os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)
    rows = summarise(load())
    json.dump(rows, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1)
    fig_sweep(rows)
    for r in rows:
        print(f"theta {r['theta']:.2e} s {r['s']} reps {r['reps']} light {r['lightning']:.3f} rec {r['recall']:.2f} prec {r['precision']:.2f} rho2 {r['rho2']:.2f} rp {r['rp']:.2f} gap {r['rho2_gap']:.3f} rho2w {r['rho2_weak']:.3f} outer_rec {r['outer_recall']:.2f} ub/th {r['ub_ratio']:.1f}")
        for k in STYLE:
            v = r[k]
            print(f"    {k:11s} n_req {v['n_req']:.2e} [{v['n_lo']:.1e},{v['n_hi']:.1e}] ratio {v['ratio']:.2f} [{v['r_lo']:.2f},{v['r_hi']:.2f}] cover {v['cover']:.2f} under {v['under']:.2f} mare {v['mare']:.2f}")
    fs = frontier_stats()
    if fs:
        out = fs[1]
        for k, v in out.items():
            print(k, {a: round(b, 4) for a, b in v.items()})
        json.dump(out, open(os.path.join(ROOT, "results", "frontier_summary.json"), "w"), indent=1)
    table_sweep(rows)


if __name__ == "__main__":
    main()
