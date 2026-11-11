"""Map figures: examples of cascades on the Surabaya feeders, and the siting designs
obtained with the reference joint outage probabilities and under independence."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from cascade import figstyle  # noqa: E402
from cascade.figstyle import FSTYLE  # noqa: E402
from cascade.model import D, DATA, NF, Cascade  # noqa: E402
from cascade.siting import Case  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)
figstyle.apply(base=7.5)
plt.rcParams["axes.grid"] = False
META = json.load(open(os.path.join(DATA, "basemap", "surabaya_canvas.json")))
IMG = plt.imread(os.path.join(DATA, "basemap", "surabaya_canvas.jpg"))
EXTENT = (META["west"], META["east"], META["south"], META["north"])
BOX = (112.585, 112.848, -7.362, -7.186)
HALO = [pe.withStroke(linewidth=1.6, foreground="white")]
DARK, LIGHT = "#3a3a3a", "#f2f2f2"
RED, ORANGE, YELLOW, BLUE, GREEN, GREY = "#D55E00", "#E69F00", "#F0E442", "#0072B2", "#009E73", "#777777"
ROUND_COL = [RED, ORANGE, YELLOW, YELLOW, YELLOW]
S_DESIGN = 3.46
# label offsets (degrees lon, degrees lat, horizontal alignment) chosen to avoid overlaps
OFF = {"Embong Wungu": (0.004, 0.0075, "left"), "Indrapura": (0.006, 0.0005, "left"), "Ploso": (0.006, 0.0, "left"),
       "Tandes": (-0.006, 0.0, "right"), "Perak": (-0.003, 0.0085, "right"), "Kenjeran": (0.0, 0.0085, "center"),
       "Darmo Permai": (-0.006, 0.0, "right"), "Dukuh Kupang": (-0.004, -0.0085, "right"),
       "Ngagel": (0.006, -0.0045, "left"), "Rungkut": (0.0, -0.0085, "center"), "Karang Pilang": (0.0, -0.0085, "center")}


def label(ax, c, j, fs=5.4):
    dx, dy, ha = OFF[c.st.station[j]]
    va = "center" if dy == 0 or abs(dy) < 0.006 else ("bottom" if dy > 0 else "top")
    ax.text(c.st.lon[j] + dx, c.st.lat[j] + dy, c.st.station[j], ha=ha, va=va, fontsize=fs, zorder=8, path_effects=HALO)


def load_polygons(cs, dm):
    """Sub-district polygons (GADM 4.1, scripts/fetch_boundaries.py) and the feeder that serves
    each sub-district, taken as the candidate station with the shortest travel time."""
    import difflib
    path = os.path.join(DATA, "boundaries", "surabaya_kecamatan.geojson")
    if not os.path.exists(path):
        return None
    feats = json.load(open(path))["features"]
    key = lambda t: "".join(ch for ch in t.lower() if ch.isalpha())
    gnames = [key(f["properties"]["NAME_3"]) for f in feats]
    out = []
    for i, k in enumerate(dm.subdistrict):
        m = difflib.get_close_matches(key(k), gnames, n=1, cutoff=0.6)
        assert m, k
        g = feats[gnames.index(m[0])]["geometry"]
        rings = [g["coordinates"][0]] if g["type"] == "Polygon" else [part[0] for part in g["coordinates"]]
        out.append((int(np.argmin(cs.d[i])), rings))
    return out


def base(ax, c, ties=True):
    ax.imshow(IMG, extent=EXTENT, zorder=0, interpolation="bilinear", aspect="auto", alpha=0.6)
    ax.set_xlim(BOX[0], BOX[1]); ax.set_ylim(BOX[2], BOX[3])
    ax.set_aspect(1 / np.cos(np.radians(META["centre_lat"])))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color("#555555")
    if ties:
        for j in range(NF):
            for k in np.flatnonzero(c.A[j]):
                if k > j:
                    ax.plot([c.st.lon[j], c.st.lon[k]], [c.st.lat[j], c.st.lat[k]], color=GREY, lw=0.9, zorder=2)


def arrow(ax, c, j, k, color="k", ls="-"):
    x0, y0, x1, y1 = c.st.lon[j], c.st.lat[j], c.st.lon[k], c.st.lat[k]
    ax.annotate("", xy=(x0 + 0.80 * (x1 - x0), y0 + 0.80 * (y1 - y0)), xytext=(x0 + 0.20 * (x1 - x0), y0 + 0.20 * (y1 - y0)),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, ls=ls, mutation_scale=7, shrinkA=0, shrinkB=0,
                                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")]), zorder=4)


def draw_cascade(ax, c, tr, title, polys=None):
    base(ax, c, ties=False)
    rnd = tr["round"]
    if polys is not None:
        for j, rings in polys:
            for ring in rings:
                ax.add_patch(MplPolygon(ring, closed=True, facecolor=DARK if rnd[j] >= 0 else LIGHT,
                                        edgecolor="white" if rnd[j] >= 0 else "#b5b5b5", lw=0.35, alpha=0.88,
                                        zorder=1.1 if rnd[j] >= 0 else 1))
    for j in range(NF):
        for k in np.flatnonzero(c.A[j]):
            if k > j:
                xs, ys = [c.st.lon[j], c.st.lon[k]], [c.st.lat[j], c.st.lat[k]]
                ax.plot(xs, ys, color="white", lw=2.0, zorder=2); ax.plot(xs, ys, color=GREY, lw=0.9, zorder=2.1)
    # transfers that tripped a feeder: from a feeder down earlier to a feeder that tripped later
    for j in range(NF):
        for k in np.flatnonzero(c.A[j]):
            if rnd[j] >= 0 and rnd[k] > rnd[j]:
                arrow(ax, c, j, k)
    for j, k in tr["blocked"]:
        xm, ym = (c.st.lon[j] + c.st.lon[k]) / 2, (c.st.lat[j] + c.st.lat[k]) / 2
        ax.plot([c.st.lon[j], c.st.lon[k]], [c.st.lat[j], c.st.lat[k]], color="white", lw=1.6, zorder=2.5)
        ax.plot([c.st.lon[j], c.st.lon[k]], [c.st.lat[j], c.st.lat[k]], color=BLUE, lw=1.1, ls=(0, (2, 1.5)), zorder=2.6)
        ax.plot(xm, ym, marker="x", color=BLUE, ms=5, mew=1.4, zorder=5)
    for j in range(NF):
        x, y = c.st.lon[j], c.st.lat[j]
        size = 34 + 30 * c.z[j] / c.z.max()
        if rnd[j] < 0:
            ax.scatter(x, y, s=size, facecolor="white", edgecolor="k", lw=0.8, zorder=5)
        else:
            mk = "*" if tr["cause"][j] == "lightning" else "o"
            ax.scatter(x, y, s=size * (2.4 if mk == "*" else 1), marker=mk, facecolor=ROUND_COL[min(rnd[j], 4)], edgecolor="k", lw=0.8, zorder=6)
            ax.text(x, y, str(rnd[j] + 1), ha="center", va="center", fontsize=5.6, fontweight="bold",
                    color="white" if rnd[j] == 0 else "k", zorder=7)
        label(ax, c, j)
    ax.set_title(title, loc="left", fontsize=7.5, pad=3)
    ax.text(0.98, 0.955, f"city load factor $F$ = {tr['F']:.1f}", transform=ax.transAxes, fontsize=6, ha="right", va="top",
            path_effects=HALO, zorder=9)


def pick_examples(c):
    rng = np.random.default_rng(11)
    out = []
    # (a) one overload, local chain: nominal law
    X = rng.standard_normal((3_000_000, D)); dn = c.simulate(X)
    for i in np.flatnonzero(dn.sum(1) == 4):
        tr = c.trace(X[i])
        if (tr["round"] == 0).sum() == 1 and "lightning" not in tr["cause"] and tr["round"].max() >= 2:
            out.append(tr); break
    # (b) lightning stroke above the critical current on one feeder (conditional example)
    X = rng.standard_normal((400_000, D)); X[:, 12:] = np.minimum(X[:, 12:], 3.0)
    j = rng.integers(0, NF, len(X)); X[np.arange(len(X)), 12 + j] = c.xi_crit + 0.3
    dn = c.simulate(X)
    for i in np.flatnonzero(dn.sum(1) >= 3):
        tr = c.trace(X[i])
        if (tr["round"] == 0).sum() == 1 and len(tr["blocked"]) >= 1:
            out.append(tr); break
    # (c) hot day: large common factor (conditional example)
    X = rng.standard_normal((400_000, D)); X[:, 0] += 2.0
    dn = c.simulate(X); k = dn.sum(1)
    for i in np.flatnonzero((k >= 6) & (k <= 7)):
        tr = c.trace(X[i])
        if "lightning" not in tr["cause"] and (tr["round"] == 0).sum() <= 2 and tr["F"] < 4.0:
            out.append(tr); break
    return out


def fig_cascade():
    c = Cascade(s=S_DESIGN)
    ex = pick_examples(c)
    titles = []
    for lab, tr, what in zip("abc", ex, ("Overload", "Lightning", "Hot day")):
        first = [c.st.station[j] for j in np.flatnonzero(tr["round"] == 0)]
        n = int((tr["round"] >= 0).sum())
        head = f"{what} at {first[0]}" if len(first) == 1 else f"{what}, {len(first)} initial trips"
        titles.append(f"({lab}) {head}: {n} feeders down")
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.32), constrained_layout=True)
    polys = load_polygons(Case(), pd.read_csv(os.path.join(DATA, "demand.csv")))
    for a, tr, lab in zip(ax, ex, "abc"):
        draw_cascade(a, c, tr, f"({lab})", polys)
    h = [Line2D([], [], marker="o", ls="none", mfc=RED, mec="k", ms=6, label="Initial trip (overload)"),
         Line2D([], [], marker="*", ls="none", mfc=RED, mec="k", ms=9, label="Initial trip (lightning)"),
         Line2D([], [], marker="o", ls="none", mfc=ORANGE, mec="k", ms=6, label="Trips in round 2"),
         Line2D([], [], marker="o", ls="none", mfc=YELLOW, mec="k", ms=6, label="Round 3 or later"),
         Line2D([], [], marker="o", ls="none", mfc="white", mec="k", ms=6, label="Live feeder"),
         Line2D([], [], color="k", lw=1.3, marker=">", ms=4, label="Transfer that trips"),
         Line2D([], [], color=BLUE, lw=1.1, ls=(0, (2, 1.5)), marker="x", ms=5, label="Tie over its limit"),
         Line2D([], [], color=GREY, lw=0.9, label="Tie switch"),
         Patch(facecolor=DARK, edgecolor="white", label="Sub-district without supply"),
         Patch(facecolor=LIGHT, edgecolor="#999999", label="Sub-district with supply")]
    fig.legend(handles=h, loc="outside lower center", ncol=5, columnspacing=1.2, handletextpad=0.4, fontsize=6.3)
    fig.savefig(os.path.join(FIG, "fig_cascade.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG, "fig_cascade.png"), dpi=300, transparent=False, facecolor="white")
    plt.close(fig)
    json.dump([{"round": t["round"].tolist(), "cause": list(t["cause"]), "F": t["F"], "blocked": t["blocked"]} for t in ex],
              open(os.path.join(ROOT, "results", "cascade_examples.json"), "w"))
    for t in titles:
        print(t)


def draw_design(ax, c, cs, dm, d, Q, title):
    base(ax, c, ties=False)
    for j in range(NF):
        for k in np.flatnonzero(c.A[j]):
            if k > j:
                ax.plot([c.st.lon[j], c.st.lon[k]], [c.st.lat[j], c.st.lat[k]], color=GREY, lw=0.6, ls=":", zorder=2)
    for i, j, k in d["assign"]:
        xi, yi = dm.lon[i], dm.lat[i]
        ax.plot([xi, c.st.lon[j]], [yi, c.st.lat[j]], color=BLUE, lw=0.55, alpha=0.8, zorder=3)
        if k != j:
            tied = c.A[j, k]
            ax.plot([xi, c.st.lon[k]], [yi, c.st.lat[k]], color=RED if tied else GREEN, lw=1.0 if tied else 0.55,
                    ls=(0, (3, 1.6)), alpha=0.95 if tied else 0.8, zorder=3.5 if tied else 3)
    for i, j, k in d["assign"]:
        nob = j == k; tied = (not nob) and c.A[j, k]
        ax.scatter(dm.lon[i], dm.lat[i], s=5 + 0.45 * cs.w[i], marker="o",
                   facecolor="white" if nob else (RED if tied else "#444444"), edgecolor="k", lw=0.4, zorder=5)
    for j in range(NF):
        x, y = c.st.lon[j], c.st.lat[j]
        if d["x"][j]:
            ax.scatter(x, y, s=62, marker="s", facecolor=BLUE, edgecolor="k", lw=0.8, zorder=6)
            ax.text(x, y, str(d["u"][j]), ha="center", va="center", color="white", fontsize=5.6, fontweight="bold", zorder=7)
            label(ax, c, j, fs=5.6)
        else:
            ax.scatter(x, y, s=22, marker="s", facecolor="white", edgecolor="#555555", lw=0.6, zorder=4)
    ax.set_title(title, loc="left", fontsize=7.5, pad=3)
    ax.text(0.02, 0.03, f"{d['cost']/1e6:.1f} million IDR per day\n{365*cs.risk(d, Q):.1f} unserved vehicles per year",
            transform=ax.transAxes, fontsize=6.2, path_effects=HALO, zorder=9, linespacing=1.25)


def fig_designs(eta=1e8):
    c = Cascade(s=S_DESIGN); cs = Case()
    dm = pd.read_csv(os.path.join(DATA, "demand.csv"))
    fr = json.load(open(os.path.join(ROOT, "results", "frontier.json")))
    Q = np.array(fr["Qref"]); p = np.diag(Q)
    Qind = np.outer(p, p); np.fill_diagonal(Qind, p)
    dref, dind = cs.solve(Q, eta), cs.solve(Qind, eta)
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.36), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1, 0.92]})
    draw_design(ax[0], c, cs, dm, dref, Q, "(a)")
    draw_design(ax[1], c, cs, dm, dind, Q, "(b)")
    D_ = fr["designs"]; a = ax[2]
    a.grid(True, color="#cccccc", lw=0.4)
    a.set_axisbelow(True); a.set_xticks([12, 16, 20, 24])
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)
    ref = sorted([x for x in D_ if x["est"] == "reference"], key=lambda x: x["cost"])
    a.plot([x["cost"] / 1e6 for x in ref], [365 * x["risk_true"] for x in ref], color="k", lw=1.0, zorder=1)
    rng = np.random.default_rng(0)
    for est in ("independent", "mc", "lcv", "lis", "is_cv", "reference"):
        lab, col, mk = FSTYLE[est]
        sel = [x for x in D_ if x["est"] == est]
        jit = 0 if est in ("reference", "independent") else rng.normal(0, 0.07, len(sel))
        big = est in ("reference", "independent")
        a.scatter(np.array([x["cost"] / 1e6 for x in sel]) + jit, [365 * x["risk_true"] for x in sel], s=20 if big else 11,
                  facecolors=col if big else "none", edgecolors="k" if big else col, marker=mk, lw=0.5 if big else 0.7,
                  label=lab, zorder=4 if big else 2)
    a.set_yscale("log"); a.set_yticks([1, 2, 5, 10, 20, 50, 100]); a.set_yticklabels(["1", "2", "5", "10", "20", "50", "100"])
    a.minorticks_off(); a.set_ylim(1.1, 110)
    a.set_xlabel("Daily cost (million IDR)"); a.set_ylabel("Unserved vehicles per year")
    a.legend(loc="upper right", fontsize=5.8, handletextpad=0.2, borderpad=0.2, labelspacing=0.3)
    a.set_title("(c)", loc="left", fontsize=7.5, pad=3)
    h = [Line2D([], [], marker="s", ls="none", mfc=BLUE, mec="k", ms=7, label="Open station (connectors)"),
         Line2D([], [], marker="s", ls="none", mfc="white", mec="#555555", ms=4, label="Candidate not opened"),
         Line2D([], [], color=BLUE, lw=0.8, label="Primary station"),
         Line2D([], [], color=GREEN, lw=0.8, ls=(0, (3, 1.6)), label="Backup on an untied feeder"),
         Line2D([], [], color=RED, lw=1.1, ls=(0, (3, 1.6)), marker="o", mfc=RED, mec="k", ms=4, label="Backup on a tied feeder"),
         Line2D([], [], marker="o", ls="none", mfc="white", mec="k", ms=4, label="Sub-district without backup"),
         Line2D([], [], color=GREY, lw=0.7, ls=":", label="Tie switch")]
    fig.legend(handles=h, loc="outside lower left", ncol=4, columnspacing=1.0, handletextpad=0.4, fontsize=6.3)
    fig.savefig(os.path.join(FIG, "fig_designs.pdf"), dpi=300)
    fig.savefig(os.path.join(FIG, "fig_designs.png"), dpi=300, transparent=False, facecolor="white")
    plt.close(fig)
    out = {}
    for name, d in (("reference", dref), ("independent", dind)):
        out[name] = {"stations": [c.st.station[j] for j in range(NF) if d["x"][j]], "cost": d["cost"],
                     "risk_per_year": 365 * cs.risk(d, Q), "tied_backup": float(np.mean([c.A[j, k] for _, j, k in d["assign"]])),
                     "no_backup": float(np.mean([j == k for _, j, k in d["assign"]]))}
        print(name, out[name])
    json.dump(out, open(os.path.join(ROOT, "results", "design_maps.json"), "w"), indent=1)


if __name__ == "__main__":
    fig_cascade()
    fig_designs()
