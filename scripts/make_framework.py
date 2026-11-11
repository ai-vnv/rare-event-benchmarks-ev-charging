"""Framework illustration of the two learned estimators, built from real components:
map crops of simulated cascade days, thumbnails of sampled days, and small vector icons."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Wedge

import make_maps as mm
from cascade.model import D, DATA, NF, Cascade
from cascade.siting import Case

ROOT = os.path.join(os.path.dirname(__file__), "..")
W, H = 7.16, 3.0
BLUE, GREEN, RED, ORANGE, YELLOW, GREY, DARK, LIGHT = mm.BLUE, mm.GREEN, mm.RED, mm.ORANGE, mm.YELLOW, "#666666", mm.DARK, "#eeeeee"
plt.rcParams.update({"font.size": 7})
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
c = Cascade(s=3.46); cs = Case(); dm = pd.read_csv(os.path.join(DATA, "demand.csv"))
polys = mm.load_polygons(cs, dm)
rng = np.random.default_rng(5)


# ------------------------------------------------------------------ icons (unit box, drawn at x, y with size s)
def bolt(x, y, s, col=YELLOW, z=20, a=ax):
    p = np.array([(0.42, 1), (0.82, 1), (0.6, 0.58), (0.86, 0.58), (0.3, 0), (0.46, 0.44), (0.2, 0.44)])
    a.add_patch(Polygon(p * s + (x, y), closed=True, fc=col, ec="k", lw=0.6, zorder=z, joinstyle="round"))


def gauge(x, y, s, z=20, a=ax):
    """Overload: a dial with the needle in the red zone."""
    cx, cy, r = x + s / 2, y + 0.18 * s, 0.5 * s
    a.add_patch(Wedge((cx, cy), r, 0, 180, fc="white", ec="k", lw=0.6, zorder=z))
    a.add_patch(Wedge((cx, cy), r, 0, 45, fc=RED, ec="none", zorder=z + 1))
    a.add_patch(Wedge((cx, cy), r, 0, 180, fc="none", ec="k", lw=0.6, zorder=z + 2))
    a.plot([cx, cx + 0.8 * r * np.cos(np.radians(22))], [cy, cy + 0.8 * r * np.sin(np.radians(22))], color="k", lw=1.0, zorder=z + 3, solid_capstyle="round")
    a.add_patch(Circle((cx, cy), 0.07 * s, fc="k", zorder=z + 4))


def bell(x, y, s, col=ORANGE, z=20):
    t = np.linspace(0, np.pi, 30)
    body = np.column_stack([0.5 + 0.3 * np.cos(t), 0.38 + 0.52 * np.sin(t)])
    body = np.vstack([(0.92, 0.2), (0.8, 0.38), body, (0.2, 0.38), (0.08, 0.2)])
    ax.add_patch(Polygon(body * s + (x, y), closed=True, fc=col, ec="k", lw=0.5, zorder=z, joinstyle="round"))
    ax.add_patch(Circle((x + 0.5 * s, y + 0.1 * s), 0.09 * s, fc=col, ec="k", lw=0.5, zorder=z))


def tree(x, y, s, shade=(1, 0, 0, 1), z=20):
    nodes = {"r": (0.5, 0.92), "a": (0.25, 0.52), "b": (0.75, 0.52), "l1": (0.1, 0.1), "l2": (0.4, 0.1), "l3": (0.6, 0.1), "l4": (0.9, 0.1)}
    for u, v in (("r", "a"), ("r", "b"), ("a", "l1"), ("a", "l2"), ("b", "l3"), ("b", "l4")):
        ax.plot([x + nodes[u][0] * s, x + nodes[v][0] * s], [y + nodes[u][1] * s, y + nodes[v][1] * s], color="k", lw=0.6, zorder=z)
    for k in ("r", "a", "b"):
        ax.add_patch(Circle((x + nodes[k][0] * s, y + nodes[k][1] * s), 0.075 * s, fc="white", ec="k", lw=0.6, zorder=z + 1))
    for k, sh in zip(("l1", "l2", "l3", "l4"), shade):
        px, py = x + nodes[k][0] * s, y + nodes[k][1] * s
        ax.add_patch(FancyBboxPatch((px - 0.07 * s, py - 0.07 * s), 0.14 * s, 0.14 * s, boxstyle="square,pad=0", fc=DARK if sh else "white", ec="k", lw=0.6, zorder=z + 1))


def arrow(x0, y0, x1, y1, col=GREY, lw=1.3, rad=0.0, z=15):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, color=col, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", zorder=z, shrinkA=0, shrinkB=0))


# ------------------------------------------------------------------ real components
LON0, LON1, LAT0, LAT1 = 112.59, 112.845, -7.36, -7.19
ASP = (LAT1 - LAT0) / ((LON1 - LON0) * np.cos(np.radians(7.27)))


def tile(x, y, w, down, frame="#888888", z=10):
    """Thumbnail of one simulated day: sub-districts dark where their feeder is down."""
    h = w * ASP
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03", fc="white", ec=frame, lw=0.6, zorder=z))
    for j, rings in polys:
        for ring in rings:
            r = np.asarray(ring)
            q = np.column_stack([x + (r[:, 0] - LON0) / (LON1 - LON0) * w, y + (r[:, 1] - LAT0) / (LAT1 - LAT0) * h])
            ax.add_patch(Polygon(q, closed=True, fc=DARK if down[j] else LIGHT, ec="white" if down[j] else "#c4c4c4", lw=0.15, zorder=z + 1))
    return h


def big_map(rect, tr):
    a = fig.add_axes(rect)
    mm.base(a, c, ties=False)
    rnd = np.asarray(tr["round"])
    for j, rings in polys:
        for ring in rings:
            a.add_patch(Polygon(ring, closed=True, fc=DARK if rnd[j] >= 0 else mm.LIGHT, ec="white" if rnd[j] >= 0 else "#b5b5b5", lw=0.3, alpha=0.88, zorder=1))
    for j in range(NF):
        for k in np.flatnonzero(c.A[j]):
            if k > j:
                xs, ys = [c.st.lon[j], c.st.lon[k]], [c.st.lat[j], c.st.lat[k]]
                a.plot(xs, ys, color="white", lw=1.6, zorder=2); a.plot(xs, ys, color="#777777", lw=0.7, zorder=2.1)
    for j in range(NF):
        for k in np.flatnonzero(c.A[j]):
            if rnd[j] >= 0 and rnd[k] > rnd[j]:
                mm.arrow(a, c, j, k)
    for j in range(NF):
        col = "white" if rnd[j] < 0 else mm.ROUND_COL[min(rnd[j], 4)]
        a.scatter(c.st.lon[j], c.st.lat[j], s=16, facecolor=col, edgecolor="k", lw=0.6, zorder=6)
    s = 0.026
    for j in np.flatnonzero(rnd == 0):
        if tr["cause"][j] == "lightning":
            bolt(c.st.lon[j] - 0.4 * s, c.st.lat[j] + 0.008, s, a=a)
        else:
            gauge(c.st.lon[j] - 0.5 * s, c.st.lat[j] + 0.008, s, a=a)
    return a


ex = json.load(open(os.path.join(ROOT, "results", "cascade_examples.json")))

# ------------------------------------------------------------------ left: simulator, pilot, classifier
ax.add_patch(FancyBboxPatch((0.06, 0.06), 2.32, H - 0.12, boxstyle="round,pad=0,rounding_size=0.08", fc="#f6f6f6", ec="#999999", lw=0.8, zorder=1))
ax.text(1.22, H - 0.2, "Cascade simulator", ha="center", va="center", fontsize=8, fontweight="bold")
mh = 1.06 * ASP
big_map([0.13 / W, (H - 0.34 - mh) / H, 1.06 / W, mh / H], ex[2])
big_map([1.25 / W, (H - 0.34 - mh) / H, 1.06 / W, mh / H], ex[1])
yl = H - 0.34 - mh - 0.2
gauge(0.36, yl - 0.02, 0.17); ax.text(0.58, yl + 0.03, "overload", va="center", fontsize=6.5)
bolt(1.42, yl - 0.05, 0.19); ax.text(1.63, yl + 0.03, "lightning", va="center", fontsize=6.5)

# pilot runs: a fan of thumbnails of real pilot days
yp = 0.46
X = 1.9 * rng.standard_normal((400, D)); dn = c.simulate(X); k = dn.sum(1)
pick = list(np.flatnonzero(k == 0)[:1]) + list(np.flatnonzero((k >= 1) & (k <= 2))[:1]) + list(np.flatnonzero(k >= 4)[:2])
for i, idx in enumerate(pick):
    tile(0.2 + 0.13 * i, yp + 0.05 * (3 - i), 0.5, dn[idx], z=10 + 3 * i)
ax.text(0.64, 0.2, "pilot runs", ha="center", va="center", fontsize=7)
arrow(1.22, yl - 0.1, 0.72, yp + 0.62, rad=0.15)
arrow(1.12, yp + 0.22, 1.27, yp + 0.22)
for i in range(3):
    tree(1.3 + 0.31 * i, yp + 0.06, 0.27, shade=[(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)][i], z=20)
ax.text(1.75, 0.2, "classifier", ha="center", va="center", fontsize=7)
bell(2.14, yp + 0.5, 0.15)

# ------------------------------------------------------------------ right: the two lanes
LX, LW = 2.62, W - 2.62 - 0.06
lanes = [(H / 2 + 0.03, BLUE, "#EAF3FA", "Learned control variate"), (0.06, GREEN, "#E8F6F1", "Learned importance sampling")]
LH = H / 2 - 0.09
arrow(2.38, 0.98, LX, 0.9, col=GREY); arrow(2.38, 1.25, LX, H / 2 + 0.7, col=GREY, rad=-0.2)


def space(x, y, w, h, kind):
    """Input space sketch: origin, learned set regions, and where the days are drawn."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc="white", ec="#aaaaaa", lw=0.6, zorder=5))
    ox, oy = x + 0.32 * w, y + 0.36 * h
    ax.plot([x + 0.05 * w, x + 0.95 * w], [oy, oy], color="#999999", lw=0.5, zorder=6)
    ax.plot([ox, ox], [y + 0.06 * h, y + 0.94 * h], color="#999999", lw=0.5, zorder=6)
    regs = [np.array([(0.62, 0.98), (0.98, 0.98), (0.98, 0.56), (0.86, 0.6), (0.72, 0.72), (0.62, 0.82)]),
            np.array([(0.74, 0.04), (0.98, 0.04), (0.98, 0.34), (0.86, 0.3), (0.78, 0.18)])]
    doms = [(0.635, 0.815), (0.745, 0.06)]
    for r in regs:
        ax.add_patch(Polygon(r * (w, h) + (x, y), closed=True, fc="#F5C6A5", ec=RED, lw=0.7, zorder=7))
    g = np.random.default_rng(3)
    if kind == "cv":
        p = np.column_stack([ox + 0.085 * w * g.standard_normal(34), oy + 0.12 * h * g.standard_normal(34)])
        ax.scatter(p[:, 0], p[:, 1], s=4, color="#444444", zorder=8, lw=0)
    else:
        for dx, dy in doms:
            px, py = x + dx * w, y + dy * h
            ax.plot([ox, px], [oy, py], color=GREEN, lw=0.7, zorder=8)
            p = np.column_stack([px + 0.02 * w + 0.07 * w * g.standard_normal(17), py + 0.1 * h * g.standard_normal(17)])
            p = p[(p[:, 0] > x + 0.03 * w) & (p[:, 0] < x + 0.97 * w) & (p[:, 1] > y + 0.04 * h) & (p[:, 1] < y + 0.96 * h)]
            ax.scatter(p[:, 0], p[:, 1], s=4, color=GREEN, zorder=9, lw=0)
            ax.scatter([px], [py], s=16, color=GREEN, edgecolor="white", lw=0.6, zorder=10)


def chip(x, y, text, col):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.5, color="white", fontweight="bold", zorder=30,
            bbox=dict(boxstyle="round,pad=0.3", fc=col, ec="none"))


# days actually drawn: nominal law for the control variate, a shifted law for importance sampling
Xn = rng.standard_normal((6, D)); dnn = c.simulate(Xn)
mu = np.zeros(D); mu[0] = 2.9
Xi = mu + rng.standard_normal((400, D)); dni = c.simulate(Xi); ki = dni.sum(1)
wi = np.exp(-Xi @ mu + 0.5 * mu @ mu)
sel = list(np.flatnonzero(ki >= 3)[:4]) + list(np.flatnonzero(ki == 0)[:1]) + list(np.flatnonzero((ki >= 1) & (ki < 3))[:1])
sel = [sel[i] for i in (0, 4, 1, 2, 5, 3)]

for ly, col, bg, name in lanes:
    ax.add_patch(FancyBboxPatch((LX, ly), LW, LH, boxstyle="round,pad=0,rounding_size=0.08", fc=bg, ec=col, lw=1.0, zorder=1))
    ax.text(LX + 0.1, ly + LH - 0.14, name, ha="left", va="center", fontsize=8, fontweight="bold", color=col)
    cvlane = col == BLUE
    sy, sh = ly + 0.12, LH - 0.42
    space(LX + 0.1, sy, 1.0, sh, "cv" if cvlane else "is")
    arrow(LX + 1.14, sy + sh / 2, LX + 1.32, sy + sh / 2, col=col)
    bw = 0.98                                          # width of the estimate box
    tx0 = LX + 1.36; gap = 0.05
    tw = (LX + LW - 0.08 - bw - 0.22 - tx0 - 2 * gap) / 3; th = tw * ASP
    rowgap = (sh - 2 * th) if cvlane else (sh - 2 * th - 0.14)
    for i in range(6):
        r, q = divmod(i, 3)
        bx = tx0 + q * (tw + gap)
        by = sy + sh - th - r * (th + max(rowgap, 0.03) + (0 if cvlane else 0.07))
        down = dnn[i] if cvlane else dni[sel[i]]
        tile(bx, by, tw, down, frame=col)
        if cvlane:
            if i == 4:                          # the classifier raises an alarm on this day
                bell(bx + tw - 0.15, by + th - 0.16, 0.15, z=40)
        else:                                   # weight of the day: likelihood ratio, on a log scale
            lw_ = np.log10(wi[sel[i]]); frac = np.clip((lw_ + 8) / 7, 0.06, 1)
            ax.add_patch(FancyBboxPatch((bx, by - 0.075), tw, 0.045, boxstyle="square,pad=0", fc="white", ec="#999999", lw=0.4, zorder=12))
            ax.add_patch(FancyBboxPatch((bx, by - 0.075), tw * frac, 0.045, boxstyle="square,pad=0", fc=col, ec="none", zorder=13))
    lab = "ordinary days" if cvlane else "targeted days, weighted"
    ax.text(LX + LW - 0.1, ly + LH - 0.14, lab, ha="right", va="center", fontsize=6.8, color=col, style="italic")
    ex_ = tx0 + 3 * (tw + gap) - gap + 0.03
    arrow(ex_, sy + sh / 2, ex_ + 0.16, sy + sh / 2, col=col)
    bx0 = LX + LW - 0.08 - bw
    cxm = bx0 + bw / 2
    ax.add_patch(FancyBboxPatch((bx0, sy), bw, sh, boxstyle="round,pad=0,rounding_size=0.04", fc="white", ec=col, lw=1.0, zorder=5))
    if cvlane:
        ax.text(cxm, sy + sh * 0.68, r"$\hat\theta=\bar{Y}-\beta(\bar{C}-\gamma)$", ha="center", va="center", fontsize=6.6, zorder=20)
        chip(cxm, sy + sh * 0.26, r"runs $\propto 1/\theta$", col)
    else:
        ax.text(cxm, sy + sh * 0.68, r"$\hat\theta=\frac{1}{n}\sum \ell(X_i)\,Y_i$", ha="center", va="center", fontsize=6.6, zorder=20)
        chip(cxm, sy + sh * 0.26, "runs flat in " + r"$\theta$", col)

os.makedirs(os.path.join(ROOT, "figs"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "figs", "fig_framework.pdf"), dpi=300)
fig.savefig(os.path.join(ROOT, "figs", "fig_framework.png"), dpi=300, transparent=False, facecolor="white")
print("weights of the six importance-sampled days:", np.round(np.log10(wi[sel]), 1), "feeders down:", ki[sel], "nominal days down:", dnn.sum(1))
