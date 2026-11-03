"""Figure 3 (DRAFT): content-free calibration of the present/past gate on four backbones (OVO-Bench, 1468 questions).

Left: distribution of the gate score log-odds(p_past) for backward questions (EPM, ASI, HLD; should read the past)
and real-time questions (should not), mirrored per backbone. Dashed line = fixed threshold 0.5; black tick = the
label-free threshold p_cf (mean p_past of "N/A", "", "[MASK]"; small triangles = the three content-free questions).
Right: share of questions sent to the past path with the fixed threshold (hollow) and with p_cf (filled).
All numbers are read from the route runs. Uses the round-17 reruns (route.json field "lo" = log P(past) - log P(now),
exact) when present; otherwise falls back to logit(p_past) from the 5-decimal p_past, which saturates for Qwen3.

usage (host): python3 paper/figs/fig3_calibration.py  ->  paper/figs/fig3_calibration.{pdf,png}
"""

import glob
import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fig1_teaser import AXIS, GRID, INK, INK2, MUTED, OURS, RUNS  # noqa: E402

ANNO = "/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json"
BACKBONES = [("Qwen2.5-VL-7B", "route_cal_p0"), ("Qwen3-VL-8B", "route_q3_cal_p0"),
             ("Qwen3-VL-4B", "route_q4_cal_p0"), ("LLaVA-OV-7B", "route_lov_cal_p0")]
BT = {"EPM", "ASI", "HLD"}
NOW = "#6f6e69"  # neutral ink for real-time questions (grey = "frames only" path, as in figure 2)
EPS = 6e-6


def logit(p):
    p = min(max(p, EPS), 1 - EPS)
    return math.log(p / (1 - p))


def data():
    task = {f"{a['task']}:{a['id']}": a["task"] for a in json.load(open(ANNO))}
    out = []
    for name, run in BACKBONES:
        new = sorted(glob.glob(f"{RUNS}/*-{run}_lo/out"))
        d = new[-1] if new else sorted(glob.glob(f"{RUNS}/*-{run}/out"))[-1]
        route, calib = json.load(open(f"{d}/route.json")), json.load(open(f"{d}/calib.json"))
        cf = calib["content_free"]
        cf = json.loads(cf.replace("'", '"')) if isinstance(cf, str) else cf
        p_cf = calib["p_cf"]
        exact = all("lo" in v for v in route.values())
        g = {"past": [], "now": []}
        for k, v in route.items():
            g["past" if task[k] in BT else "now"].append((v["p_past"], v["lo"] if exact else logit(v["p_past"])))
        rate = {(grp, rule): 100 * np.mean([x >= th for x, _ in xs])
                for grp, xs in g.items() for rule, th in (("fixed", 0.5), ("cal", p_cf))}
        cf_lo = list(calib["lo_content_free"].values()) if exact else [logit(c) for c in cf.values()]
        out.append(dict(name=name, p_cf=p_cf, cf_lo=cf_lo, lo={k: [y for _, y in xs] for k, xs in g.items()},
                        rate=rate, n={k: len(xs) for k, xs in g.items()}, exact=exact,
                        run=os.path.basename(os.path.dirname(d))))
    return out


def ridge(ax, rows):
    exact = all(r["exact"] for r in rows)
    span = max(abs(x) for r in rows for xs in r["lo"].values() for x in xs)
    LIM = float(min(30, max(12, math.ceil(span)))) if exact else 12.0
    bins = np.linspace(-LIM - 0.5, LIM + 0.5, 61 if exact else 50)
    H = 0.40
    for i, r in enumerate(rows):
        y0 = len(rows) - 1 - i
        hp, _ = np.histogram(np.clip(r["lo"]["past"], -LIM, LIM), bins)
        hn, _ = np.histogram(np.clip(r["lo"]["now"], -LIM, LIM), bins)
        hp, hn = hp / hp.sum(), hn / hn.sum()
        sc = H / max(hp.max(), hn.max())
        w = bins[1] - bins[0]
        ax.bar(bins[:-1], hp * sc, width=w * 0.86, bottom=y0, align="edge", color=OURS, lw=0, zorder=2)
        ax.bar(bins[:-1], -hn * sc, width=w * 0.86, bottom=y0, align="edge", color=NOW, lw=0, zorder=2)
        ax.plot([-LIM - 0.6, LIM + 0.6], [y0, y0], color=AXIS, lw=0.5, zorder=3)
        t = logit(r["p_cf"])
        ax.plot([t, t], [y0 - 0.46, y0 + 0.5], color=INK, lw=1.1, zorder=4, solid_capstyle="butt")
        m, e = f"{r['p_cf']:.1e}".split("e")
        lab = f"p$_{{cf}}$ = {r['p_cf']:.2f}" if r["p_cf"] > 0.01 else f"p$_{{cf}}$ = {m}×10$^{{{int(e)}}}$"
        right = t < 0
        ax.text(t + (0.35 if right else -0.35), y0 + 0.43, lab, fontsize=5.2, color=INK, ha="left" if right else "right",
                va="center", zorder=5)
        for c in r["cf_lo"]:
            ax.scatter([min(max(c, -LIM), LIM)], [y0 - 0.47], marker="^", s=7, c=INK2, lw=0, zorder=5, clip_on=False)
        ax.text(-LIM - 1.2, y0, r["name"], fontsize=6.0, color=INK, ha="right", va="center", fontweight="bold")
    ax.axvline(0, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0, len(rows) - 0.42, "fixed 0.5", fontsize=5.2, color=MUTED, ha="center", va="bottom")
    ax.set_xlim(-LIM - 0.6, LIM + 0.6)
    ax.set_ylim(-0.55, len(rows) - 0.3)
    ax.set_yticks([])
    step = 10 if LIM > 20 else 4
    ticks = [t for t in range(-int(LIM), int(LIM) + 1) if t % step == 0]
    labels = [f"{t:d}".replace("-", "−") for t in ticks]
    if span > LIM:
        labels[0], labels[-1] = "≤" + labels[0], "≥" + labels[-1]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlabel("gate score: log-odds of p$_{past}$(question)", fontsize=6.2, color=INK2, labelpad=2)
    ax.tick_params(labelsize=5.6, colors=INK2, length=2, width=0.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.text(LIM + 0.5, len(rows) - 0.42, "▲ backward (past) questions", fontsize=5.2, color=OURS, ha="right",
            va="bottom", fontweight="bold")
    ax.text(LIM + 0.5, len(rows) - 0.62, "▼ real-time questions", fontsize=5.2, color=NOW, ha="right",
            va="bottom", fontweight="bold")
    ax.set_title("(a) Gate scores and thresholds", fontsize=6.8, fontweight="bold", color=INK, loc="left", pad=3)


def rates(ax, rows):
    n = len(rows)
    for i, r in enumerate(rows):
        y0 = n - 1 - i
        for grp, col, dy in (("past", OURS, 0.13), ("now", NOW, -0.13)):
            a, b = r["rate"][(grp, "fixed")], r["rate"][(grp, "cal")]
            y = y0 + dy
            if abs(b - a) > 1.5:
                ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15",
                                                                          color=col, lw=0.7, shrinkA=2.5, shrinkB=2.5))
            ax.scatter([a], [y], s=16, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3)
            ax.scatter([b], [y], s=16, c=col, edgecolors="white", linewidths=0.6, zorder=4)
            ax.text(b + (2.5 if b >= a else -2.5), y, f"{b:.0f}%", fontsize=5.0, color=INK2, va="center",
                    ha="left" if b >= a else "right")
        ax.plot([0, 100], [y0 - 0.5, y0 - 0.5], color=GRID, lw=0.4) if i < n - 1 else None
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, n - 0.3)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("questions sent to the past path (%)", fontsize=6.2, color=INK2, labelpad=2)
    ax.tick_params(labelsize=5.6, colors=INK2, length=2, width=0.5)
    ax.grid(True, axis="x", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.scatter([58], [n - 0.42], s=12, facecolors="white", edgecolors=INK2, linewidths=0.8, clip_on=False)
    ax.text(61, n - 0.42, "fixed 0.5", fontsize=5.0, color=INK2, va="center")
    ax.scatter([80], [n - 0.42], s=12, c=INK2, clip_on=False)
    ax.text(83, n - 0.42, "p$_{cf}$", fontsize=5.0, color=INK2, va="center")
    ax.set_title("(b) Read rate", fontsize=6.8, fontweight="bold", color=INK, loc="left", pad=3)


def main():
    rows = data()
    for r in rows:
        print(r["name"], r["run"], "exact" if r["exact"] else "rounded", f"p_cf={r['p_cf']:.3g}",
              {f"{k[0]}/{k[1]}": round(v, 1) for k, v in r["rate"].items()})
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(7.0, 2.6))
    ridge(fig.add_axes([0.125, 0.15, 0.53, 0.76]), rows)
    rates(fig.add_axes([0.705, 0.15, 0.27, 0.76]), rows)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig3_calibration.{ext}"), dpi=300, facecolor="white")
    print("saved", os.path.join(HERE, "fig3_calibration.{pdf,png}"))


if __name__ == "__main__":
    main()
