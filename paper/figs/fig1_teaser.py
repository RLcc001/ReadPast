"""Figure 1 (teaser): three ways to give a streaming VLM its past, and where they land on OVO-Bench (Qwen3-VL-8B).

(a) recent window only (SimpleStream), (b) + history frames (visual memory), (c) ours: self-written text timeline,
read only when the question asks about the past. (d) real-time accuracy (RT) vs memory accuracy (mean of EPM, ASI),
arrows from SimpleStream. Numbers are computed from the run records (kfs/ analysis helpers), not typed in.

usage (host): python3 paper/figs/fig1_teaser.py  ->  paper/figs/fig1_teaser.{pdf,png}
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "kfs"))
from analyze_pilot import load, scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_route import RUNS  # noqa: E402

# palette (dataviz reference, validated all-pairs on #ffffff; aqua < 3:1 -> every point is direct-labelled)
OURS, VIS, TXT = "#2a78d6", "#eb6834", "#1baf7a"
OURS_TINT, VIS_TINT = "#cde2fb", "#fbe0d4"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FRAME, FRAME_HI = "#e9e8e3", "#b9b7ae"
GOOD, BAD = "#0ca30c", "#d03b3b"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.6})


def ovo_points():
    """(memory = mean(EPM, ASI), RT) for each configuration on Qwen3-VL-8B."""
    cfg = BACKBONES["Qwen3-VL-8B (dev)"]
    runs = {n: load(r, RUNS) for n, r in [("base", "r4_q3"), ("text", "cap16_q3"), ("f4", "clip4_ss_q3"),
                                           ("f16", "clip16_lr_ss_q3")]}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    tasks = {k: runs["base"][k]["task"] for k in keys}
    tl = {k: runs["text"][k].get("timeline_tokens") or 0 for k in keys}
    a = gates(cfg, "ovo", tl)[0]["(a) ours"]
    cor = {n: {k: r[k]["correct"] for k in keys} for n, r in runs.items()}
    cor["ours"] = {k: (cor["text"] if a[k] else cor["base"])[k] for k in keys}
    out = {}
    for n, c in cor.items():
        s = scores(c, tasks)
        out[n] = ((s["EPM"] + s["ASI"]) / 2, s["rt"])
    return out


def box(ax, x, y, w, h, fc, ec, lw=0.6, r=2.5, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, p, q, color=INK2, lw=0.7, ls="-", z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>,head_length=3,head_width=1.6", color=color, lw=lw, ls=ls,
                                 zorder=z, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"))


N_FR, FX0, FW, FH, FG = 12, 116, 9.5, 9, 2.3  # frames: count, first x, width, height, gap
STEP = FW + FG


def fmt(v):
    return f"{v:+.1f}".replace("-", "\u2212")


def frames(ax, y, hist=()):
    """A strip of frames; the last 4 are the current view, `hist` indices are retrieved history frames."""
    for i in range(N_FR):
        x = FX0 + i * STEP
        if i >= N_FR - 4:
            fc, ec, lw = FRAME_HI, INK2, 0.6
        elif i in hist:
            fc, ec, lw = VIS_TINT, VIS, 0.9
        else:
            fc, ec, lw = FRAME, FRAME, 0.4
        ax.add_patch(Rectangle((x, y), FW, FH, fc=fc, ec=ec, lw=lw, zorder=2))
    x_end = FX0 + N_FR * STEP - FG
    ax.add_patch(FancyArrowPatch((FX0, y - 4), (x_end + 4, y - 4), arrowstyle="-|>,head_length=2.5,head_width=1.2",
                                 color=MUTED, lw=0.5, shrinkA=0, shrinkB=0, zorder=1))
    ax.text(x_end + 6, y - 4, "now", ha="left", va="center", fontsize=5.2, color=MUTED)
    return x_end


def outcome(ax, x, y, present, past, p_note, m_note):
    for dy, ok, lab, note in ((10, present, "Present", p_note), (-8, past, "Past", m_note)):
        ax.text(x, y + dy, "\u2713" if ok else "\u2717", color=GOOD if ok else BAD, fontsize=8.5, ha="center",
                va="center", fontweight="bold")
        ax.text(x + 7, y + dy, lab, color=INK, fontsize=6.3, ha="left", va="center")
        ax.text(x + 7, y + dy - 7.5, note, color=INK2, fontsize=5.0, ha="left", va="center")


def vlm(ax, y):
    box(ax, 290, y - 11, 36, 22, "white", INK2)
    ax.text(308, y, "VLM", ha="center", va="center", fontsize=6.8, color=INK)


def title(ax, y, t, sub, color=INK):
    ax.text(2, y, t, fontsize=6.8, fontweight="bold", color=color, va="center")
    for i, line in enumerate(sub):
        ax.text(2, y - 9.5 - 7.5 * i, line, fontsize=5.6, color=INK2, va="center")


def schematic(ax, pts):
    ax.set_xlim(0, 420)
    ax.set_ylim(0, 270)
    ax.axis("off")
    ox = 345  # outcome column
    d = {n: (pts[n][0] - pts["base"][0], pts[n][1] - pts["base"][1]) for n in pts}

    # (a) recent window only
    y = 236
    title(ax, y + 13, "(a) Recent window", ["SimpleStream"])
    xe = frames(ax, y - 5)
    vlm(ax, y)
    arrow(ax, (xe + 18, y), (289, y))
    outcome(ax, ox, y, True, False, f"RT {pts['base'][1]:.1f}", "no memory")

    # (b) + history frames
    y = 166
    title(ax, y + 13, "(b) + History frames", ["visual memory"])
    xe = frames(ax, y - 5, hist=(1, 3, 5, 6))
    vlm(ax, y)
    arrow(ax, (xe + 18, y), (289, y))
    ax.text(FX0 + (N_FR * STEP) / 2, y - 18, "more frames dilute the current view", fontsize=5.0, color=VIS,
            ha="center", va="center", style="italic")
    outcome(ax, ox, y, False, True, f"RT {fmt(d['f4'][1])}", f"Mem {fmt(d['f4'][0])}")

    # (c) ours
    y = 50
    title(ax, 113, "(c) ReadPast (ours)", ["see the present,", "read the past"], color=OURS)
    fy = 22
    xe = frames(ax, fy)
    # self-written timeline over the earlier clips
    tx0, tw = FX0, 8 * STEP - FG
    box(ax, tx0, 40, tw, 18, OURS_TINT, OURS, lw=0.5, r=2)
    ax.text(tx0 + 3, 53, "[0\u201316s] opens the fridge", fontsize=4.6, color=INK, va="center")
    ax.text(tx0 + 3, 45, "[16\u201332s] takes out milk", fontsize=4.6, color=INK, va="center")
    ax.text(tx0, 63, "self-written timeline, every 16 s", fontsize=4.8, color=OURS, va="center", style="italic")
    for j in range(2):
        cx = FX0 + (4 * j + 1.5) * STEP + FW / 2
        arrow(ax, (cx, fy + FH + 0.5), (cx, 39.5), color=OURS, lw=0.5)
    # question -> gate
    box(ax, FX0, 82, 88, 20, "white", MUTED, lw=0.5, r=3)
    ax.text(FX0 + 44, 95.5, "Q: What did I take", ha="center", va="center", fontsize=5.0, color=INK)
    ax.text(FX0 + 44, 88, "out of the fridge?", ha="center", va="center", fontsize=5.0, color=INK)
    gx, gw = 214, 70
    box(ax, gx, 82, gw, 20, OURS, OURS, r=3)
    ax.text(gx + gw / 2, 95.5, "past or now?", ha="center", va="center", fontsize=5.8, color="white",
            fontweight="bold")
    ax.text(gx + gw / 2, 87.5, "question-only gate", ha="center", va="center", fontsize=4.7, color="white")
    arrow(ax, (FX0 + 89, 92), (gx - 1, 92))
    vlm(ax, y)
    arrow(ax, (gx + gw / 2 + 12, 81), (300, 61.5), color=OURS, rad=0.2)
    ax.text(gx + gw / 2 - 2, 72, "past \u2192 + timeline", fontsize=4.8, color=OURS, ha="right", va="center")
    arrow(ax, (tx0 + tw + 1, 49), (289, 52), color=OURS, lw=0.6, ls=(0, (2, 1.5)))
    arrow(ax, (xe + 18, fy + FH / 2), (289, 44), color=INK2)
    outcome(ax, ox, y, True, True, f"RT {fmt(d['ours'][1])}", f"Mem {fmt(d['ours'][0])}")

    for yy in (203, 131):
        ax.plot([2, 418], [yy, yy], color=GRID, lw=0.6, zorder=1)


def scatter(ax, pts):
    spec = [  # key, label, color, marker, size, label offset (points), ha
        ("base", "SimpleStream", MUTED, "o", 26, (0, -8), "center"),
        ("f4", "+4 history frames", VIS, "o", 26, (-6, 0), "right"),
        ("f16", "+16 low-res frames\n(same #tokens)", VIS, "s", 24, (-6, 0), "right"),
        ("text", "text memory,\nalways read", TXT, "o", 26, (0, -10), "center"),
        ("ours", "ReadPast (ours)", OURS, "o", 40, (0, 8), "center"),
    ]
    bx, by = pts["base"]
    for k, lab, col, mk, s, off, ha in spec:
        x, y = pts[k]
        if k != "base":
            ax.add_patch(FancyArrowPatch((bx, by), (x, y), arrowstyle="-|>,head_length=3,head_width=1.5", color=col,
                                         lw=0.8, alpha=0.55, shrinkA=4, shrinkB=4, zorder=2))
        ax.scatter([x], [y], s=s, c=col, marker=mk, edgecolors="white", linewidths=0.8, zorder=4)
        ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=5.6, color=INK,
                    ha=ha, va="center", linespacing=1.0,
                    fontweight="bold" if k == "ours" else "normal")
    ax.set_xlim(53.5, 64.5)
    ax.set_ylim(74.0, 83.2)
    ax.set_xlabel("Memory acc. (EPM, ASI)  →", fontsize=6.5, color=INK2, labelpad=2)
    ax.set_ylabel("Real-time acc. (RT)  →", fontsize=6.5, color=INK2, labelpad=2)
    ax.tick_params(labelsize=5.8, colors=INK2, length=2, width=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.grid(True, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(d) OVO-Bench, Qwen3-VL-8B", fontsize=7, fontweight="bold", color=INK, loc="left", pad=4)
    ax.text(64.3, 82.9, "better ↗", ha="right", va="top", fontsize=5.5, color=MUTED, style="italic")


def main():
    pts = ovo_points()
    for k, (m, rt) in pts.items():
        print(f"{k:5s} memory {m:.2f}  RT {rt:.2f}")
    fig = plt.figure(figsize=(7.0, 2.7))
    schematic(fig.add_axes([0.0, 0.0, 0.62, 1.0]), pts)
    scatter(fig.add_axes([0.695, 0.16, 0.295, 0.74]), pts)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig1_teaser.{ext}"), dpi=300, facecolor="white")
    print("saved", os.path.join(HERE, "fig1_teaser.{pdf,png}"))


if __name__ == "__main__":
    main()
