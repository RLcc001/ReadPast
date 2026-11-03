"""Figure 2 (method overview, DRAFT): ReadPast on a real OVO-Bench question (EPM:173, Qwen3-VL-8B).

Top, write side: while the video streams, the same VLM captions every finished 16-s clip into a text timeline
(question-agnostic; the clip still in progress is not in the timeline yet). Bottom, read side: when a question
arrives, a question-only gate decides past/now with the content-free threshold p_cf; "past" adds the timeline to the
recent 4 frames. Frames come from the video, caption excerpts are verbatim substrings of the cached captions, and all
numbers (p_past, p_cf, answers, probabilities, timeline tokens, read rate) are read from the run records.

usage (host): python3 paper/figs/fig2_method.py  ->  paper/figs/fig2_method.{pdf,png}
frames: paper/figs/assets/epm173/ (paper/figs/extract_frames.py, run inside ren_stream_g23)
"""

import glob
import json
import os
import sys
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fig1_teaser import (BAD, GOOD, GRID, INK, INK2, MUTED, OURS, OURS_TINT, RUNS, arrow, box,  # noqa: E402
                         load)

KEY, RUN_BASE, RUN_OURS, ROUTE = "EPM:173", "r4_q3", "cap16_q3", "route_q3_cal_p0"
CAPS = "/data/REN/streamkv/cache/captions_q3_s16/Ego4D__clips__a99baf07-ce1c-4f73-ab20-ed0dfc079510.mp4.json"
ANNO = "/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json"
ASSETS = os.path.join(HERE, "assets", "epm173")
EXCERPTS = [  # verbatim substrings of the three finished captions (asserted below)
    "selecting and placing a pomegranate into a clear plastic bag",
    "walks along multiple shelves filled with various fruits like pomegranates, papayas, pineapples",
    "browsing a well-stocked, curved produce aisle in a supermarket, focusing on various fruits",
]
SEG_THUMBS = [(4, 12), (20, 28), (36, 44)]
PARTIAL_THUMB, RECENT = 52, (58, 59, 60, 60.9)
TW, TH, TG = 44, 33, 3  # thumbnail width / height / gap (axis units = 0.01 in)


def data():
    base, ours = load(RUN_BASE, RUNS)[KEY], load(RUN_OURS, RUNS)[KEY]
    rdir = sorted(glob.glob(f"{RUNS}/*-{ROUTE}/out"))[-1]
    route, calib = json.load(open(f"{rdir}/route.json")), json.load(open(f"{rdir}/calib.json"))
    anno = next(a for a in json.load(open(ANNO)) if f"{a['task']}:{a['id']}" == KEY)
    caps = json.load(open(CAPS))["captions"][:3]
    for c, e in zip(caps, EXCERPTS):
        assert e in c["text"], e
    read = sum(v["p_past"] >= calib["p_cf"] for v in route.values()) / len(route)
    p = lambda r: json.loads(r["p_opts"]) if isinstance(r["p_opts"], str) else r["p_opts"]  # noqa: E731
    return dict(anno=anno, caps=caps, base=base, ours=ours, p_base=p(base), p_ours=p(ours),
                p_past=route[KEY]["p_past"], p_cf=calib["p_cf"], read=read, tl=int(ours["timeline_tokens"]))


def thumb(ax, x, y, t, ec=None, lw=0.0, alpha=1.0):
    im = Image.open(os.path.join(ASSETS, f"t{t:06.2f}.jpg"))
    ax.imshow(im, extent=(x, x + TW, y, y + TH), zorder=3, alpha=alpha, interpolation="lanczos")
    if ec:
        ax.add_patch(Rectangle((x, y), TW, TH, fc="none", ec=ec, lw=lw, zorder=4))


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m}×10$^{{{int(e)}}}$"


def main():
    d = data()
    fig = plt.figure(figsize=(7.0, 3.05))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 700)
    ax.set_ylim(0, 305)
    ax.axis("off")

    # ---------------- write side ----------------
    ax.text(8, 294, "① Write", fontsize=7.2, fontweight="bold", color=INK, va="center")
    for i, line in enumerate(["online, every", "16 s, before", "any question"]):
        ax.text(8, 283 - 8 * i, line, fontsize=5.2, color=INK2, va="center")
    ty = 222
    seg_x = [86, 188, 290]
    for (t0, t1), sx in zip(SEG_THUMBS, seg_x):
        thumb(ax, sx, ty, t0)
        thumb(ax, sx + TW + TG, ty, t1)
    px = 392
    thumb(ax, px, ty, PARTIAL_THUMB, alpha=0.55)
    rx0 = 450
    for i, t in enumerate(RECENT):
        thumb(ax, rx0 + i * (TW + TG), ty, t)
    rx1 = rx0 + 4 * TW + 3 * TG
    box(ax, rx0 - 3, ty - 3, rx1 - rx0 + 6, TH + 6, "none", INK, lw=0.9, r=2, z=5)
    ax.text((rx0 + rx1) / 2, ty + TH + 9, "current view: recent 4 frames", ha="center", va="center", fontsize=5.6,
            color=INK, fontweight="bold")
    for sx, (lab) in zip(seg_x + [px], ["0–16 s", "16–32 s", "32–48 s", "48– s"]):
        w = 2 * TW + TG if sx != px else TW
        ax.plot([sx, sx + w], [ty + TH + 4] * 2, color=MUTED, lw=0.5)
        ax.text(sx + w / 2, ty + TH + 9, lab, ha="center", va="center", fontsize=5.0, color=INK2)
    # time axis
    ay = ty - 7
    ax.annotate("", xy=(rx1 + 22, ay), xytext=(82, ay),
                arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18", color=MUTED, lw=0.6))
    ax.text(rx1 + 24, ay, f"now\n{d['anno']['realtime']} s", fontsize=5.0, color=MUTED, va="center", ha="left",
            linespacing=1.0)

    # caption entries = text timeline
    cy, ch, cw = 136, 48, 2 * TW + TG
    for c, e, sx in zip(d["caps"], EXCERPTS, seg_x):
        arrow(ax, (sx + cw / 2, ay - 2), (sx + cw / 2, cy + ch + 1), color=OURS, lw=0.6)
        box(ax, sx, cy, cw, ch, OURS_TINT, OURS, lw=0.5, r=2)
        ax.text(sx + 3, cy + ch - 6, f"[{c['t0']}–{c['t1']} s]", fontsize=5.0, color=OURS, fontweight="bold",
                va="center")
        body = textwrap.fill("…" + e + "…", 27)
        ax.text(sx + 3, cy + ch - 13, body, fontsize=4.55, color=INK, va="top", linespacing=1.15)
    box(ax, px, cy, TW, ch, "white", MUTED, lw=0.5, r=2)
    ax.patches[-1].set_linestyle((0, (2, 1.5)))
    ax.text(px + TW / 2, cy + ch / 2, "clip in\nprogress:\nnot written\nyet", fontsize=4.6, color=MUTED,
            ha="center", va="center", linespacing=1.15, style="italic")
    ax.text(86, cy - 9, f"text timeline: {len(d['caps'])} entries, {d['tl']} tokens "
            "(the same VLM writes each entry)", fontsize=5.2, color=OURS, va="center", style="italic")
    ax.text(86, cy - 18, "only text is kept from past clips; every later question reuses the same timeline",
            fontsize=5.0, color=INK2, va="center")
    ax.plot([2, 698], [101, 101], color=GRID, lw=0.6)

    # ---------------- read side ----------------
    ax.text(8, 90, "② Read", fontsize=7.2, fontweight="bold", color=INK, va="center")
    for i, line in enumerate(["when a", "question", "arrives"]):
        ax.text(8, 79 - 8 * i, line, fontsize=5.2, color=INK2, va="center")
    a = d["anno"]
    qx, qy, qw, qh = 86, 22, 188, 66
    box(ax, qx, qy, qw, qh, "white", MUTED, lw=0.5, r=3)
    ax.text(qx + 5, qy + qh - 9, f"Q ({a['realtime']} s): {re.sub(r'\bi\b', 'I', a['question'].strip())}", fontsize=5.8,
            color=INK, va="center", fontweight="bold")
    for i, o in enumerate(a["options"]):
        ax.text(qx + 8 + (i % 2) * 96, qy + qh - 24 - (i // 2) * 10, f"({chr(65 + i)}) {o}", fontsize=5.0,
                color=INK2, va="center")
    ax.text(qx + 5, qy + 8, f"{d['read'] * 100:.0f}% of OVO questions go to the past path", fontsize=4.8,
            color=MUTED, va="center", style="italic")

    gx, gy, gw, gh = 294, 40, 112, 48
    box(ax, gx, gy, gw, gh, OURS, OURS, r=3)
    ax.text(gx + gw / 2, gy + gh - 10, "past or now?", ha="center", va="center", fontsize=6.4, color="white",
            fontweight="bold")
    ax.text(gx + gw / 2, gy + gh - 23, f"p$_{{past}}$(Q) = {d['p_past']:.4f}", ha="center", va="center",
            fontsize=5.2, color="white")
    ax.text(gx + gw / 2, gy + gh - 33, f"≥ p$_{{cf}}$ = {sci(d['p_cf'])}  → past", ha="center",
            va="center", fontsize=5.2, color="white")
    ax.text(gx + gw / 2, gy - 9, "p$_{cf}$: p$_{past}$ of content-free questions", ha="center", va="center",
            fontsize=4.7, color=INK2)
    ax.text(gx + gw / 2, gy - 17, "(“N/A”, “”, “[MASK]”): no labels, no tuning",
            ha="center", va="center", fontsize=4.7, color=INK2)
    arrow(ax, (qx + qw + 1, qy + qh / 2 + 9), (gx - 1, gy + gh / 2))

    vx, vy, vw, vh = 520, 42, 46, 30
    box(ax, vx, vy, vw, vh, "white", INK2, lw=0.7)
    ax.text(vx + vw / 2, vy + vh / 2, "VLM", ha="center", va="center", fontsize=6.8, color=INK)
    arrow(ax, (gx + gw + 1, gy + gh / 2 + 6), (vx - 1, vy + vh / 2 + 5), color=OURS, lw=0.9)
    ax.text(vx - 6, gy + gh / 2 + 13, "past: + timeline", ha="right", va="center", fontsize=5.0, color=OURS,
            fontweight="bold")
    arrow(ax, (gx + gw + 1, gy + gh / 2 - 8), (vx - 1, vy + vh / 2 - 6), color=MUTED, lw=0.6, ls=(0, (2, 1.5)))
    ax.text(gx + gw + 6, gy + gh / 2 - 16, "now: frames only", ha="left", va="center", fontsize=5.0, color=MUTED)
    # timeline -> VLM (dashed), current view -> VLM
    arrow(ax, (290 + cw - 10, cy - 1), (gx + gw + 30, gy + gh / 2 + 5.5), color=OURS, lw=0.7, ls=(0, (2, 1.5)), rad=0.0)
    arrow(ax, ((rx0 + rx1) / 2 + 20, ty - 4 - 6), (vx + vw - 10, vy + vh + 1), color=INK2, lw=0.6)

    # answers
    ox = 592
    gt = a["gt"]
    rows = [("ReadPast (ours)", d["ours"], d["p_ours"], OURS), ("SimpleStream", d["base"], d["p_base"], MUTED)]
    for i, (lab, rec, p, col) in enumerate(rows):
        yy = vy + vh - 2 - i * 22
        k = ord(rec["response"]) - 65
        ok = k == gt
        ax.text(ox, yy + 4, lab, fontsize=5.2, color=col, va="center", fontweight="bold" if i == 0 else "normal")
        ax.text(ox, yy - 5, f"({rec['response']}) {a['options'][k]}  p = {p[k]:.2f}", fontsize=5.2, color=INK,
                va="center")
        ax.text(ox - 8, yy - 0.5, "✓" if ok else "✗", fontsize=8, color=GOOD if ok else BAD, ha="center",
                va="center", fontweight="bold")
    arrow(ax, (vx + vw + 1, vy + vh / 2), (ox - 17, vy + vh / 2), color=INK2)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig2_method.{ext}"), dpi=300, facecolor="white")
    print("saved", os.path.join(HERE, "fig2_method.{pdf,png}"), {k: d[k] for k in ("p_past", "p_cf", "read", "tl")})


if __name__ == "__main__":
    main()
