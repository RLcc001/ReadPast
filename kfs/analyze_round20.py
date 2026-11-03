"""Round 20, offline part (plan in EXPERIMENTS.md): gate-threshold sweep and content-free input ablation, OVO 1468,
standard avg = (BT + RT) / 2. Report-only; the main method stays at the mean-probability threshold p_cf.

Gate: read the timeline iff lo >= t, lo = log P(past) - log P(now) from the round-17 route reruns `*_cal_p0_lo`
(exact log-odds). Answers are composed exactly as in round 17: the 187 questions whose options contain an abstention
choice use the `*_cinstr` runs (p_opts argmax; r4 or cap16 by the gate), the other 1281 use r4 / cap16 correctness.

Thresholds compared:
  fixed 0.5              t = 0
  single content-free    t = lo("N/A"), lo(""), lo("[MASK]")
  mean probability (ours) t = logit(mean_x p(x)) = logit(p_cf)            (Zhao et al. 2021 averaging)
  mean log-odds          t = mean_x lo(x)
  sweep                  t = logit(p_cf) + z * sd(lo), z in [-2, 2]; the best z uses OVO labels -> reference only

usage: python3 kfs/analyze_round20.py [--plot results/round20_threshold_sweep.png]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load as pload  # noqa: E402
from analyze_pilot import scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_round17 import CFG, corr, raw  # noqa: E402
from analyze_route import RUNS  # noqa: E402

K187 = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
BT = {"EPM", "ASI", "HLD"}
ZS = [round(-2 + 0.05 * i, 2) for i in range(81)]


def logit(p):
    return math.log(p / (1 - p))


def backbone(bb):
    cfg = BACKBONES[bb]
    sfx, _, route_run = CFG[bb]
    d = sorted(glob.glob(f"{RUNS}/*-{route_run}_lo/out"))[-1]
    route, calib = json.load(open(f"{d}/route.json")), json.load(open(f"{d}/calib.json"))
    lo = {k: v["lo"] for k, v in route.items()}
    cf = calib["lo_content_free"]
    base, mem = (pload(n, RUNS) for n in cfg["ovo"])
    cb, cm = raw("r4_cinstr" + sfx), raw("cap16_cinstr" + sfx)
    keys = sorted(base)
    tasks = {k: base[k]["task"] for k in keys}
    tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
    ref_gate = gates(cfg, "ovo", tl)[0]["(a) ours"]

    def evaluate(t):
        g = {k: lo[k] >= t for k in keys}
        c = {k: corr((cm if g[k] else cb)[k]) if k in K187 else int((mem if g[k] else base)[k]["correct"]) for k in keys}
        s = scores(c, tasks)
        bt = [k for k in keys if tasks[k] in BT]
        rt = [k for k in keys if tasks[k] not in BT]
        return dict(t=t, avg=s["avg"], rt=s["rt"], read_bt=100 * sum(g[k] for k in bt) / len(bt),
                    read_rt=100 * sum(g[k] for k in rt) / len(rt)), g

    t_ours = logit(calib["p_cf"])
    sd = statistics.pstdev(lo.values())
    rules = {"fixed 0.5": 0.0, **{f'single "{x}"': v for x, v in cf.items()},
             "mean log-odds": sum(cf.values()) / len(cf), "mean probability (ours)": t_ours}
    rows = {name: evaluate(t)[0] for name, t in rules.items()}
    sweep = [dict(z=z, **evaluate(t_ours + z * sd)[0]) for z in ZS]
    _, g_ours = evaluate(t_ours)
    mismatch = sum(g_ours[k] != ref_gate[k] for k in keys)
    return dict(rows=rows, sweep=sweep, sd=sd, t_ours=t_ours, mismatch=mismatch, n=len(keys))


def plot(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, "paper/figs")
    from fig1_teaser import AXIS, GRID, INK, INK2, MUTED, OURS

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(res), figsize=(7.0, 2.1), sharey=False)
    for ax, (bb, r) in zip(axes, res.items()):
        zs = [p["z"] for p in r["sweep"]]
        ax.plot(zs, [p["avg"] for p in r["sweep"]], color=INK2, lw=1.4, zorder=2)
        for name, row in r["rows"].items():
            z = (row["t"] - r["t_ours"]) / r["sd"]
            if not -2.05 <= z <= 2.05:
                continue
            ours = name.startswith("mean probability")
            fixed = name == "fixed 0.5"
            ax.scatter([z], [row["avg"]], s=30 if ours else 16, zorder=4, linewidths=0.8,
                       c=OURS if ours else ("white" if fixed else MUTED), edgecolors=OURS if ours else INK2,
                       marker="o" if not name.startswith("single") else "^")
        ax.set_title(bb.split(" (")[0], fontsize=7, color=INK, loc="left")
        ax.tick_params(labelsize=6, colors=INK2, length=2, width=0.5)
        ax.grid(True, color=GRID, lw=0.4)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(AXIS)
    axes[0].set_ylabel("OVO avg", fontsize=6.5, color=INK2)
    fig.supxlabel("threshold offset from p$_{cf}$ (in sd of the gate log-odds; fixed 0.5 is off-range for LLaVA-OV)",
                  fontsize=6, color=INK2, y=0.08)
    fig.text(0.99, 0.02, "● p$_{cf}$ (ours)   ○ fixed 0.5   ▲ single content-free input   — sweep", fontsize=6,
             color=INK2, ha="right")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200, facecolor="white")
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="")
    args = ap.parse_args()
    res = {}
    for bb in CFG:
        r = res[bb] = backbone(bb)
        print(f"\n=== {bb}  n={r['n']}  sd(lo)={r['sd']:.2f}  t_ours={r['t_ours']:.3f}  "
              f"gate mismatch vs round-17 (a): {r['mismatch']}")
        print(f"  {'threshold':26s} {'t':>8s} {'avg':>6s} {'RT':>6s} {'read BT':>8s} {'read RT':>8s}")
        for name, row in r["rows"].items():
            print(f"  {name:26s} {row['t']:8.2f} {row['avg']:6.2f} {row['rt']:6.2f} {row['read_bt']:7.0f}% "
                  f"{row['read_rt']:7.0f}%")
        best = max(r["sweep"], key=lambda p: p["avg"])
        inner = [p["avg"] for p in r["sweep"] if abs(p["z"]) <= 0.5 + 1e-9]
        print(f"  sweep: best z={best['z']:+.2f} avg {best['avg']:.2f} (uses OVO labels, reference only); "
              f"range within |z|<=0.5: {min(inner):.2f}-{max(inner):.2f}")
    os.makedirs("results", exist_ok=True)
    json.dump(res, open("results/round20_threshold.json", "w"), indent=1)
    if args.plot:
        plot(res, args.plot)


if __name__ == "__main__":
    main()
