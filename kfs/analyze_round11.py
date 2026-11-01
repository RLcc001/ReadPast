"""Round 11 (pre-registered, EXPERIMENTS.md): abstention hint on Qwen2.5 (development), offline gate (a) combos.

L0 in {r4_ss, r4_note (--view_note)} x L1 in {cap16, cap16_note (--cap_note --view_note)}, routed by gate (a).
Freeze criteria: gate(a)+both notes vs gate(a) without notes: standard avg >= +1.0, mean(EPM, ASI) drop <= 1.0,
RT drop <= 0.3.

usage: python3 kfs/analyze_round11.py
"""

import sys

sys.path.insert(0, "kfs")
from analyze_pilot import BACKWARD, REALTIME, load, scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_route import RUNS, no_hld  # noqa: E402

CFG = BACKBONES["Qwen2.5-VL-7B (dev)"]
COLS = ["avg", "bt", "rt", "nohld", "mem"] + BACKWARD + REALTIME


def main():
    runs = {n: load(n, RUNS) for n in ("r4_ss", "r4_note", "cap16", "cap16_note")}
    keys = sorted(set.intersection(*(set(r) for r in runs.values())))
    tasks = {k: runs["r4_ss"][k]["task"] for k in keys}
    cor = {n: {k: r[k]["correct"] for k in keys} for n, r in runs.items()}
    tl = {k: runs["cap16"][k].get("timeline_tokens") or 0 for k in keys}
    a = gates(CFG, "ovo", tl)[0]["(a) ours"]
    rows = {}
    for l0 in ("r4_ss", "r4_note"):
        rows[f"never: {l0}"] = cor[l0]
    for l1 in ("cap16", "cap16_note"):
        rows[f"always: {l1}"] = cor[l1]
    for l0 in ("r4_ss", "r4_note"):
        for l1 in ("cap16", "cap16_note"):
            rows[f"gate(a): {l0} + {l1}"] = {k: (cor[l1] if a[k] else cor[l0])[k] for k in keys}
    s = {}
    for n, c in rows.items():
        x = scores(c, tasks)
        x["nohld"], x["mem"] = no_hld(x), (x["EPM"] + x["ASI"]) / 2
        s[n] = x
    print(f"OVO n={len(keys)}; gate (a) reads BT {100 * sum(a[k] for k in keys if tasks[k] in BACKWARD) / sum(tasks[k] in BACKWARD for k in keys):.0f}%"
          f" / RT {100 * sum(a[k] for k in keys if tasks[k] in REALTIME) / sum(tasks[k] in REALTIME for k in keys):.0f}%")
    print(f"{'':36s}" + "".join(f"{c:>7s}" for c in COLS))
    for n, x in s.items():
        print(f"{n:36s}" + "".join(f"{x[c]:7.2f}" for c in COLS))
    ref, new = s["gate(a): r4_ss + cap16"], s["gate(a): r4_note + cap16_note"]
    d = {c: new[c] - ref[c] for c in COLS}
    print("\ngate(a)+notes - gate(a):" + "".join(f" {c} {d[c]:+.2f}" for c in ("avg", "rt", "mem", "HLD", "nohld")))
    ok = d["avg"] >= 1.0 and d["mem"] >= -1.0 and d["rt"] >= -0.3
    print(f"criteria: avg {d['avg']:+.2f} >= +1.0 [{d['avg'] >= 1.0}], mem {d['mem']:+.2f} >= -1.0 [{d['mem'] >= -1.0}],"
          f" RT {d['rt']:+.2f} >= -0.3 [{d['rt'] >= -0.3}] -> {'FREEZE' if ok else 'not met'}")
    sr, sn = s["never: r4_ss"], s["never: r4_note"]
    print(f"SimpleStream + same note (r4_note - r4_ss): avg {sn['avg'] - sr['avg']:+.2f}, HLD {sn['HLD'] - sr['HLD']:+.2f},"
          f" RT {sn['rt'] - sr['rt']:+.2f}; ours+notes - r4_note: avg {new['avg'] - sn['avg']:+.2f}"
          f" vs ours - r4_ss: {ref['avg'] - sr['avg']:+.2f}")


if __name__ == "__main__":
    main()
