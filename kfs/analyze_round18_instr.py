"""Round 18 ablation (pre-registered, EXPERIMENTS.md): D-HSM's abstention instruction on EVERY multiple-choice
question (D-HSM's own usage, `--abst_instr_when always`) vs only when the options already contain an abstention
option (ReadPast, `--abst_instr_when option`). Host-side, stdlib only.

The 187 questions with an abstention option get the identical prompt under both usages -> reuse round-17 `*_cinstr`
(scored by p_opts argmax, as in round 17). The other 1281 questions come from the `*_ainstr*` runs, scored like the
original runs (generated letter == ground truth).

  Delta = (gate (a) + conditional) - (gate (a) + always), standard OVO.
  Delta > 0 on >= 3/4 backbones -> "conditional trigger beats D-HSM's uniform usage"; else "equivalent".

usage: python3 kfs/analyze_round18_instr.py
"""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load as pload  # noqa: E402
from analyze_pilot import scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_round17 import CFG, corr, raw  # noqa: E402
from analyze_route import RUNS  # noqa: E402

K187 = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
K1281 = set(json.load(open("kfs/data/keys_ovo_noabst1281.json")))
TASKS = ["EPM", "ASI", "HLD", "OCR", "ACR", "ATR", "STU", "FPD", "OJR"]


def try_load(name):
    try:
        rows = pload(name, RUNS)
    except FileNotFoundError:
        return None
    return rows if K1281 <= set(rows) else None


def main():
    wins = ready = 0
    for bb, (sfx, _, _) in CFG.items():
        cfg = BACKBONES[bb]
        print(f"\n=== {bb}")
        cb, cm = raw("r4_cinstr" + sfx), raw("cap16_cinstr" + sfx)
        ab, am = try_load("r4_ainstr" + sfx), try_load("cap16_ainstr" + sfx)
        if ab is None or am is None:
            print("  always-instruction runs not ready")
            continue
        ready += 1
        base0, mem0 = (pload(n, RUNS) for n in cfg["ovo"])
        keys = sorted(base0)
        assert set(keys) == K187 | K1281
        tasks = {k: base0[k]["task"] for k in keys}
        tl = {k: mem0[k].get("timeline_tokens") or 0 for k in keys}
        ga = gates(cfg, "ovo", tl)[0]["(a) ours"]
        ss = {k: int(base0[k]["correct"]) for k in keys}
        ours = {k: int((mem0 if ga[k] else base0)[k]["correct"]) for k in keys}
        sp = {k: corr(cb[k]) for k in K187}  # SimpleStream on the 187 (same prompt under both usages)
        gp = {k: corr((cm if ga[k] else cb)[k]) for k in K187}
        rows = {
            "SimpleStream": ss,
            "SimpleStream + cond. instr": {k: sp[k] if k in K187 else ss[k] for k in keys},
            "SimpleStream + always instr": {k: sp[k] if k in K187 else ab[k]["correct"] for k in keys},
            "gate (a)": ours,
            "gate (a) + cond. instr": {k: gp[k] if k in K187 else ours[k] for k in keys},
            "gate (a) + always instr": {k: gp[k] if k in K187 else (am if ga[k] else ab)[k]["correct"] for k in keys},
        }
        ref = scores(ss, tasks)["avg"]
        print(f"  {'':30s} {'avg':>6s} {'(dSS)':>7s} {'RT':>6s} " + " ".join(f"{t:>5s}" for t in TASKS))
        for name, c in rows.items():
            s = scores(c, tasks)
            print(f"  {name:30s} {s['avg']:6.2f} {s['avg'] - ref:+7.2f} {s['rt']:6.2f} "
                  + " ".join(f"{s[t]:5.1f}" for t in TASKS))
        d = scores(rows["gate (a) + cond. instr"], tasks)["avg"] - scores(rows["gate (a) + always instr"], tasks)["avg"]
        wins += d > 0
        for lab, a, b in (("near 4 frames", base0, ab), ("cap16 (read all)", mem0, am)):
            flip = [k for k in K1281 if a[k]["response"] != b[k]["response"]]
            fixed = sum(b[k]["correct"] > a[k]["correct"] for k in flip)
            broke = sum(b[k]["correct"] < a[k]["correct"] for k in flip)
            print(f"  always vs none on the 1281, {lab}: {len(flip)} answers changed, {fixed} fixed, {broke} broken")
        print(f"  Delta (cond - always, gate (a)) = {d:+.2f}")
    if ready < 4:
        print(f"\n{ready}/4 backbones ready; verdict after all four")
        return
    print(f"\nDelta > 0 on {wins}/4 backbones -> "
          + ("conditional trigger beats D-HSM's uniform usage" if wins >= 3 else "report as equivalent"))


if __name__ == "__main__":
    main()
