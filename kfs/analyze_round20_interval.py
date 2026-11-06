"""Round 20, GPU part (plan in EXPERIMENTS.md): caption interval s in {8, 16, 32} s (8 frames per segment), OVO 1468,
standard avg = (BT + RT) / 2. Report-only; the main method stays at s = 16.

Composition exactly as round 17: gate (a) decisions (route.json, independent of the timeline) are reused; the 187
questions whose options contain an abstention choice use the conditional-instruction runs (r4_cinstr when the gate says
"now", the s-specific *cinstr* run when it says "past"); the other 1281 use r4 / the s-specific always-read run.
Check: s = 16 reproduces the round-17 numbers (70.78 / 73.34).

usage: python3 kfs/analyze_round20_interval.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import BACKWARD, REALTIME  # noqa: E402
from analyze_pilot import load as pload  # noqa: E402
from analyze_pilot import scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_round17 import OFFLINE, corr, raw  # noqa: E402
from analyze_route import RUNS  # noqa: E402

K187 = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
CFG = {  # backbone -> (cinstr suffix, {s: (always-read run, cinstr run, caption cache)})
    "Qwen2.5-VL-7B (dev)": ("", {8: ("r20cap8_eval_q25", "r20cinstr8_eval_q25", "captions_q25_s8f8"),
                                 16: ("cap16", "cap16_cinstr", "captions_q25_s16"),
                                 32: ("r20cap32_eval_q25", "r20cinstr32_eval_q25", "captions_q25_s32")}),
    "Qwen3-VL-8B (dev)": ("_q3", {8: ("r20cap8_eval_q3", "r20cinstr8_eval_q3", "captions_q3_s8"),
                                  16: ("cap16_q3", "cap16_cinstr_q3", "captions_q3_s16"),
                                  32: ("r20cap32_eval_q3", "r20cinstr32_eval_q3", "captions_q3_s32")}),
}


def n_captions(cache):
    d = f"cache/{cache}"
    return sum(len(json.load(open(f"{d}/{f}"))["captions"]) for f in os.listdir(d) if f.endswith(".json"))


def main():
    for bb, (sfx, runs) in CFG.items():
        cfg = BACKBONES[bb]
        base = pload(cfg["ovo"][0], RUNS)
        keys = sorted(base)
        tasks = {k: base[k]["task"] for k in keys}
        cb = raw("r4_cinstr" + sfx)
        ss = {k: int(base[k]["correct"]) for k in keys}
        ref = scores(ss, tasks)
        print(f"\n=== {bb}   SimpleStream avg {ref['avg']:.2f}  BT {ref['bt']:.2f}  RT {ref['rt']:.2f}")
        cols = ["avg", "bt", "rt"] + BACKWARD + REALTIME
        print(f"  {'s':>3} {'rule':26s} " + " ".join(f"{c:>6s}" for c in cols) + "  read%  TL-tok(med)  #caps")
        for s, (mem_n, cm_n, cache) in runs.items():
            mem, cm = pload(mem_n, RUNS), raw(cm_n)
            assert set(mem) == set(keys) and set(cm) == K187, (mem_n, cm_n)
            tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
            ga = gates(cfg, "ovo", tl)[0]["(a) ours"]
            rows = {
                "always read + cond. instr": {k: corr(cm[k]) if k in K187 else int(mem[k]["correct"]) for k in keys},
                "ReadPast (gate a + instr)": {k: corr((cm if ga[k] else cb)[k]) if k in K187
                                              else int((mem if ga[k] else base)[k]["correct"]) for k in keys},
            }
            ncap = n_captions(cache)
            for name, c in rows.items():
                sc = scores(c, tasks)
                read = 100 * sum(ga[k] for k in keys) / len(keys) if name.startswith("ReadPast") else 100.0
                print(f"  {s:>3} {name:26s} " + " ".join(f"{sc[c_]:6.2f}" for c_ in cols)
                      + f"  {read:5.1f}  {statistics.median(tl.values()):8.0f}  {ncap:6d}")
            if s == 16 and bb in OFFLINE:
                v = scores(rows["ReadPast (gate a + instr)"], tasks)["avg"]
                print(f"      check s=16 vs round 17: {v:.2f} vs {OFFLINE[bb]:.2f} -> "
                      f"{'PASS' if abs(v - OFFLINE[bb]) <= 0.005 else 'FAIL'}")


if __name__ == "__main__":
    main()
