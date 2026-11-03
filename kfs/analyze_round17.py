"""Round 17: conditional abstention instruction (D-HSM's, only on questions whose options contain an abstention option)
replacing R1, plus the route.py rerun with log-odds. Host-side, stdlib only.

Checks (pre-registered in EXPERIMENTS.md):
  - 187 special questions: p_opts identical to round 16 *_inj_instr (Qwen2.5 / Qwen3-8B / 4B);
  - standard OVO with gate (a) = offline composition (kfs/analyze_cond_instr.py) within 0.1;
  - route reruns: p_past identical to the old route.json, sigmoid(lo) consistent with p_past.

usage: python3 kfs/analyze_round17.py
"""

from __future__ import annotations

import glob
import json
import math
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load as pload  # noqa: E402
from analyze_pilot import scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_route import RUNS  # noqa: E402

CFG = {  # backbone -> (suffix of round-17 runs, suffix of round-16 instruction runs or None, route run)
    "Qwen2.5-VL-7B (dev)": ("", "", "route_cal_p0"),
    "Qwen3-VL-8B (dev)": ("_q3", "_q3", "route_q3_cal_p0"),
    "Qwen3-VL-4B (B3, held-out)": ("_q4", "_q4", "route_q4_cal_p0"),
    "LLaVA-OV-7B (B4, report only)": ("_lov", None, "route_lov_cal_p0"),
}
OFFLINE = {"Qwen2.5-VL-7B (dev)": 70.78, "Qwen3-VL-8B (dev)": 73.34, "Qwen3-VL-4B (B3, held-out)": 72.60}
KEYS = json.load(open("kfs/data/keys_ovo_abstopt187.json"))


def raw(name):
    paths = sorted(glob.glob(f"{RUNS}/*-{name}/out/results_incremental.jsonl"))
    return {(r := json.loads(line))["_key"]: r for line in open(paths[-1])} if paths else None


def popts(r):
    return r["p_opts"] if isinstance(r["p_opts"], list) else json.loads(r["p_opts"])


def corr(r):
    p = popts(r)
    return int(chr(65 + max(range(len(p)), key=p.__getitem__)) == r["ground_truth"])


def route_check(run):
    old = sorted(glob.glob(f"{RUNS}/*-{run}/out"))
    new = sorted(glob.glob(f"{RUNS}/*-{run}_lo/out"))
    if not old or not new:
        return "route rerun not ready"
    a, b = json.load(open(f"{old[-1]}/route.json")), json.load(open(f"{new[-1]}/route.json"))
    same = sum(a[k]["p_past"] == b[k]["p_past"] for k in a)
    sig = lambda x: 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))  # noqa: E731
    dev = max(abs(sig(b[k]["lo"]) - b[k]["p_past"]) for k in b)
    lo = [b[k]["lo"] for k in b]
    ca, cb = json.load(open(f"{old[-1]}/calib.json")), json.load(open(f"{new[-1]}/calib.json"))
    return (f"route: p_past identical {same}/{len(a)}; max |sigmoid(lo) - p_past| {dev:.2e}; lo range "
            f"[{min(lo):.1f}, {max(lo):.1f}]; p_cf {ca['p_cf']:.6g} -> {cb['p_cf']:.6g}")


def main():
    for bb, (sfx, sfx16, route) in CFG.items():
        cfg = BACKBONES[bb]
        print(f"\n=== {bb}")
        print("  " + route_check(route))
        cb, cm = raw("r4_cinstr" + sfx), raw("cap16_cinstr" + sfx)
        if cb is None or cm is None or len(cb) < len(KEYS) or len(cm) < len(KEYS):
            print("  evaluation runs not ready")
            continue
        if sfx16 is not None:
            for n17, n16 in (("r4_cinstr", "r4_inj_instr"), ("cap16_cinstr", "cap16_inj_instr")):
                a, b = raw(n17 + sfx), raw(n16 + sfx16)
                same = sum(popts(a[k]) == popts(b[k]) for k in KEYS)
                print(f"  {n17}{sfx} vs round 16 {n16}{sfx16}: p_opts identical {same}/{len(KEYS)}")
        base0, mem0 = (pload(n, RUNS) for n in cfg["ovo"])
        keys = sorted(base0)
        tasks = {k: base0[k]["task"] for k in keys}
        tl = {k: mem0[k].get("timeline_tokens") or 0 for k in keys}
        ga = gates(cfg, "ovo", tl)[0]["(a) ours"]
        special = set(KEYS)
        ss = {k: int(base0[k]["correct"]) for k in keys}
        ours = {k: int((mem0 if ga[k] else base0)[k]["correct"]) for k in keys}
        rows = {
            "SimpleStream": ss,
            "SimpleStream + cond. instr": {k: corr(cb[k]) if k in special else ss[k] for k in keys},
            "gate (a)": ours,
            "gate (a) + cond. instr": {k: corr((cm if ga[k] else cb)[k]) if k in special else ours[k] for k in keys},
        }
        ref = scores(ss, tasks)["avg"]
        for name, c in rows.items():
            s = scores(c, tasks)
            print(f"  {name:28s} avg {s['avg']:.2f} ({s['avg'] - ref:+.2f})  RT {s['rt']:.2f}  EPM {s['EPM']:.2f}  "
                  f"ASI {s['ASI']:.2f}  HLD {s['HLD']:.2f}")
        if bb in OFFLINE:
            s = scores(rows["gate (a) + cond. instr"], tasks)["avg"]
            print(f"  check: real {s:.2f} vs offline {OFFLINE[bb]:.2f} -> {'PASS' if abs(s - OFFLINE[bb]) <= 0.1 else 'FAIL'}")


if __name__ == "__main__":
    main()
