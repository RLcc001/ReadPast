"""Round 18 (pre-registered, EXPERIMENTS.md): public D-HSM code (third_party/D-HSM @04579a0, paper setting:
dhsm memory + uniform_abstention, 4 recent frames, 1 fps) rerun on our local OVO (1468 MCQ) and SB-RT (2500), scored
with our scorer, next to our SimpleStream and ReadPast (gate (a) + conditional instruction). Host-side, stdlib only.

Qwen2.5 OVO backward = round-15 cell (4) (`dhsm_mem_abstain_s*`, verified identical on the round-18 smoke);
realtime = `dhsm_mem_abstain_rt_s*`. Qwen3-8B OVO = `dhsm_mem_abstain_q3_s*`. SB = `dhsm_sb_{q25,q3}_s*`.
Report-only: D-HSM's own recent-only base (`dhsm_recent_abstain_*`, same instruction, no memory).
Only runs whose rx status is "done" are used; a later run of the same name overrides an earlier one.

usage: python3 kfs/analyze_round18_dhsm.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import mcq, scores  # noqa: E402
from analyze_round9 import BACKBONES  # noqa: E402
from analyze_round9 import sb as our_sb  # noqa: E402
from analyze_route import RUNS  # noqa: E402

TASKS = ["EPM", "ASI", "HLD", "OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
N_OVO, N_SB = 1468, 2500
OURS = {"Qwen2.5-VL-7B (dev)": 70.78, "Qwen3-VL-8B (dev)": 73.34}  # ReadPast, round 17 (standard OVO)
OURS_SS_INSTR = {"Qwen2.5-VL-7B (dev)": 68.88, "Qwen3-VL-8B (dev)": 71.87}  # SimpleStream + conditional instruction
PAPER = {  # D-HSM Table 2 (4 frames; (RT+BT)/2 recomputed) and SB real-time
    "Qwen2.5-VL-7B (dev)": dict(RT=79.0, EPM=52.9, ASI=63.5, HLD=72.0, avg=70.9, SB=82.5),
    "Qwen3-VL-8B (dev)": dict(RT=81.6, EPM=54.6, ASI=62.2, HLD=67.2, avg=71.45, SB=83.1),
}
CFG = {  # backbone -> (memory run name patterns, recent-only patterns, SB pattern)
    "Qwen2.5-VL-7B (dev)": (["dhsm_mem_abstain_s[0-9]", "dhsm_mem_abstain_rt_s[0-9]"],
                            ["dhsm_recent_abstain_s[0-9]", "dhsm_recent_abstain_rt_s[0-9]"], "dhsm_sb_q25_s[0-9]"),
    "Qwen3-VL-8B (dev)": (["dhsm_mem_abstain_q3_s[0-9]"], ["dhsm_recent_abstain_q3_s[0-9]"], "dhsm_sb_q3_s[0-9]"),
}


def done_runs(pattern):
    """Latest finished run per run name matching pattern."""
    by = {}
    for d in sorted(glob.glob(f"{RUNS}/*-{pattern}")):
        try:
            m = json.load(open(f"{d}/meta.json"))
        except (OSError, ValueError):
            continue
        if m.get("status") == "done":
            by[m["name"]] = d
    return list(by.values())


def load_ovo(patterns):
    rows, n_runs = {}, 0
    for pat in patterns:
        for d in done_runs(pat):
            paths = sorted(glob.glob(f"{d}/out/hub_and_spoke_results_*.json"))
            if not paths:
                continue
            n_runs += 1
            res = json.load(open(paths[-1]))
            for split in ("backward", "realtime"):
                for r in res.get(split, []):
                    rows[f"{r['task']}:{r['id']}"] = r
    return rows, n_runs


def load_sb(pattern):
    rows, n_runs = {}, 0
    for d in done_runs(pattern):
        paths = sorted(glob.glob(f"{d}/out/hub_and_spoke_streamingbench_*.json"))
        if not paths:
            continue
        n_runs += 1
        for r in json.load(open(paths[-1]))["results"]:
            rows[(r["video_path"], r["time_stamp"], r["task_type"], r["question"])] = r
    return rows, n_runs


def ovo_report(name, rows, ref):
    keys = [k for k in rows if rows[k]["task"] in TASKS]
    c = {k: int((mcq(rows[k].get("response")) or "") == rows[k]["ground_truth"]) for k in keys}
    s = scores(c, {k: rows[k]["task"] for k in keys})
    miss = sum(rows[k].get("response") is None for k in keys)
    bt = [k for k in keys if rows[k]["task"] in ("EPM", "ASI", "HLD")]
    rt = [k for k in keys if k not in set(bt)]
    rd = lambda ks: 100 * sum(bool(rows[k].get("route_use_memory")) for k in ks) / max(1, len(ks))  # noqa: E731
    d = f"{s['avg'] - ref:+6.2f}" if ref is not None else "      "
    print(f"  {name:34s} n={len(keys):4d} miss={miss:2d}  avg {s['avg']:6.2f} {d}  RT {s['rt']:6.2f}  "
          + " ".join(f"{s.get(t, float('nan')):5.1f}" for t in TASKS) + f"   memory route BT {rd(bt):3.0f}% RT {rd(rt):3.0f}%")
    return s, len(keys)


def main():
    print("columns: " + " ".join(TASKS))
    for bb, (mem_pats, rec_pats, sb_pat) in CFG.items():
        cfg = BACKBONES[bb]
        print(f"\n=== {bb}")
        ss = None
        sbo = our_sb(cfg)[0]
        print(f"  ours (same data):  SimpleStream SB {sbo['never (r4)']['acc']:.2f}, ReadPast SB {sbo['(a) ours']['acc']:.2f};"
              f"  ReadPast OVO {OURS[bb]:.2f}, SimpleStream + cond. instr OVO {OURS_SS_INSTR[bb]:.2f}")
        rows, n = load_ovo(mem_pats)
        if rows:
            s, nk = ovo_report(f"D-HSM rerun ({n} runs)", rows, OURS[bb])
            ss = s
            if nk < N_OVO:
                print(f"    incomplete: {nk}/{N_OVO} questions")
        else:
            print("  D-HSM OVO not ready")
        rrows, n = load_ovo(rec_pats)
        if rrows:
            rs, _ = ovo_report(f"D-HSM own recent-only base ({n})", rrows, None)
            if ss is not None:
                print(f"    D-HSM over its own base: {ss['avg'] - rs['avg']:+.2f}")
        p = PAPER[bb]
        print(f"  {'D-HSM paper':34s} avg {p['avg']:6.2f}  RT {p['RT']:6.2f}  EPM {p['EPM']} ASI {p['ASI']} HLD {p['HLD']}")
        if ss is not None:
            print(f"    rerun - paper: avg {ss['avg'] - p['avg']:+.2f}, RT {ss['rt'] - p['RT']:+.2f}, "
                  + ", ".join(f"{t} {ss[t] - p[t]:+.1f}" for t in ("EPM", "ASI", "HLD")))
        sbr, n = load_sb(sb_pat)
        if sbr:
            ok = [int((mcq(r.get("response")) or "") == r["answer_gt"]) for r in sbr.values()]
            agree = sum(int(bool(r.get("correct"))) == o for r, o in zip(sbr.values(), ok))
            err = sum(bool(r.get("error")) for r in sbr.values())
            acc = 100 * sum(ok) / len(ok)
            print(f"  D-HSM SB-RT rerun ({n} runs): {acc:.2f} on {len(ok)}/{N_SB} (errors {err}; agrees with its own "
                  f"'correct' on {agree}/{len(ok)}); paper {p['SB']:.1f} ({acc - p['SB']:+.2f}); "
                  f"vs ReadPast {acc - sbo['(a) ours']['acc']:+.2f}")
        else:
            print("  D-HSM SB not ready")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    main()
