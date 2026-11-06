"""Fast offline check of "look back at raw frames when the timeline is not enough" (zero GPU).
Criteria written BEFORE running (11-05):
  S = questions gate (a) sends to the timeline AND whose options have no abstention choice (not in the 187).
  L1 = timeline answer (cap16 run), V = history-frame answer (clip4_ss run: recent 4 + 4 CLIP frames, no timeline),
  L0 = recent-only answer. Everything outside S is exactly ReadPast.
  Primary rule R_conf: on S take V if max p_opts(V) > max p_opts(L1), else L1 (no tuning).
  Report-only: R_low (take V if max p_opts(L1) < 0.5), oracle(L1,V), oracle(L1,L0) (noise reference).
  Decision (both Qwen2.5 and Qwen3-8B):
    headroom = oracle(L1,V) - oracle(L1,L0) on OVO avg. < +1.0 on both -> drop the direction.
    headroom >= +1.0 and R_conf >= ReadPast + 0.3 -> GPU pilot (timeline + frames, separate call).
    headroom >= +1.0 but R_conf < +0.3 -> info exists but confidence cannot find it; pilot must test a sufficiency trigger.
"""
import json, sys
sys.path.insert(0, "kfs")
from analyze_pilot import load as pload, scores
from analyze_round9 import BACKBONES, gates
from analyze_round17 import CFG, corr, raw
from analyze_route import RUNS
K187 = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
BT = {"EPM", "ASI", "HLD"}
SETS = {"Qwen2.5-VL-7B (dev)": ("clip4_ss", ["clip4_lr_ss", "clip16_lr_ss"]),
        "Qwen3-VL-8B (dev)": ("clip4_ss_q3", ["clip4_lr_ss_q3", "clip16_lr_ss_q3"])}
for bb, (vname, extra) in SETS.items():
    cfg = BACKBONES[bb]; sfx = CFG[bb][0]
    base, mem = (pload(n, RUNS) for n in cfg["ovo"])
    keys = sorted(base); tasks = {k: base[k]["task"] for k in keys}
    tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
    g = gates(cfg, "ovo", tl)[0]["(a) ours"]
    cb, cm = raw("r4_cinstr" + sfx), raw("cap16_cinstr" + sfx)
    rp = {k: corr((cm if g[k] else cb)[k]) if k in K187 else int((mem if g[k] else base)[k]["correct"]) for k in keys}
    S = [k for k in keys if g[k] and k not in K187]
    pm = lambda r, k: max(r[k]["p_opts"]) if r[k].get("p_opts") else 0.0
    s0 = scores(rp, tasks)["avg"]
    print(f"\n=== {bb}: ReadPast {s0:.2f}; |S| = {len(S)} (BT {sum(tasks[k] in BT for k in S)}, RT {sum(tasks[k] not in BT for k in S)})")
    for vn in [vname] + extra:
        V = pload(vn, RUNS)
        def rule(pick):
            c = dict(rp); sw = fx = br = 0
            for k in S:
                if pick(k):
                    sw += 1; new = int(V[k]["correct"]); fx += new > c[k]; br += new < c[k]; c[k] = new
            return scores(c, tasks)["avg"], sw, fx, br
        def orac(other):
            c = dict(rp)
            for k in S: c[k] = max(c[k], int(other[k]["correct"]))
            return scores(c, tasks)["avg"]
        rc = rule(lambda k: pm(V, k) > pm(mem, k)); rl = rule(lambda k: pm(mem, k) < 0.5)
        ov, o0 = orac(V), orac(base)
        tag = "PRIMARY" if vn == vname else "report"
        print(f"  V={vn:16s} [{tag}] R_conf {rc[0]:.2f} ({rc[0]-s0:+.2f}; switched {rc[1]}, fixed {rc[2]}, broke {rc[3]})"
              f" | R_low {rl[0]:.2f} ({rl[0]-s0:+.2f}; sw {rl[1]}, +{rl[2]}/-{rl[3]})"
              f" | oracle(L1,V) {ov:.2f} ({ov-s0:+.2f}) oracle(L1,L0) {o0:.2f} ({o0-s0:+.2f}) headroom {ov-o0:+.2f}")
