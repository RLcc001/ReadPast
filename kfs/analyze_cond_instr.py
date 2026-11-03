# usage (host): python3 kfs/analyze_cond_instr.py
"""Standard OVO with the D-HSM abstention instruction applied only to questions whose options already contain an
abstention option (HLD + 1 EPM); everything else unchanged. Offline composition of existing runs."""
import sys, json
sys.path.insert(0, "kfs")
from analyze_round16 import CFG, load, argmax, compose
from analyze_round9 import BACKBONES, gates
from analyze_pilot import scores, load as pload
from analyze_route import RUNS

def corr(rec):  # argmax scoring (letters beyond D)
    return int(chr(65 + argmax(rec["p_opts"] if isinstance(rec["p_opts"], list) else json.loads(rec["p_opts"]))) == rec["ground_truth"])

for bb, sfx in CFG.items():
    cfg = BACKBONES[bb]
    base0, mem0 = (pload(n, RUNS) for n in cfg["ovo"])
    ib, im = load("r4_inj_instr" + sfx), load("cap16_inj_instr" + sfx)
    A = set(load("r4_inj" + sfx))
    keys = sorted(base0)
    special = [k for k in keys if k not in A]  # HLD + questions that already had an abstention option
    tasks = {k: base0[k]["task"] for k in keys}
    tl = {k: mem0[k].get("timeline_tokens") or 0 for k in keys}
    ga = gates(cfg, "ovo", tl)[0]["(a) ours"]
    orig = {k: int(v["correct"]) for k, v in compose(ga, mem0, base0, keys).items()}
    instr = compose(ga, im, ib, keys)
    cond = {k: (corr(instr[k]) if k in special else orig[k]) for k in keys}
    ss = {k: int(base0[k]["correct"]) for k in keys}
    ss_cond = {k: (corr(ib[k]) if k in special else ss[k]) for k in keys}
    out = []
    for name, c in [("SimpleStream", ss), ("SimpleStream + cond. instr", ss_cond), ("gate (a)", orig), ("gate (a) + cond. instr", cond)]:
        s = scores(c, tasks)
        out.append(f"{name}: avg {s['avg']:.2f} RT {s['rt']:.2f} BT {s['bt']:.2f} EPM {s['EPM']:.2f} ASI {s['ASI']:.2f} HLD {s['HLD']:.2f}")
    print(f"=== {bb}  special={len(special)} ({sorted(set(tasks[k] for k in special))})", *out, sep="\n  ")
