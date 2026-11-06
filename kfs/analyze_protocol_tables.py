"""Paper Tables I-II (11-05): all rows under the same answer prompt (D-HSM instruction only on the 187 abstention-option Qs)."""
import glob, json, sys
sys.path.insert(0, "kfs")
from analyze_pilot import load, scores
from analyze_round9 import BACKBONES, gates
RUNS = "/data/REN/runs/streamkv"
SFX = {"Qwen2.5-VL-7B (dev)": "", "Qwen3-VL-8B (dev)": "_q3", "Qwen3-VL-4B (B3, held-out)": "_q4", "LLaVA-OV-7B (B4, report only)": "_lov"}
special = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
def raw(name):
    p = sorted(glob.glob(f"{RUNS}/*-{name}/out/results_incremental.jsonl"))[-1]
    return {(r := json.loads(l))["_key"]: r for l in open(p)}
def ok(r):
    p = r["p_opts"] if isinstance(r["p_opts"], list) else json.loads(r["p_opts"])
    return int(chr(65 + max(range(len(p)), key=p.__getitem__)) == r["ground_truth"])
for bb, cfg in BACKBONES.items():
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    cb, cm = raw("r4_cinstr" + SFX[bb]), raw("cap16_cinstr" + SFX[bb])
    keys = sorted(set(base) & set(mem)); tasks = {k: base[k]["task"] for k in keys}
    tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
    gs, info = gates(cfg, "ovo", tl)
    rdir = sorted(glob.glob(f"{RUNS}/*-{cfg['route_ovo']}/out"))[-1]
    route = json.load(open(f"{rdir}/route.json"))
    rules = {"SimpleStream": {k: False for k in keys}, "always read": {k: True for k in keys},
             "fixed 0.5": {k: route[k]["p_past"] >= 0.5 for k in keys},
             "SS 10-shot + cf": gs["(b) SS 10-shot + cf"], "ReadPast (a)": gs["(a) ours"]}
    print(f"\n== {bb}")
    for rn, g in rules.items():
        c = {k: (ok(cm[k]) if g[k] else ok(cb[k])) if k in special else (mem if g[k] else base)[k]["correct"] for k in keys}
        s = scores(c, tasks)
        per = {t: 100 * sum(c[k] for k in keys if tasks[k] == t) / sum(tasks[k] == t for k in keys) for t in ("EPM", "ASI", "HLD")}
        rd = 100 * sum(g.values()) / len(g)
        print(f"  {rn:16s} EPM {per['EPM']:5.1f} ASI {per['ASI']:5.1f} HLD {per['HLD']:5.1f} RT {s['rt']:5.2f} BT {s['bt']:5.2f} avg {s['avg']:6.2f}  read {rd:4.1f}%")
