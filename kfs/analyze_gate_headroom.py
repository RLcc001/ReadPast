"""Gate headroom under the shared answer prompt: OVO + SB, never/always/gate(a)/label route/oracle, plus L0-confidence combos."""
import glob, json, sys
sys.path.insert(0, "kfs")
from analyze_pilot import load, scores, BACKWARD
from analyze_round9 import BACKBONES, gates
RUNS = "/data/REN/runs/streamkv"
SFX = {"Qwen2.5-VL-7B (dev)": "", "Qwen3-VL-8B (dev)": "_q3", "Qwen3-VL-4B (B3, held-out)": "_q4", "LLaVA-OV-7B (B4, report only)": "_lov"}
special = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
def raw(name):
    p = sorted(glob.glob(f"{RUNS}/*-{name}/out/results_incremental.jsonl"))[-1]
    return {(r := json.loads(l))["_key"]: r for l in open(p)}
def popt(r): return r["p_opts"] if isinstance(r["p_opts"], list) else json.loads(r["p_opts"])
def ok(r):
    p = popt(r); return int(chr(65 + max(range(len(p)), key=p.__getitem__)) == r["ground_truth"])
for bb, cfg in BACKBONES.items():
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    cb, cm = raw("r4_cinstr" + SFX[bb]), raw("cap16_cinstr" + SFX[bb])
    keys = sorted(set(base) & set(mem)); tasks = {k: base[k]["task"] for k in keys}
    tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
    gs, _ = gates(cfg, "ovo", tl); a = gs["(a) ours"]
    B = {k: ok(cb[k]) if k in special else base[k]["correct"] for k in keys}
    M = {k: ok(cm[k]) if k in special else mem[k]["correct"] for k in keys}
    pmax = {k: max(popt(base[k])) for k in keys}
    rules = {"never": {k: 0 for k in keys}, "always": {k: 1 for k in keys}, "gate (a)": a,
             "label route": {k: tasks[k] in BACKWARD for k in keys},
             "(a) or L0 pmax<0.5": {k: a[k] or pmax[k] < 0.5 for k in keys},
             "(a) or L0 pmax<0.8": {k: a[k] or pmax[k] < 0.8 for k in keys}}
    s0 = scores(B, tasks)["avg"]
    print(f"\n== {bb}  OVO (shared prompt)")
    for rn, g in rules.items():
        c = {k: M[k] if g[k] else B[k] for k in keys}; s = scores(c, tasks)
        print(f"  {rn:20s} avg {s['avg']:6.2f} (vs never {s['avg']-s0:+5.2f})  RT {s['rt']:5.2f}  read {100*sum(map(bool,g.values()))/len(g):5.1f}%")
    orc = {k: max(B[k], M[k]) for k in keys}; s = scores(orc, tasks)
    fx = sum(M[k] > B[k] for k in keys); br = sum(M[k] < B[k] for k in keys)
    fxa = sum(M[k] > B[k] for k in keys if a[k]); bra = sum(M[k] < B[k] for k in keys if a[k])
    print(f"  {'oracle (per Q)':20s} avg {s['avg']:6.2f} (vs never {s['avg']-s0:+5.2f})  | read helps {fx} / hurts {br} Qs; gate (a) captures +{fxa} / -{bra}")
    # SB
    rows = {}
    for p in sorted(glob.glob(f"{RUNS}/{cfg['sb']}/out/results_incremental.jsonl")):
        for l in open(p): r = json.loads(l); rows[r["_key"]] = r
    rows = list(rows.values()); qid = {r["_key"]: r["_key"].split("_", 1)[1] for r in rows}
    tls = {qid[r["_key"]]: r["timeline_tokens"] for r in rows}
    gsb, _ = gates(cfg, "sb", tls); asb = gsb["(a) ours"]
    n = len(rows); has = lambda r: r["timeline_tokens"] > 0
    acc = lambda f: 100 * sum(f(r) for r in rows) / n
    nev = acc(lambda r: r["correct_l0"]); alw = acc(lambda r: r["correct_l1"])
    ga = acc(lambda r: r["correct_l1"] if has(r) and asb[qid[r["_key"]]] else r["correct_l0"])
    orc = acc(lambda r: max(r["correct_l0"], r["correct_l1"]))
    rd = 100 * sum(has(r) and asb[qid[r["_key"]]] for r in rows) / n
    print(f"  SB-RT: never {nev:.2f}  always {alw:.2f} ({alw-nev:+.2f})  gate (a) {ga:.2f} ({ga-nev:+.2f}, read {rd:.0f}%)  oracle {orc:.2f} ({orc-nev:+.2f})")
