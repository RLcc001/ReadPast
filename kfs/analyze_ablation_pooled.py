"""Pooled over 4 backbones (mean of per-backbone deltas), joint video bootstrap; shared answer prompt."""
import glob, json, random, sys
from collections import defaultdict
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
OVO, SB = [], []
for bb, cfg in BACKBONES.items():
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    cb, cm = raw("r4_cinstr" + SFX[bb]), raw("cap16_cinstr" + SFX[bb])
    keys = sorted(set(base) & set(mem)); tasks = {k: base[k]["task"] for k in keys}
    a = gates(cfg, "ovo", {k: mem[k].get("timeline_tokens") or 0 for k in keys})[0]["(a) ours"]
    route = json.load(open(sorted(glob.glob(f"{RUNS}/*-{cfg['route_ovo']}/out"))[-1] + "/route.json"))
    B = {k: ok(cb[k]) if k in special else base[k]["correct"] for k in keys}
    M = {k: ok(cm[k]) if k in special else mem[k]["correct"] for k in keys}
    sel = lambda g: {k: M[k] if g[k] else B[k] for k in keys}
    OVO.append(dict(tasks=tasks, video={k: base[k]["video"] for k in keys}, SS=B, always=M, gate=sel(a),
                    fixed=sel({k: route[k]["p_past"] >= .5 for k in keys})))
    rows = {}
    for p in sorted(glob.glob(f"{RUNS}/{cfg['sb']}/out/results_incremental.jsonl")):
        for l in open(p): r = json.loads(l); rows[r["_key"]] = r
    qid = {k: k.split("_", 1)[1] for k in rows}
    asb = gates(cfg, "sb", {qid[k]: r["timeline_tokens"] for k, r in rows.items()})[0]["(a) ours"]
    SB.append(dict(video={k: r.get("video") or k.split("_")[0] for k, r in rows.items()}, SS={k: r["correct_l0"] for k, r in rows.items()},
                   always={k: r["correct_l1"] for k, r in rows.items()},
                   gate={k: r["correct_l1"] if r["timeline_tokens"] > 0 and asb[qid[k]] else r["correct_l0"] for k, r in rows.items()}))
def pooled_ovo(x, y, m, reps=1000):
    vids = sorted(set(OVO[0]["video"].values()))
    by = defaultdict(list)
    for k, v in OVO[0]["video"].items(): by[v].append(k)
    def f(ks):
        out = []
        for o in OVO:
            ks2 = [k for k in ks if k in o["tasks"]]; t = {k: o["tasks"][k] for k in ks2}
            out.append(scores({k: o[x][k] for k in ks2}, t)[m] - scores({k: o[y][k] for k in ks2}, t)[m])
        return sum(out) / len(out)
    rng = random.Random(0); ds = sorted(f([k for v in (rng.choice(vids) for _ in vids) for k in by[v]]) for _ in range(reps))
    return f([k for v in vids for k in by[v]]), ds[25], ds[974]
def pooled_sb(x, y, reps=1000):
    by = defaultdict(list)
    for k, v in SB[0]["video"].items(): by[v].append(k)
    vids = sorted(by)
    f = lambda ks: sum(100 * (sum(s[x][k] for k in ks if k in s[x]) - sum(s[y][k] for k in ks if k in s[y])) / len(ks) for s in SB) / len(SB)
    rng = random.Random(0); ds = sorted(f([k for v in (rng.choice(vids) for _ in vids) for k in by[v]]) for _ in range(reps))
    return f([k for v in vids for k in by[v]]), ds[25], ds[974]
fmt = lambda d, lo, hi: f"{d:+5.2f} [{lo:+5.2f},{hi:+5.2f}]" + ("*" if lo > 0 or hi < 0 else "")
print("pooled over 4 backbones (mean delta)")
print("  OVO ReadPast - SS avg   ", fmt(*pooled_ovo("gate", "SS", "avg")))
print("  OVO timeline (always-SS)", fmt(*pooled_ovo("always", "SS", "avg")), " RT", fmt(*pooled_ovo("always", "SS", "rt")))
print("  OVO gate - always   avg ", fmt(*pooled_ovo("gate", "always", "avg")), " RT", fmt(*pooled_ovo("gate", "always", "rt")))
print("  OVO calibrated - fixed  ", fmt(*pooled_ovo("gate", "fixed", "avg")))
print("  SB  ReadPast - SS       ", fmt(*pooled_sb("gate", "SS")))
print("  SB  gate - always       ", fmt(*pooled_sb("gate", "always")))
