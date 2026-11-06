"""Ablation effect sizes + video-bootstrap 95% CIs under the shared answer prompt (OVO standard avg; SB-RT acc)."""
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
def ci(by, f, reps=1000):
    vids = sorted(by); rng = random.Random(0); ds = []
    for _ in range(reps):
        ks = [k for v in (rng.choice(vids) for _ in vids) for k in by[v]]
        ds.append(f(ks))
    ds.sort(); return ds[int(.025 * reps)], ds[int(.975 * reps)]
def fmt(d, lo, hi):
    sig = "*" if lo > 0 or hi < 0 else " "
    return f"{d:+5.2f} [{lo:+5.2f},{hi:+5.2f}]{sig}"
for bb, cfg in BACKBONES.items():
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    cb, cm = raw("r4_cinstr" + SFX[bb]), raw("cap16_cinstr" + SFX[bb])
    keys = sorted(set(base) & set(mem)); tasks = {k: base[k]["task"] for k in keys}
    by = defaultdict(list)
    for k in keys: by[base[k]["video"]].append(k)
    tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
    gs, _ = gates(cfg, "ovo", tl); a = gs["(a) ours"]
    rdir = sorted(glob.glob(f"{RUNS}/*-{cfg['route_ovo']}/out"))[-1]
    route = json.load(open(f"{rdir}/route.json"))
    B = {k: ok(cb[k]) if k in special else base[k]["correct"] for k in keys}
    M = {k: ok(cm[k]) if k in special else mem[k]["correct"] for k in keys}
    sel = lambda g: {k: M[k] if g[k] else B[k] for k in keys}
    C = {"SS": B, "always": M, "gate": sel(a), "fixed0.5": sel({k: route[k]["p_past"] >= .5 for k in keys})}
    def d(x, y, m="avg"):
        f = lambda ks: scores({k: C[x][k] for k in ks}, {k: tasks[k] for k in ks})[m] - scores({k: C[y][k] for k in ks}, {k: tasks[k] for k in ks})[m]
        return fmt(f(keys), *ci(by, f))
    print(f"\n== {bb}")
    print(f"  timeline (always - SS)     avg {d('always','SS')}   RT {d('always','SS','rt')}")
    print(f"  ReadPast (gate - SS)       avg {d('gate','SS')}   RT {d('gate','SS','rt')}")
    print(f"  gate vs always             avg {d('gate','always')}   RT {d('gate','always','rt')}")
    print(f"  calibrated vs fixed 0.5    avg {d('gate','fixed0.5')}")
    pt = lambda c, t: 100 * sum(c[k] for k in keys if tasks[k] == t) / sum(tasks[k] == t for k in keys)
    print("  per-task ReadPast - SS: " + "  ".join(f"{t} {pt(C['gate'],t)-pt(B,t):+5.1f}" for t in ("EPM","ASI","HLD","OCR","ACR","ATR","STU","FPD","OJR")))
    # SB
    rows = {}
    for p in sorted(glob.glob(f"{RUNS}/{cfg['sb']}/out/results_incremental.jsonl")):
        for l in open(p): r = json.loads(l); rows[r["_key"]] = r
    rows = list(rows.values()); qid = {r["_key"]: r["_key"].split("_", 1)[1] for r in rows}
    gsb, _ = gates(cfg, "sb", {qid[r["_key"]]: r["timeline_tokens"] for r in rows}); asb = gsb["(a) ours"]
    R = {r["_key"]: r for r in rows}
    G = {k: (r["correct_l1"] if r["timeline_tokens"] > 0 and asb[qid[k]] else r["correct_l0"]) for k, r in R.items()}
    sby = defaultdict(list)
    for k, r in R.items(): sby[r.get("video") or k.split("_")[0]].append(k)
    acc = lambda ks, f: 100 * sum(f(k) for k in ks) / len(ks)
    f1 = lambda ks: acc(ks, lambda k: G[k]) - acc(ks, lambda k: R[k]["correct_l0"])
    f2 = lambda ks: acc(ks, lambda k: G[k]) - acc(ks, lambda k: R[k]["correct_l1"])
    print(f"  SB: ReadPast - SS {fmt(f1(list(R)), *ci(sby, f1))}   gate - always {fmt(f2(list(R)), *ci(sby, f2))}   (SB groups: {len(sby)})")
