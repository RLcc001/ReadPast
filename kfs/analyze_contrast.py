"""Offline: content-contrastive memory reading (round 7). score = log p1 - b*log pf (cc) or log p0 + b*(log p1 - log pf) (ev).
p0 = r4_ss, p1 = cap16, pf = cap16_foreign (same #entries from another video). Host stdlib: python3 kfs/analyze_contrast.py"""
import glob, json, math, sys
sys.path.insert(0, "kfs")
from analyze_pilot import BACKWARD, REALTIME, load, scores
from analyze_route import RUNS, boot, no_hld, row

r4, cap, fr = load("r4_ss", RUNS), load("cap16", RUNS), load("cap16_foreign", RUNS)
keys = sorted(r4)
assert all(k in cap and k in fr for k in keys), (len(cap), len(fr))
cal = sorted(glob.glob(f"{RUNS}/*-route_cal_p0/out"))[-1]
route = json.load(open(f"{cal}/route.json")); p_cf = json.load(open(f"{cal}/calib.json"))["p_cf"]
tasks = {k: r4[k]["task"] for k in keys}; video = {k: r4[k]["video"] for k in keys}
L = "ABCDEFGHIJ"
def am(s): return L[max(range(len(s)), key=lambda i: s[i])]
def lg(p): return [math.log(max(x, 1e-6)) for x in p]
agree = sum(am(cap[k]["p_opts"]) == (cap[k]["response"] or "").strip()[:1] for k in keys)
print(f"argmax(p_opts) == response letter: {agree}/{len(keys)}")
def corr(pred): return {k: int(pred[k] == r4[k]["ground_truth"]) for k in keys}
has_tl = {k: (cap[k].get("n_captions") or 0) > 0 for k in keys}
def rule_all(k): return has_tl[k]
def rule_cal(k): return has_tl[k] and route[k]["p_past"] >= p_cf
base = {"r4": {k: am(r4[k]["p_opts"]) for k in keys}}
sb = scores(corr(base["r4"]), tasks)
print(row("r4 (argmax)", sb))
for rname, rule in (("always L1", rule_all), ("cal router", rule_cal)):
    ref = None
    for name, f in [("cap16", lambda k, b: lg(cap[k]["p_opts"]))] + \
        [(f"cc b={b}", (lambda b: lambda k, _: [x - b * y for x, y in zip(lg(cap[k]["p_opts"]), lg(fr[k]["p_opts"]))])(b)) for b in (0.25, 0.5, 1.0)] + \
        [(f"ev b={b}", (lambda b: lambda k, _: [z + b * (x - y) for x, y, z in zip(lg(cap[k]["p_opts"]), lg(fr[k]["p_opts"]), lg(r4[k]["p_opts"]))])(b)) for b in (0.5, 1.0, 2.0)]:
        pred = {k: (am(f(k, None)) if rule(k) else base["r4"][k]) for k in keys}
        c = corr(pred); s = scores(c, tasks)
        line = row(f"{rname} | {name}", s, sb)
        if ref is None: ref = c
        else:
            for lab, fn in (("avg", lambda s: s["avg"]), ("noHLD", no_hld)):
                lo, hi, p = boot(keys, tasks, video, ref, c, fn)
                line += f"  {lab} vs cap16 [{lo:+.2f},{hi:+.2f}]"
        print(line + f"  noHLD {no_hld(s):.2f}")
