"""R1 sanity checks (offline, 11-02, not pre-registered): (1) always pick the abstain option when offered, on OVO;
(2) estimated net effect of R1 under different shares of unanswerable questions (assumes adding an abstain option leaves
the other options' relative probabilities unchanged; answerable loss = correct answers on which the R1 test fires).

usage: python3 kfs/analyze_r1_mix.py
"""
import glob, json, math, sys
sys.path.insert(0, "kfs")
from analyze_pilot import load, mcq, scores
from analyze_route import RUNS
from analyze_round9 import BACKBONES, gates
from analyze_round12 import ABSTAIN, ANNO, CFG, EPS, r1

# (1) always abstain
anno = {f"{a['task']}:{a['id']}": a for a in json.load(open(ANNO))}
for bb in ["Qwen2.5-VL-7B (dev)", "Qwen3-VL-8B (dev)"]:
    cfg = BACKBONES[bb]
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    keys = sorted(set(base) & set(mem)); tasks = {k: base[k]["task"] for k in keys}; gt = {k: base[k]["ground_truth"] for k in keys}
    tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
    a = gates(cfg, "ovo", tl)[0]["(a) ours"]
    for sn, recs in [("SimpleStream", base), ("gate (a)", {k: (mem if a[k] else base)[k] for k in keys})]:
        def ab(k):
            u = [i for i, o in enumerate(anno[k]["options"]) if ABSTAIN.match(o)]
            return chr(65 + u[0]) if u else mcq(recs[k]["response"])
        s = scores({k: int(ab(k) == gt[k]) for k in keys}, tasks)
        print(f"{bb[:12]:12s} {sn:13s} always-abstain: avg {s['avg']:.2f} HLD {s['HLD']:.1f} EPM {s['EPM']:.2f} RT {s['rt']:.2f}")

# (2) R1 net effect by question mix
anno = {f"{a['task']}:{a['id']}": a for a in json.load(open(ANNO))}
for name, (bb, pn) in CFG.items():
    cfg = BACKBONES[bb]; prior = json.load(open(sorted(glob.glob(f"{RUNS}/*-{pn}/out/prior.json"))[-1]))
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    keys = sorted(set(base) & set(mem)); tasks = {k: base[k]["task"] for k in keys}; gt = {k: base[k]["ground_truth"] for k in keys}
    a = gates(cfg, "ovo", {k: mem[k].get("timeline_tokens") or 0 for k in keys})[0]["(a) ours"]
    recs = {k: (mem if a[k] else base)[k] for k in keys}
    acc0, acc1 = {}, {}  # per task: accuracy without / with R1 (answerable: estimated loss = correct answers that would fire)
    for t in set(tasks.values()):
        ks = [k for k in keys if tasks[k] == t]
        if t == "HLD":
            acc0[t] = 100 * sum(mcq(recs[k]["response"]) == gt[k] for k in ks) / len(ks)
            acc1[t] = 100 * sum(r1(recs[k], prior[k]["lp_q"], anno[k]["options"]) == gt[k] for k in ks) / len(ks)
            continue
        c = f = 0
        for k in ks:
            ans = mcq(recs[k]["response"]); ok = ans == gt[k]; c += ok
            if ok and ord(ans) - 65 < len(recs[k]["p_opts"]):
                o = ord(ans) - 65; f += math.log(max(recs[k]["p_opts"][o], EPS)) <= prior[k]["lp_q"][o]
        acc0[t], acc1[t] = 100 * c / len(ks), 100 * (c - f) / len(ks)
    d = {t: acc1[t] - acc0[t] for t in acc0}
    mem_d = (d["EPM"] + d["ASI"]) / 2
    rt_d = sum(d[t] for t in ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]) / 6
    print(f"{name:12s} dHLD {d['HLD']:+.1f}  dEPM {d['EPM']:+.1f} dASI {d['ASI']:+.1f}  dRT {rt_d:+.1f} | "
          f"1:1 HLD vs memory {(d['HLD'] + mem_d) / 2:+.2f} | OVO weights, abstain option on every Q {(rt_d + (d['EPM'] + d['ASI'] + d['HLD']) / 3) / 2:+.2f}"
          f" | memory only (BT) {(d['EPM'] + d['ASI'] + d['HLD']) / 3:+.2f}")
