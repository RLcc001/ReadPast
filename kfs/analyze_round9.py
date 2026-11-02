"""Round 9 (pre-registered, EXPERIMENTS.md): held-out test of the memory-access gate on a third backbone.

Candidates (frozen before the B3 runs):
  (a) ours: route.py prompt P0, read iff p_past >= p_cf (mean over "N/A", "", "[MASK]")
  (b) ShallowStream 10-shot prompt, read iff sigmoid(logit_A - logit_B) >= mean over the same content-free questions
  (c) cost-aware: (a) OR timeline <= 512 tokens (backbone tokenizer)
References: never (r4), always L1, label route (OVO BT -> L1).

Per candidate: pass = OVO avg >= r4 + 1.0 and RT >= r4 - 0.3, SB >= r4, and non-degenerate (OVO RT read < 80%).
G = (OVO noHLD gain + SB gain) / 2 over r4. Decision on B3: (b) or (c) replaces (a) iff it passes and G >= G(a) + 0.3
(larger G if both); otherwise keep (a). Qwen2.5 / Qwen3-8B rows are development data, shown for reference.

usage: python3 kfs/analyze_round9.py
"""

import glob
import json
import math
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import BACKWARD, REALTIME, load, scores  # noqa: E402
from analyze_route import RUNS, boot, no_hld  # noqa: E402
from analyze_sb import acc  # noqa: E402
from analyze_sb import boot as boot_sb  # noqa: E402

BACKBONES = {
    "Qwen2.5-VL-7B (dev)": dict(ovo=("r4_ss", "cap16"), sb="261025-04464*-sb_casc_s[0-3]", route_ovo="route_cal_p0",
                                route_sb="route_cal_sb_p0", routers="routers_cf_q25"),
    "Qwen3-VL-8B (dev)": dict(ovo=("r4_q3", "cap16_q3"), sb="261101-0240*-sb_casc_q3_s[0-3]", route_ovo="route_q3_cal_p0",
                              route_sb="route_q3_cal_sb_p0", routers="routers_cf_q3"),
    "Qwen3-VL-4B (B3, held-out)": dict(ovo=("r4_q4", "cap16_q4"), sb="2611*-sb_casc_q4_s[0-3]", route_ovo="route_q4_cal_p0",
                                       route_sb="route_q4_cal_sb_p0", routers="routers_q4"),
    "LLaVA-OV-7B (B4, report only)": dict(ovo=("r4_lov", "cap16_lov"), sb="2611*-sb_casc_lov_s[0-3]", route_ovo="route_lov_cal_p0",
                                          route_sb="route_lov_cal_sb_p0", routers="routers_lov"),
}
B3 = "Qwen3-VL-4B (B3, held-out)"
TL_MAX = 512


def newest(pattern):
    paths = sorted(glob.glob(f"{RUNS}/{pattern}"))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths[-1]


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def gates(cfg, bench, tl):
    """name -> key -> read memory? (keys of route.json / scores_<bench>.json); tl: key -> timeline tokens."""
    rdir = newest(f"*-{cfg['route_' + bench]}/out")
    route, p_cf = json.load(open(f"{rdir}/route.json")), json.load(open(f"{rdir}/calib.json"))["p_cf"]
    sdir = newest(f"*-{cfg['routers']}/out")
    sc, cf = json.load(open(f"{sdir}/scores_{bench}.json")), json.load(open(f"{sdir}/scores_cf.json"))
    p_ss = sum(sigmoid(v["ss_fs10"]) for v in cf.values()) / len(cf)
    a = {k: v["p_past"] >= p_cf for k, v in route.items()}
    return {
        "(a) ours": a,
        "(b) SS 10-shot + cf": {k: sigmoid(v["ss_fs10"]) >= p_ss for k, v in sc.items()},
        "(c) (a) or TL<=512": {k: a[k] or tl.get(k, 0) <= TL_MAX for k in a},
    }, {"p_cf": p_cf, "ss_fs10 cf gap": round(math.log(p_ss / (1 - p_ss)), 3)}


def ovo(cfg):
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    keys = sorted(set(base) & set(mem))
    tasks = {k: base[k]["task"] for k in keys}
    video = {k: base[k]["video"] for k in keys}
    b = {k: base[k]["correct"] for k in keys}
    m = {k: mem[k]["correct"] for k in keys}
    tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
    gs, info = gates(cfg, "ovo", tl)
    rules = {"never (r4)": {k: False for k in keys}, "always L1": {k: True for k in keys},
             "label route": {k: tasks[k] in BACKWARD for k in keys}, **gs}
    bt = [k for k in keys if tasks[k] in BACKWARD]
    rt = [k for k in keys if tasks[k] in REALTIME]
    s0 = scores(b, tasks)
    ref = {k: (m if gs["(a) ours"][k] else b)[k] for k in keys}
    out = {}
    for rn, dec in rules.items():
        c = {k: (m if dec[k] else b)[k] for k in keys}
        s = scores(c, tasks)
        lo, hi, _ = boot(keys, tasks, video, ref, c, no_hld, reps=1000)
        out[rn] = dict(avg=s["avg"], d=s["avg"] - s0["avg"], rt=s["rt"], d_rt=s["rt"] - s0["rt"],
                       d_nh=no_hld(s) - no_hld(s0), read_bt=100 * sum(dec[k] for k in bt) / len(bt),
                       read_rt=100 * sum(dec[k] for k in rt) / len(rt), ci_nh_vs_a=(lo, hi),
                       d_nh_vs_a=no_hld(s) - no_hld(scores(ref, tasks)))
    return out, info, len(keys)


def sb(cfg):
    rows = {}
    for p in sorted(glob.glob(f"{RUNS}/{cfg['sb']}/out/results_incremental.jsonl")):
        for line in open(p):
            r = json.loads(line)
            rows[r["_key"]] = r
    rows = list(rows.values())
    qid = {r["_key"]: r["_key"].split("_", 1)[1] for r in rows}
    tl = {qid[r["_key"]]: r["timeline_tokens"] for r in rows}
    gs, info = gates(cfg, "sb", tl)
    has_tl = lambda r: r["timeline_tokens"] > 0  # noqa: E731

    def pick(dec):
        if dec is None:
            return lambda r: r["correct_l0"]
        if dec == "all":
            return lambda r: r["correct_l1"]
        return lambda r: r["correct_l1"] if has_tl(r) and dec[qid[r["_key"]]] else r["correct_l0"]

    rules = {"never (r4)": None, "always L1": "all", "label route": None, **gs}
    base, ref = pick(None), pick(gs["(a) ours"])
    out = {}
    for rn, dec in rules.items():
        f = pick(dec)
        read = 100 * sum(has_tl(r) and (dec == "all" or (isinstance(dec, dict) and dec[qid[r["_key"]]])) for r in rows) / len(rows)
        lo, hi, _ = boot_sb(rows, ref, f)
        out[rn] = dict(acc=acc(rows, f), d=acc(rows, f) - acc(rows, base), read=read, ci_vs_a=(lo, hi),
                       d_vs_a=acc(rows, f) - acc(rows, ref))
    return out, info, len(rows)


def main():
    final = {}
    for name, cfg in BACKBONES.items():
        try:
            o, oi, no = ovo(cfg)
            s, si, ns = sb(cfg)
        except (FileNotFoundError, KeyError) as e:
            print(f"\n=== {name}: not ready ({e})")
            continue
        print(f"\n=== {name}  OVO n={no} {oi} | SB n={ns} {si}")
        print(f"{'rule':22s} {'OVO avg':>7s} {'d':>6s} {'dRT':>6s} {'d noHLD':>7s} {'read BT/RT':>10s} | {'SB':>6s} {'d':>6s} {'read':>5s}"
              f" |    G   pass  vs (a): noHLD [CI]  SB [CI]")
        for rn in o:
            x, y = o[rn], s[rn]
            g = (x["d_nh"] + y["d"]) / 2
            ok = x["d"] >= 1.0 and x["d_rt"] >= -0.3 and y["d"] >= 0 and x["read_rt"] < 80
            print(f"{rn:22s} {x['avg']:7.2f} {x['d']:+6.2f} {x['d_rt']:+6.2f} {x['d_nh']:+7.2f} {x['read_bt']:5.0f}/{x['read_rt']:<4.0f}"
                  f" | {y['acc']:6.2f} {y['d']:+6.2f} {y['read']:5.0f} | {g:+5.2f}  {'yes' if ok else 'no ':3s}"
                  f"  {x['d_nh_vs_a']:+5.2f} [{x['ci_nh_vs_a'][0]:+.2f},{x['ci_nh_vs_a'][1]:+.2f}]"
                  f"  {y['d_vs_a']:+5.2f} [{y['ci_vs_a'][0]:+.2f},{y['ci_vs_a'][1]:+.2f}]")
            if name == B3:
                final[rn] = (g, ok)
    if final:
        ga = final["(a) ours"][0]
        win = [(g, rn) for rn, (g, ok) in final.items() if rn[:3] in ("(b)", "(c)") and ok and g >= ga + 0.3]
        print(f"\nB3 decision (pre-registered): (a) passes={final['(a) ours'][1]}, G(a)={ga:+.2f}; "
              + (f"main gate -> {max(win)[1]} (G {max(win)[0]:+.2f})" if win else "keep (a)"))


if __name__ == "__main__":
    main()
