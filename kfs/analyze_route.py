"""Module 1 offline: route each question to L0 (r4_ss) or L1 (cap16) and combine the existing runs (stdlib, host).

L1 when p_past >= th (kfs/route.py) or max option prob of L0 < tau0. Since decoding is greedy, the combination
equals running the cascade's first two levels. Prints the fixed setting (th=0.5, tau0=0.5) with its ablations,
paired video-bootstrap CIs, a 2-fold-by-video CV over (th, tau0), and writes the L2 trigger key list
(L1 questions whose L1 max option prob < tau1) for kfs/eval_ovo_kfs.py --keys.

usage: python3 kfs/analyze_route.py [--route route_q25] [--tau1 0.6] [--keys_out cache/cascade_keys.json]
"""

import argparse
import glob
import json
import random
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import BACKWARD, REALTIME, load, scores  # noqa: E402

RUNS = "/data/REN/runs/streamkv"
THS = [0.3, 0.4, 0.5, 0.6, 0.7]
TAU0S = [0.0, 0.3, 0.5, 0.7, 0.9]


def no_hld(s):
    return ((s["EPM"] + s["ASI"]) / 2 + s["rt"]) / 2


def row(name, s, ref=None):
    cols = ["avg", "bt", "rt"] + BACKWARD + REALTIME
    d = f" ({s['avg'] - ref['avg']:+.2f})" if ref else ""
    return f"{name:34s} " + " ".join(f"{s[c]:5.1f}" for c in cols) + f"  noHLD {no_hld(s):5.2f}{d}"


def boot(keys, tasks, video, a, b, f, reps=500):
    vids = sorted({video[k] for k in keys})
    by = {v: [k for k in keys if video[k] == v] for v in vids}
    rng = random.Random(0)
    ds = []
    for _ in range(reps):
        ks = [(i, k) for i, v in enumerate(rng.choice(vids) for _ in vids) for k in by[v]]
        t = {x: tasks[x[1]] for x in ks}
        ds.append(f(scores({x: b[x[1]] for x in ks}, t)) - f(scores({x: a[x[1]] for x in ks}, t)))
    ds.sort()
    return ds[int(0.025 * reps)], ds[int(0.975 * reps)], sum(d <= 0 for d in ds) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", default="route_q25")
    ap.add_argument("--base", default="r4_ss")
    ap.add_argument("--mem", default="cap16")
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--tau0", type=float, default=0.5)
    ap.add_argument("--tau1", type=float, default=0.6)
    ap.add_argument("--keys_out", default="")
    args = ap.parse_args()

    route = json.load(open(sorted(glob.glob(f"{RUNS}/*-{args.route}/out/route.json"))[-1]))
    base, mem = load(args.base, RUNS), load(args.mem, RUNS)
    keys = sorted(set(base) & set(mem))
    tasks = {k: base[k]["task"] for k in keys}
    video = {k: base[k]["video"] for k in keys}
    pp = {k: route[k]["p_past"] for k in keys}
    p0 = {k: max(base[k]["p_opts"]) for k in keys}

    def l1(k, th, tau0):
        return pp[k] >= th or p0[k] < tau0

    def combine(ks, th, tau0):
        return {k: (mem if l1(k, th, tau0) else base)[k]["correct"] for k in ks}

    # router quality: BT labelled "past"
    pos = [pp[k] for k in keys if tasks[k] in BACKWARD]
    neg = [pp[k] for k in keys if tasks[k] in REALTIME]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    print(f"router AUC (BT=past vs RT=now) {auc:.3f}; p_past>=0.5: BT {100 * sum(p >= 0.5 for p in pos) / len(pos):.0f}% "
          f"RT {100 * sum(n >= 0.5 for n in neg) / len(neg):.0f}%")
    print("by task p_past>=th:", {t: round(100 * sum(pp[k] >= args.th for k in keys if tasks[k] == t) / sum(tasks[k] == t for k in keys)) for t in BACKWARD + REALTIME})

    b = {k: base[k]["correct"] for k in keys}
    m = {k: mem[k]["correct"] for k in keys}
    sb, sm = scores(b, tasks), scores(m, tasks)
    print("\n" + " " * 35 + "  avg    bt    rt " + " ".join(f"{c:>5s}" for c in BACKWARD + REALTIME))
    print(row(args.base, sb))
    print(row(args.mem + " (always L1)", sm, sb))
    variants = {
        "label route (BT->L1, reference)": {k: (m if tasks[k] in BACKWARD else b)[k] for k in keys},
        f"router only (p_past>={args.th})": combine(keys, args.th, 0.0),
        f"L0-conf only (p0<{args.tau0})": combine(keys, 2.0, args.tau0),
        f"module 1 (router OR conf)": combine(keys, args.th, args.tau0),
    }
    for name, c in variants.items():
        s = scores(c, tasks)
        rate = {g: 100 * sum(l1(k, args.th, args.tau0) if "module" in name else 0 for k in keys if tasks[k] in ts) / sum(tasks[k] in ts for k in keys)
                for g, ts in (("BT", BACKWARD), ("RT", REALTIME))}
        print(row(name, s, sb) + (f"  L1 rate BT {rate['BT']:.0f}% RT {rate['RT']:.0f}%" if "module" in name else ""))
    mod1 = variants["module 1 (router OR conf)"]
    for ref_name, ref in ((args.base, b), (args.mem, m)):
        for lab, f in (("avg", lambda s: s["avg"]), ("noHLD", no_hld)):
            lo, hi, p = boot(keys, tasks, video, ref, mod1, f)
            d = f(scores(mod1, tasks)) - f(scores(ref, tasks))
            print(f"module 1 - {ref_name:6s} [{lab:5s}] {d:+.2f}  CI [{lo:+.2f}, {hi:+.2f}]  P(<=0) {p:.3f}")

    # 2-fold CV by video over (th, tau0)
    vids = sorted(set(video.values()))
    cv_b, cv_m, picks = [], [], []
    for seed in range(20):
        random.Random(seed).shuffle(vids)
        half = set(vids[: len(vids) // 2])
        A = [k for k in keys if video[k] in half]
        B = [k for k in keys if video[k] not in half]
        for tr, te in ((A, B), (B, A)):
            th, tau0 = max(((t, u) for t in THS for u in TAU0S), key=lambda x: scores(combine(tr, *x), tasks)["avg"])
            picks.append((th, tau0))
            s = scores(combine(te, th, tau0), tasks)["avg"]
            cv_b.append(s - scores({k: b[k] for k in te}, tasks)["avg"])
            cv_m.append(s - scores({k: m[k] for k in te}, tasks)["avg"])
    print(f"\nCV (2-fold by video x20, grid th {THS} x tau0 {TAU0S}): vs {args.base} {sum(cv_b) / len(cv_b):+.2f} "
          f"[{min(cv_b):+.2f},{max(cv_b):+.2f}]  vs {args.mem} {sum(cv_m) / len(cv_m):+.2f} [{min(cv_m):+.2f},{max(cv_m):+.2f}]")
    print("  picked (th, tau0):", {p: picks.count(p) for p in sorted(set(picks))})

    trig = [k for k in keys if l1(k, args.th, args.tau0) and max(mem[k]["p_opts"]) < args.tau1]
    print(f"\nL2 trigger (L1 and L1 max p < {args.tau1}): {len(trig)} questions;",
          {t: sum(tasks[k] == t for k in trig) for t in BACKWARD + REALTIME})
    if args.keys_out:
        json.dump(trig, open(args.keys_out, "w"))
        print("wrote", args.keys_out)


if __name__ == "__main__":
    main()
