"""Round 16 (pre-registered, EXPERIMENTS.md): cost and selectivity of R1 when every OVO question has an abstention option.

A = 1281 answerable questions with an inserted "Unable to answer" option (kfs/make_inject.py); U = HLD (186); N = the one
EPM question that already has an abstention option. Inputs: r4 (SimpleStream), always-read (cap16), gate (a).
Answers are the argmax of p_opts over the option letters (letter E exists after insertion).

Runs per backbone (suffix "", "_q3", "_q4"): r4_inj, cap16_inj (A only), r4_inj_instr, cap16_inj_instr (all, D-HSM
instruction), prior_inj (text-only priors with the inserted options). U and N without the instruction come from the
existing r4 / cap16 runs.

usage: python3 kfs/analyze_round16.py [--reps 1000]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from collections import defaultdict

sys.path.insert(0, "kfs")
from analyze_pilot import scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_route import RUNS  # noqa: E402

EPS = 1e-6
CFG = {"Qwen2.5-VL-7B (dev)": "", "Qwen3-VL-8B (dev)": "_q3", "Qwen3-VL-4B (B3, held-out)": "_q4"}
DELTAS = [-1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0]
TAUS = [0.5, 0.7, 0.9]


def load(name):
    paths = sorted(glob.glob(f"{RUNS}/*-{name}/out/results_incremental.jsonl"))
    if not paths:
        return None
    return {(r := json.loads(line))["_key"]: r for line in open(paths[-1])}


def prior_of(name):
    paths = sorted(glob.glob(f"{RUNS}/*-{name}/out/prior.json"))
    return json.load(open(paths[-1])) if paths else None


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def pmi(rec, lp_q, u):
    """log p'_E(o*) - log p'_Q(o*), both renormalised over the non-abstention options; o* = argmax."""
    o = argmax(rec["p_opts"])
    c = [i for i in range(len(rec["p_opts"])) if i != u]
    pe = sum(max(rec["p_opts"][i], EPS) for i in c)
    pq = sum(math.exp(lp_q[i]) for i in c)
    return math.log(max(rec["p_opts"][o], EPS) / pe) - (lp_q[o] - math.log(pq))


def pe_norm(rec, u):
    o = argmax(rec["p_opts"])
    return rec["p_opts"][o] / sum(rec["p_opts"][i] for i in range(len(rec["p_opts"])) if i != u)


class Cond:
    """Per-question state of one input condition with an abstention option everywhere."""

    def __init__(self, recs, prior, keys, abst):
        self.keys, self.recs = keys, recs
        self.u = abst
        self.ans = {k: argmax(recs[k]["p_opts"]) for k in keys}
        self.gt = {k: ord(recs[k]["ground_truth"]) - 65 for k in keys}
        self.task = {k: recs[k]["task"] for k in keys}
        self.video = {k: recs[k]["video"] for k in keys}
        live = [k for k in keys if self.ans[k] != self.u[k]]
        self.g = {k: pmi(recs[k], prior[k]["lp_q"], self.u[k]) for k in live}
        self.pe = {k: pe_norm(recs[k], self.u[k]) for k in live}
        self.pq = {k: math.exp(prior[k]["lp_q"][self.ans[k]]) for k in live}

    def correct(self, flips=frozenset()):
        return {k: int((self.u[k] if k in flips else self.ans[k]) == self.gt[k]) for k in self.keys}

    def r1(self, delta=0.0, tau=None):
        return {k for k, g in self.g.items() if g <= -delta and (tau is None or self.pe[k] < tau)}

    def lowconf(self, n):
        return set(sorted(self.g, key=lambda k: (self.pe[k], k))[:n])

    def oracle(self):
        return {k for k in self.g if self.ans[k] != self.gt[k]}


def stats(c: Cond, flips, A, U):
    """Flip accounting on A (answerable) and U (HLD)."""
    s = {}
    a_live = [k for k in A if k in c.g]
    corr = [k for k in a_live if c.ans[k] == c.gt[k]]
    wrong = [k for k in a_live if c.ans[k] != c.gt[k]]
    fc, fw = sum(k in flips for k in corr), sum(k in flips for k in wrong)
    by = defaultdict(lambda: [0, 0, 0])  # task -> [n_live, n_wrong, k]
    for k in a_live:
        b = by[c.task[k]]
        b[0] += 1
        b[1] += c.ans[k] != c.gt[k]
        b[2] += k in flips
    exp_w = sum(k_ * w / n for n, w, k_ in by.values() if n)
    u_live = [k for k in U if k in c.g]
    fu = sum(k in flips for k in u_live)
    s.update(n_corr=len(corr), n_wrong=len(wrong), fc=fc, fw=fw, fu=fu, n_u_live=len(u_live),
             ER=100 * fc / max(1, len(corr)), WR=100 * fw / max(1, len(wrong)), catch=100 * fu / max(1, len(u_live)))
    s["S_raw"] = s["WR"] / s["ER"] if s["ER"] else float("inf")
    s["S_A"] = fw / exp_w if exp_w else float("nan")
    s["S_U"] = s["catch"] / s["ER"] if s["ER"] else float("inf")
    s["c*_A"] = fc / fw if fw else float("inf")
    s["c*_AU"] = fc / (fw + fu) if fw + fu else float("inf")
    s["good"], s["bad"] = fw + fu, fc + sum(k in flips for k in c.g if k not in A and k not in U and c.ans[k] == c.gt[k])
    return s


def ovo_scores(c: Cond, flips, A, U):
    corr = c.correct(flips)
    s = scores(corr, c.task)
    s["A_acc"] = 100 * sum(corr[k] for k in A) / len(A)
    s["mem"] = (s["EPM"] + s["ASI"]) / 2
    s["nohld"] = (s["mem"] + s["rt"]) / 2
    s["abst_A"] = 100 * sum((k in flips) or c.ans[k] == c.u[k] for k in A) / len(A)
    phi = [1 if corr[k] else (0 if ((k in flips) or c.ans[k] == c.u[k]) else -1) for k in c.keys]
    s["phi1"] = 100 * sum(phi) / len(phi)
    return s


def boot(c: Cond, A, U, f1, f0, reps, seed=0):
    """Video bootstrap: stratified S_A of f1, good(f1) - good(f0) (f0 = matched low-conf), OVO-inj(f1) - OVO-inj(none)."""
    vids = sorted(set(c.video.values()))
    per = defaultdict(list)
    for k in c.keys:
        per[c.video[k]].append(k)
    corr0, corr1 = c.correct(), c.correct(f1)
    rng = random.Random(seed)
    out = defaultdict(list)
    Aset, Uset = set(A), set(U)
    for _ in range(reps):
        sample = [rng.choice(vids) for _ in vids]
        by = defaultdict(lambda: [0, 0, 0])
        fw = gd = 0
        tc = defaultdict(lambda: [0, 0, 0])
        for v in sample:
            for k in per[v]:
                t = c.task[k]
                tc[t][0] += 1
                tc[t][1] += corr0[k]
                tc[t][2] += corr1[k]
                if k not in c.g:
                    continue
                wrong = c.ans[k] != c.gt[k]
                if k in Aset:
                    b = by[t]
                    b[0] += 1
                    b[1] += wrong
                    b[2] += k in f1
                    fw += wrong and k in f1
                if wrong and (k in Aset or k in Uset):
                    gd += (k in f1) - (k in f0)
        exp_w = sum(k_ * w / n for n, w, k_ in by.values() if n)
        out["S_A"].append(fw / exp_w if exp_w else float("nan"))
        out["good_diff"].append(gd)
        acc0 = {t: 100 * v[1] / v[0] for t, v in tc.items()}
        acc1 = {t: 100 * v[2] / v[0] for t, v in tc.items()}
        out["d_ovo"].append(macro(acc1) - macro(acc0))
    ci = {}
    for n, xs in out.items():
        xs = sorted(x for x in xs if x == x)
        ci[n] = (xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1])
    return ci


def macro(acc):
    bt = [acc[t] for t in ("EPM", "ASI", "HLD") if t in acc]
    rt = [acc[t] for t in ("OCR", "ACR", "ATR", "STU", "FPD", "OJR") if t in acc]
    return (sum(bt) / len(bt) + sum(rt) / len(rt)) / 2


def compose(gate, mem, base, keys):
    return {k: (mem if gate[k] else base)[k] for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    verdicts = {}
    for bb, sfx in CFG.items():
        cfg = BACKBONES[bb]
        names = {n: n + sfx for n in ("r4_inj", "cap16_inj", "r4_inj_instr", "cap16_inj_instr", "prior_inj")}
        base0, mem0 = (load(n) for n in cfg["ovo"])
        inj = {n: load(names[n]) for n in ("r4_inj", "cap16_inj", "r4_inj_instr", "cap16_inj_instr")}
        prior = prior_of(names["prior_inj"])
        if prior is None or inj["r4_inj"] is None or inj["cap16_inj"] is None:
            print(f"\n=== {bb}: not ready")
            continue
        A = sorted(inj["r4_inj"])
        assert len(A) == 1281 and set(A) == set(inj["cap16_inj"]), (len(A), len(inj["cap16_inj"]))
        keys = sorted(base0)
        U = [k for k in keys if base0[k]["task"] == "HLD"]
        abst = {k: prior[k]["abst_idx"] for k in keys}
        assert all(v is not None for v in abst.values())
        r4 = {k: (inj["r4_inj"] if k in inj["r4_inj"] else base0)[k] for k in keys}
        l1 = {k: (inj["cap16_inj"] if k in inj["cap16_inj"] else mem0)[k] for k in keys}
        tl = {k: mem0[k].get("timeline_tokens") or 0 for k in keys}
        ga = gates(cfg, "ovo", tl)[0]["(a) ours"]
        inputs = {"r4 (SimpleStream)": r4, "always read": l1, "gate (a)": compose(ga, l1, r4, keys)}
        print(f"\n=== {bb}: A={len(A)} U={len(U)} N={len(keys) - len(A) - len(U)}")

        # 0. effect of the inserted option itself (A only, no R1): original-option accuracy vs injected
        for iname, (b, m) in {"r4 (SimpleStream)": (base0, base0), "always read": (mem0, mem0), "gate (a)": (mem0, base0)}.items():
            orig = {k: (m if (iname == "always read" or (iname == "gate (a)" and ga[k])) else b)[k] for k in A}
            acc0 = 100 * sum(argmax(orig[k]["p_opts"]) == ord(orig[k]["ground_truth"]) - 65 for k in A) / len(A)
            recs = inputs[iname]
            acc1 = 100 * sum(argmax(recs[k]["p_opts"]) == ord(recs[k]["ground_truth"]) - 65 for k in A) / len(A)
            ab = 100 * sum(argmax(recs[k]["p_opts"]) == abst[k] for k in A) / len(A)
            print(f"  [option effect] {iname:18s} A acc {acc0:.2f} -> {acc1:.2f} ({acc1 - acc0:+.2f}), chooses abstention on {ab:.2f}% of A")

        conds = {n: Cond(r, prior, keys, abst) for n, r in inputs.items()}
        print(f"  {'input / rule':34s} {'OVO-inj':>8s} {'noHLD':>7s} {'HLD':>6s} {'A acc':>6s} {'abstA':>6s} {'phi1':>6s}"
              f" {'ER':>5s} {'WR':>5s} {'S_A':>5s} {'S_U':>5s} {'c*_A':>5s} {'good':>5s} {'bad':>4s}")
        res = {}
        for cn, c in conds.items():
            f_r1 = c.r1()
            k = len(f_r1)
            rules = {"none": set(), "+ R1": f_r1, f"lowconf (k={k})": c.lowconf(k), "oracle": c.oracle(),
                     "always abstain": set(c.g)}
            for rn, fl in rules.items():
                s, o = stats(c, fl, A, U), ovo_scores(c, fl, A, U)
                res[(cn, rn)] = (s, o)
                print(f"  {cn + ' ' + rn:34s} {o['avg']:8.2f} {o['nohld']:7.2f} {o['HLD']:6.2f} {o['A_acc']:6.2f} {o['abst_A']:6.2f}"
                      f" {o['phi1']:6.2f} {s['ER']:5.1f} {s['WR']:5.1f} {s['S_A']:5.2f} {s['S_U']:5.2f} {s['c*_A']:5.2f}"
                      f" {s['good']:5d} {s['bad']:4d}")
            n_live = len(c.g)
            wrong_live = sum(c.ans[q] != c.gt[q] for q in c.g)
            print(f"  {cn + ' random (k=' + str(k) + ', expected)':34s} good {k * wrong_live / n_live:.1f}, bad {k * (n_live - wrong_live) / n_live:.1f}")
        # instruction control (gate (a) composition of the instruction runs)
        if inj["r4_inj_instr"] and inj["cap16_inj_instr"]:
            ci_ = Cond(compose(ga, inj["cap16_inj_instr"], inj["r4_inj_instr"], keys), prior, keys, abst)
            o = ovo_scores(ci_, set(), A, U)
            res[("gate (a)", "instruction")] = (None, o)
            print(f"  {'gate (a) instruction (D-HSM)':34s} {o['avg']:8.2f} {o['nohld']:7.2f} {o['HLD']:6.2f} {o['A_acc']:6.2f}"
                  f" {o['abst_A']:6.2f} {o['phi1']:6.2f}")
        else:
            print("  instruction runs not ready")

        # sweeps (report only): R1 margin delta and confidence-gated R1 vs low-confidence at the same k (gate (a))
        c = conds["gate (a)"]
        print("  [sweep, gate (a)] rule: k  HLD  A-ER  good/bad   | lowconf same k: HLD  A-ER  good/bad")
        for d in DELTAS:
            for tau in [None] + TAUS:
                if d != 0.0 and tau is not None:
                    continue
                fl = c.r1(d, tau)
                lc = c.lowconf(len(fl))
                s, o = stats(c, fl, A, U), ovo_scores(c, fl, A, U)
                s2, o2 = stats(c, lc, A, U), ovo_scores(c, lc, A, U)
                tag = f"delta={d:+.2f}" + (f" tau<{tau}" if tau else "")
                print(f"    {tag:20s} {len(fl):4d} {o['HLD']:6.2f} {s['ER']:5.1f} {s['good']:4d}/{s['bad']:<4d} |"
                      f" {o2['HLD']:6.2f} {s2['ER']:5.1f} {s2['good']:4d}/{s2['bad']:<4d}")
        # buckets (report only): false abstentions by p'_Q(o*) quartile and by task (gate (a))
        f_r1 = c.r1()
        corrA = [q for q in A if q in c.g and c.ans[q] == c.gt[q]]
        qs = sorted(c.pq[q] for q in corrA)
        cuts = [qs[int(len(qs) * i / 4)] for i in (1, 2, 3)]
        bucket = defaultdict(lambda: [0, 0])
        for q in corrA:
            b = sum(c.pq[q] >= x for x in cuts)
            bucket[b][0] += 1
            bucket[b][1] += q in f_r1
        print("  [buckets] ER by p'_Q(o*) quartile (low->high): " + ", ".join(
            f"Q{b + 1} {100 * v[1] / v[0]:.1f}%" for b, v in sorted(bucket.items())))
        bt = defaultdict(lambda: [0, 0])
        for q in corrA:
            bt[c.task[q]][0] += 1
            bt[c.task[q]][1] += q in f_r1
        print("  [buckets] ER by task: " + ", ".join(f"{t} {100 * v[1] / v[0]:.1f}%" for t, v in sorted(bt.items())))
        # offline round-12 estimate vs actual (gate (a))
        old_prior = prior_of({"": "prior_q25", "_q3": "prior_q3", "_q4": "prior_q4"}[sfx])
        if old_prior:
            orig = compose(ga, mem0, base0, keys)
            est = defaultdict(lambda: [0, 0])
            for q in A:
                r = orig[q]
                o_ = argmax(r["p_opts"])
                if o_ == ord(r["ground_truth"]) - 65:
                    g = "EPM/ASI" if r["task"] in ("EPM", "ASI") else "RT"
                    est[g][0] += 1
                    est[g][1] += math.log(max(r["p_opts"][o_], EPS)) <= old_prior[q]["lp_q"][o_]
            act = defaultdict(lambda: [0, 0])
            for q in corrA:
                g = "EPM/ASI" if c.task[q] in ("EPM", "ASI") else "RT"
                act[g][0] += 1
                act[g][1] += q in f_r1
            print("  [offline estimate vs actual ER] " + "; ".join(
                f"{g}: est {100 * est[g][1] / est[g][0]:.1f}% vs actual {100 * act[g][1] / act[g][0]:.1f}%" for g in ("EPM/ASI", "RT")))

        # pre-registered criteria (gate (a))
        f_r1 = c.r1()
        lc = c.lowconf(len(f_r1))
        ci = boot(c, A, U, f_r1, lc, args.reps)
        s_r1, o_r1 = res[("gate (a)", "+ R1")]
        s_lc = res[("gate (a)", f"lowconf (k={len(f_r1)})")][0]
        o_none = res[("gate (a)", "none")][1]
        c1 = ci["S_A"][0] > 1
        c2 = s_r1["good"] >= s_lc["good"]
        ins = res.get(("gate (a)", "instruction"))
        c3 = None if ins is None else not (ins[1]["HLD"] >= o_r1["HLD"] and ins[1]["A_acc"] >= o_r1["A_acc"])
        c4 = o_r1["avg"] >= o_none["avg"]
        print(f"  [criteria, gate (a)] 1 S_A {s_r1['S_A']:.2f} CI [{ci['S_A'][0]:.2f}, {ci['S_A'][1]:.2f}] -> {c1};"
              f" 2 good R1 {s_r1['good']} vs lowconf {s_lc['good']} (diff CI [{ci['good_diff'][0]}, {ci['good_diff'][1]}]) -> {c2};"
              f" 3 not dominated by instruction -> {c3};"
              f" 4 OVO-inj {o_r1['avg'] - o_none['avg']:+.2f} CI [{ci['d_ovo'][0]:+.2f}, {ci['d_ovo'][1]:+.2f}] -> {c4}")
        verdicts[bb] = (c1, c2, c3, c4)
    dev = [v for b, v in verdicts.items() if "dev" in b]
    if len(dev) == 2:
        allc = [all(v[i] for v in dev) for i in range(4)]
        print(f"\ncriteria on both dev backbones: 1 {allc[0]}, 2 {allc[1]}, 3 {allc[2]}, 4 {allc[3]}")
        if all(allc):
            print("-> R1 in the main text as part of the method, with its cost")
        elif allc[0] and (allc[1] or allc[2]):
            print("-> keep as is: '+R1' row, HLD footnote, cost numbers in the limitations paragraph")
        else:
            print("-> move R1 out of the method into an appendix diagnostic")


if __name__ == "__main__":
    main()
