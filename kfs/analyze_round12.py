"""Round 12 (pre-registered, EXPERIMENTS.md): answer-side abstention rule R1 on Qwen2.5 and Qwen3-VL-8B (offline).

R1: on questions with an abstain option u, if the model's answer o* != u and, renormalised over the non-abstain
options, log p'_E(o*) <= log p'_Q(o*) (E = the routed input, Q = the question and options alone, no frames/timeline),
answer u instead. Report only: Zhao-style calibration p_E / p_N ("N/A" question) and question-only PMI p_E / p_Q.

usage: python3 kfs/analyze_round12.py
"""

import glob
import json
import math
import re
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load, mcq, scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_route import RUNS, no_hld  # noqa: E402

ANNO = "/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json"
ABSTAIN = re.compile(r"^\s*(unable to answer|cannot tell)", re.I)
CFG = {"Qwen2.5-VL-7B": ("Qwen2.5-VL-7B (dev)", "prior_q25"), "Qwen3-VL-8B": ("Qwen3-VL-8B (dev)", "prior_q3"),
       "Qwen3-VL-4B": ("Qwen3-VL-4B (B3, held-out)", "prior_q4"),  # 4B: run once after adoption, report only
       "LLaVA-OV-7B": ("LLaVA-OV-7B (B4, report only)", "prior_lov")}  # round 13, report only
COLS = ["avg", "rt", "mem", "EPM", "ASI", "HLD", "nohld"]
EPS = 1e-6


def r1(rec, lp_q, opts):
    """Letter after R1, or the original answer."""
    ans = mcq(rec["response"])
    u = [i for i, o in enumerate(opts) if ABSTAIN.match(o)]
    if not u or not ans or ord(ans) - 65 == u[0] or ord(ans) - 65 >= len(opts):
        return ans
    c = [i for i in range(len(opts)) if i not in u]
    o = ord(ans) - 65
    pe = [max(rec["p_opts"][i], EPS) for i in c]
    pq = [math.exp(lp_q[i]) for i in c]
    g = math.log(max(rec["p_opts"][o], EPS) / sum(pe)) - math.log(math.exp(lp_q[o]) / sum(pq))
    return chr(65 + u[0]) if g <= 0 else ans


def ratio(rec, lp_ref):
    s = [math.log(max(p, EPS)) - lr for p, lr in zip(rec["p_opts"], lp_ref)]
    return chr(65 + max(range(len(s)), key=s.__getitem__))


def main():
    anno = {f"{a['task']}:{a['id']}": a for a in json.load(open(ANNO))}
    summary = {}
    for name, (bb, prior_name) in CFG.items():
        cfg = BACKBONES[bb]
        paths = sorted(glob.glob(f"{RUNS}/*-{prior_name}/out/prior.json"))
        if not paths:
            print(f"\n=== {name}: not ready")
            continue
        prior = json.load(open(paths[-1]))
        base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
        keys = sorted(set(base) & set(mem))
        tasks = {k: base[k]["task"] for k in keys}
        gt = {k: base[k]["ground_truth"] for k in keys}
        tl = {k: mem[k].get("timeline_tokens") or 0 for k in keys}
        a = gates(cfg, "ovo", tl)[0]["(a) ours"]
        src = {"r4 (SimpleStream)": {k: base[k] for k in keys}, "always L1": {k: mem[k] for k in keys},
               "gate (a)": {k: (mem if a[k] else base)[k] for k in keys}}
        rows, flips = {}, {}
        for sn, recs in src.items():
            variants = {
                "": {k: mcq(recs[k]["response"]) for k in keys},
                " + R1": {k: r1(recs[k], prior[k]["lp_q"], anno[k]["options"]) for k in keys},
                " + Zhao p_E/p_N (report)": {k: ratio(recs[k], prior[k]["lp_na"]) for k in keys},
                " + PMI p_E/p_Q (report)": {k: ratio(recs[k], prior[k]["lp_q"]) for k in keys},
            }
            for vn, ans in variants.items():
                c = {k: int(ans[k] == gt[k]) for k in keys}
                s = scores(c, tasks)
                s["mem"], s["nohld"] = (s["EPM"] + s["ASI"]) / 2, no_hld(s)
                rows[sn + vn] = s
                if vn == " + R1":
                    ch = [k for k in keys if ans[k] != variants[""][k]]
                    flips[sn] = (len(ch), sum(ans[k] == gt[k] for k in ch), sum(variants[""][k] == gt[k] for k in ch),
                                 sum(tasks[k] != "HLD" for k in ch))
        print(f"\n=== {name} OVO n={len(keys)}")
        print(f"{'':40s}" + "".join(f"{c:>7s}" for c in COLS))
        for n, s in rows.items():
            print(f"{n:40s}" + "".join(f"{s[c]:7.2f}" for c in COLS))
        for sn, (n, fix, brk, non_hld) in flips.items():
            print(f"R1 on {sn}: {n} answers changed (now right {fix}, were right {brk}; non-HLD {non_hld})")
        g, g1 = rows["gate (a)"], rows["gate (a) + R1"]
        d = {c: g1[c] - g[c] for c in COLS}
        ok = d["avg"] >= 1.0 and d["mem"] >= -1.0 and d["rt"] >= -0.3
        lead0 = g["avg"] - rows["r4 (SimpleStream)"]["avg"]
        lead1 = g1["avg"] - rows["r4 (SimpleStream) + R1"]["avg"]
        print(f"gate(a)+R1 - gate(a): avg {d['avg']:+.2f} [>= +1.0: {d['avg'] >= 1.0}], mem {d['mem']:+.2f}, RT {d['rt']:+.2f},"
              f" HLD {d['HLD']:+.2f} -> {'pass' if ok else 'fail'}; lead over SimpleStream {lead0:+.2f} -> {lead1:+.2f} (both + R1)")
        summary[name] = ok
        false_abstain(name, cfg, prior, base, mem, a, keys)
    dev = [v for n, v in summary.items() if "4B" not in n and "LLaVA" not in n]
    print(f"\nadopt R1 (both dev backbones pass): {all(dev)}")


def false_abstain(name, cfg, prior, base, mem, a, keys):
    """Answerable questions have no abstain option on OVO, so R1's cost is invisible there. Estimate it: would the R1
    test (log p_E(o*) <= log p_Q(o*), all options) fire on the answer actually given? Assumes adding an abstain option
    leaves the other options' relative probabilities unchanged."""
    out = {}
    for k in keys:
        r = (mem if a[k] else base)[k]
        ans = mcq(r["response"])
        if r["task"] == "HLD" or not ans or ord(ans) - 65 >= len(r["p_opts"]):
            continue
        o = ord(ans) - 65
        fire = math.log(max(r["p_opts"][o], EPS)) <= prior[k]["lp_q"][o]
        s = out.setdefault("EPM/ASI" if r["task"] in ("EPM", "ASI") else "RT", [0, 0, 0, 0])
        s[0 if r["correct"] else 2] += 1
        s[1 if r["correct"] else 3] += fire
    print("  est. R1 abstention on answerable questions (gate (a)): " + "; ".join(
        f"{g}: correct {s[1]}/{s[0]} ({100 * s[1] / s[0]:.1f}%), wrong {s[3]}/{s[2]} ({100 * s[3] / s[2]:.1f}%)"
        for g, s in out.items()))


if __name__ == "__main__":
    main()
