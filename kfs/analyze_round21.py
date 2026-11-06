"""Round 21 analysis: label-free calibration of the OVO Forward yes/no decision (SSR, CRR).

usage: python3 kfs/analyze_round21.py [--runs fw_q25,fw_q3]   (host, stdlib)
Step 0: L0 verdicts vs official SimpleStream forward responses. Then per backbone x task x condition:
raw (generated answer, SimpleStream scoring), B (p_yes >= p_cf, primary), A (text-only content-free), C (p_yes(L0) >=
p_yes(gray frames), L0 only), best (threshold tuned on test labels = upper bound), AUC, Yes rate.
Pre-registered criteria (EXPERIMENTS.md round 21), on L0, both backbones:
  1. CRR: B >= raw + 5    2. SSR: B >= raw - 1    3. CRR: best - B <= 3
  1' (added after the 17-probe smoke showed B can collapse to always-Yes): CRR B > always-Yes accuracy.
Report-only label-free variant BC (batch calibration): threshold = mean p_yes over all probes of the task x condition.
"""
import argparse, glob, json, random, re, sys
from collections import defaultdict

RUNS = "/data/REN/runs/streamkv"
OFFICIAL = {"q25": "/data/REN/streamkv/results/ovo_qwen25vl_recent4/results_incremental.jsonl",
            "q3": f"{RUNS}/261101-014501-ss_q3_ovo_r4/out/results_incremental.jsonl"}


def score_yesno(response, gt_type):  # = SimpleStream ovo_constants.score_yesno
    if response is None or not str(response).strip():
        return 0
    s = str(response).strip().upper()
    if (s == "N" or "NO" in s) and gt_type == 0:
        return 1
    if (s == "Y" or "YES" in s) and gt_type == 1:
        return 1
    return 0


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg))


def best_threshold(rows):
    vals = sorted({r["p_yes"] for r in rows})
    cands = [-1.0] + vals + [2.0]
    best = max(cands, key=lambda th: sum((r["p_yes"] >= th) == (r["type"] == 1) for r in rows))
    return best, 100 * sum((r["p_yes"] >= best) == (r["type"] == 1) for r in rows) / len(rows)


def acc(rows, pred):
    return 100 * sum(pred(r) == (r["type"] == 1) for r in rows) / len(rows)


def boot_diff(rows, pa, pb, reps=1000):
    by = defaultdict(list)
    for r in rows:
        by[r["id"]].append(r)
    ids = sorted(by)
    rng, ds = random.Random(0), []
    for _ in range(reps):
        rs = [r for i in (rng.choice(ids) for _ in ids) for r in by[i]]
        ds.append(acc(rs, pa) - acc(rs, pb))
    ds.sort()
    return ds[int(.025 * reps)], ds[int(.975 * reps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="fw_q25,fw_q3")
    args = ap.parse_args()
    verdict = defaultdict(dict)
    for name in args.runs.split(","):
        paths = sorted(glob.glob(f"{RUNS}/*-{name}/out"))
        if not paths:
            print(f"== {name}: not found"); continue
        out = paths[-1]
        bb = "q3" if "q3" in name else "q25"
        rows = [json.loads(l) for l in open(f"{out}/probes.jsonl")]
        cf = json.load(open(f"{out}/cf.json"))
        P = {(r["task"], r["id"], r["index"], r["cond"]): r for r in rows}
        print(f"\n===== {name}  ({out})  probes: {len(rows)}")
        # Step 0
        off = {}
        for l in open(OFFICIAL[bb]):
            a = json.loads(l)
            if a.get("task") in ("SSR", "CRR"):
                for i, p in enumerate(a["test_info"]):
                    off[(a["task"], a["id"], i)] = p.get("response")
        same_v = same_w = n0 = 0
        for (t, i, k, c), r in P.items():
            if c != "L0" or (t, i, k) not in off:
                continue
            n0 += 1
            same_v += score_yesno(off[(t, i, k)], r["type"]) == r["score"]
            w = lambda s: (re.findall(r"[a-z]+", (s or "").lower()) or [""])[0]  # noqa: E731
            same_w += w(off[(t, i, k)]) == w(r["response"])
        print(f"Step 0 (L0 vs official SimpleStream): same verdict {same_v}/{n0}, same first word {same_w}/{n0}")
        for t in ("SSR", "CRR"):
            for c in ("L0", "L1"):
                rs = [r for r in rows if r["task"] == t and r["cond"] == c]
                if not rs or c not in cf.get(t, {}):
                    continue
                pB, pA = cf[t][c]["B"]["p_cf"], cf[t][c]["A"]["p_cf"]
                raw = 100 * sum(r["score"] for r in rs) / len(rs)
                fB = lambda r, th=pB: r["p_yes"] >= th  # noqa: E731
                fA = lambda r, th=pA: r["p_yes"] >= th  # noqa: E731
                accB, accA = acc(rs, fB), acc(rs, fA)
                th_best, acc_best = best_threshold(rs)
                th_bc = sum(r["p_yes"] for r in rs) / len(rs)
                accBC = acc(rs, lambda r: r["p_yes"] >= th_bc)
                always_yes = 100 * sum(r["type"] == 1 for r in rs) / len(rs)
                yes_raw = 100 * sum("YES" in (r["response"] or "").upper() for r in rs) / len(rs)
                yes_B = 100 * sum(fB(r) for r in rs) / len(rs)
                a_ = auc([r["p_yes"] for r in rs if r["type"] == 1], [r["p_yes"] for r in rs if r["type"] == 0])
                lo, hi = boot_diff(rs, fB, lambda r: score_yesno(r["response"], 1) == 1)
                line = (f"  {t} {c}: n={len(rs)} raw {raw:5.1f} (Yes {yes_raw:4.1f}%) | B {accB:5.1f} (Yes {yes_B:4.1f}%, "
                        f"p_cf {pB:.4f}; B-raw {accB - raw:+5.1f} [{lo:+.1f},{hi:+.1f}]) | A {accA:5.1f} (p_cf {pA:.4f}) | "
                        f"BC {accBC:5.1f} (th {th_bc:.4f}) | best {acc_best:5.1f} (th {th_best:.4f}) | AUC {a_:.3f} | "
                        f"always-Yes {always_yes:5.1f}")
                if c == "L0":
                    pairs = [(r, P[(r["task"], r["id"], r["index"], "L0gray")]) for r in rs
                             if (r["task"], r["id"], r["index"], "L0gray") in P]
                    accC = 100 * sum((r["p_yes"] >= x["p_yes"]) == (r["type"] == 1) for r, x in pairs) / max(1, len(pairs))
                    line += f" | C {accC:5.1f}"
                    verdict[bb][t] = dict(raw=raw, B=accB, best=acc_best, always_yes=always_yes)
                print(line)
                if t == "CRR":
                    for lab, sub in (("type0 (before clue)", [r for r in rs if r["type"] == 0]),
                                     ("type1 clue<=4s ago", [r for r in rs if r["type"] == 1 and r["realtime"] - r["clue_time"] <= 4]),
                                     ("type1 clue>4s ago", [r for r in rs if r["type"] == 1 and r["realtime"] - r["clue_time"] > 4])):
                        if sub:
                            print(f"      {lab:22s} n={len(sub):3d} raw {100 * sum(r['score'] for r in sub) / len(sub):5.1f}"
                                  f"  B {acc(sub, fB):5.1f}")
            if t == "CRR" and all(c in cf.get(t, {}) for c in ("L0", "L1")):
                print("    (L1 timeline tokens mean: %.0f)" % (sum(r["timeline_tokens"] for r in rows if r["task"] == t and r["cond"] == "L1") /
                                                         max(1, sum(1 for r in rows if r["task"] == t and r["cond"] == "L1"))))
    if len(verdict) == 2:
        c1 = all(v["CRR"]["B"] >= v["CRR"]["raw"] + 5 for v in verdict.values())
        c2 = all(v["SSR"]["B"] >= v["SSR"]["raw"] - 1 for v in verdict.values())
        c3 = all(v["CRR"]["best"] - v["CRR"]["B"] <= 3 for v in verdict.values())
        c1b = all(v["CRR"]["B"] > v["CRR"]["always_yes"] for v in verdict.values())
        print(f"\nCriteria (L0, both backbones): 1 CRR B>=raw+5: {c1} | 1' CRR B>always-Yes: {c1b} | 2 SSR B>=raw-1: {c2} | "
              f"3 CRR best-B<=3: {c3} => {'PASS' if c1 and c1b and c2 and c3 else 'FAIL'}")
        for bb, v in verdict.items():
            d = (v["SSR"]["B"] - v["SSR"]["raw"] + v["CRR"]["B"] - v["CRR"]["raw"]) / 3
            print(f"  {bb}: Forward avg change if REC unchanged (B vs raw, L0): {d:+.2f}")


if __name__ == "__main__":
    main()
