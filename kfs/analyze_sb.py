"""StreamingBench-RT: derive r4 / cap16 / module 1 / module 1 + L2 from kfs/eval_sb_kfs.py runs (stdlib, host).

Shards are merged from every /data/REN/runs/streamkv/*-<prefix>*/out/results_incremental.jsonl. L0 is checked
against SimpleStream's own r4 run (results/sb_qwen25vl_recent4).

usage: python3 kfs/analyze_sb.py --prefix sb_casc_desc [--th 0.5 --tau0 0.5]
"""

import argparse
import glob
import json
import random
from collections import defaultdict

RUNS = "/data/REN/runs/streamkv"
SS_R4 = "/data/REN/streamkv/results/sb_qwen25vl_recent4/results_incremental.jsonl"


def acc(rows, pick):
    return 100 * sum(pick(r) for r in rows) / len(rows)


def boot(rows, a, b, reps=1000):
    by = defaultdict(list)
    for r in rows:
        by[r["video"]].append(r)
    vids = sorted(by)
    rng = random.Random(0)
    ds = []
    for _ in range(reps):
        rs = [r for v in (rng.choice(vids) for _ in vids) for r in by[v]]
        ds.append(acc(rs, b) - acc(rs, a))
    ds.sort()
    return ds[int(0.025 * reps)], ds[int(0.975 * reps)], sum(d <= 0 for d in ds) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--th", type=float, default=0.5)
    ap.add_argument("--tau0", type=float, default=0.5)
    args = ap.parse_args()
    rows = {}
    for p in sorted(glob.glob(f"{RUNS}/*-{args.prefix}*/out/results_incremental.jsonl")):
        for line in open(p):
            r = json.loads(line)
            rows[r["_key"]] = r
    rows = list(rows.values())
    ss = {json.loads(line)["_key"]: json.loads(line) for line in open(SS_R4)}
    same = sum(r["response_l0"] == ss[r["_key"]]["response"] for r in rows if r["_key"] in ss)
    print(f"n={len(rows)}; L0 response == SimpleStream r4 run: {same}/{sum(r['_key'] in ss for r in rows)}")

    has_tl = lambda r: r["timeline_tokens"] > 0  # noqa: E731
    l1_m1 = lambda r: has_tl(r) and (r["p_past"] >= args.th or max(r["p_opts_l0"]) < args.tau0)  # noqa: E731
    l1_ro = lambda r: has_tl(r) and r["p_past"] >= args.th  # noqa: E731
    variants = {
        "r4 (L0)": lambda r: r["correct_l0"],
        "cap16 (always L1)": lambda r: r["correct_l1"],
        "module 1 (router OR conf)": lambda r: r["correct_l1"] if l1_m1(r) else r["correct_l0"],
        "module 1 + L2": lambda r: r["correct"],
        "router only": lambda r: r["correct_l1"] if l1_ro(r) else r["correct_l0"],
        "router only + L2": lambda r: (r["correct"] if r["level"] == 2 else r["correct_l1"]) if l1_ro(r) else r["correct_l0"],
    }
    if all("correct_l1f" in r for r in rows):
        variants.update({
            "focused (always L1f)": lambda r: r["correct_l1f"],
            "module 1 + focused": lambda r: r["correct_l1f"] if l1_m1(r) else r["correct_l0"],
            "router only + focused": lambda r: r["correct_l1f"] if l1_ro(r) else r["correct_l0"],
        })
    groups = defaultdict(list)
    for r in rows:
        groups[r["task_type"]].append(r)
        groups["[" + r["temporal_clue_type"] + "]"].append(r)
    cols = sorted(groups)
    print(f"{'':28s} {'all':>6s} " + " ".join(f"{c[:8]:>8s}" for c in cols))
    for name, f in variants.items():
        print(f"{name:28s} {acc(rows, f):6.2f} " + " ".join(f"{acc(groups[c], f):8.1f}" for c in cols))
    print(f"L1 rate module 1 {100 * sum(map(l1_m1, rows)) / len(rows):.0f}%, router only {100 * sum(map(l1_ro, rows)) / len(rows):.0f}%;"
          f" L2 {100 * sum(r['level'] == 2 for r in rows) / len(rows):.1f}%; with timeline {100 * sum(map(has_tl, rows)) / len(rows):.0f}%")
    base = variants["r4 (L0)"]
    for name in list(variants)[1:]:
        lo, hi, p = boot(rows, base, variants[name])
        print(f"{name:28s} - r4: {acc(rows, variants[name]) - acc(rows, base):+.2f}  CI [{lo:+.2f}, {hi:+.2f}]  P(<=0) {p:.3f}")


if __name__ == "__main__":
    main()
