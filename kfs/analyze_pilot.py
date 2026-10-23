"""Paired analysis of kfs runs against the recent-only baseline (stdlib only, runs on the host).

For each history variant V vs base B (same questions):
  - plain accuracy (SimpleStream scoring: BT / RT task-macro averages, avg = (BT+RT)/2)
  - oracle gate:     per question correct(B) or correct(V)                -> headroom of a perfect switch
  - group gate:      BT questions from V, RT from B                       -> uses benchmark labels, reference only
  - confidence gate: use V only when B's max option prob < tau            -> sweep of trigger rate vs accuracy

usage: python3 kfs/analyze_pilot.py --base r4_ss --variants uni4_ss uni4_tag clip4_tag ...
Run names resolve to the newest /data/REN/runs/streamkv/*-<name>/out/results_incremental.jsonl.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict

BACKWARD = ["EPM", "ASI", "HLD"]
REALTIME = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]


def mcq(response: str | None) -> str | None:
    """Same parsing as SimpleStream's extract_mcq_answer."""
    if response is None or not str(response).strip():
        return None
    text = str(response).strip().upper()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([1-4])\b", text)
    return chr(64 + int(m.group(1))) if m else None


def load(name: str, runs_root: str) -> dict[str, dict]:
    paths = sorted(glob.glob(f"{runs_root}/*-{name}/out/results_incremental.jsonl"))
    if not paths:
        raise FileNotFoundError(name)
    rows = {}
    for line in open(paths[-1]):
        r = json.loads(line)
        r["correct"] = int(mcq(r["response"]) == r["ground_truth"])
        rows[r["_key"]] = r
    return rows


def scores(correct: dict[str, int], tasks: dict[str, str]) -> dict[str, float]:
    by = defaultdict(list)
    for k, c in correct.items():
        by[tasks[k]].append(c)
    acc = {t: 100.0 * sum(v) / len(v) for t, v in by.items()}
    bt = sum(acc[t] for t in BACKWARD if t in acc) / max(1, sum(t in acc for t in BACKWARD))
    rt = sum(acc[t] for t in REALTIME if t in acc) / max(1, sum(t in acc for t in REALTIME))
    return {"bt": bt, "rt": rt, "avg": (bt + rt) / 2, **acc}


def fmt(s: dict[str, float], ref: dict[str, float] | None = None) -> str:
    cols = ["avg", "bt", "rt"] + BACKWARD + REALTIME
    out = []
    for c in cols:
        v = s.get(c, float("nan"))
        out.append(f"{v:5.1f}" if ref is None else f"{v:5.1f}({v - ref.get(c, 0):+.1f})")
    return " ".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="r4_ss")
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--runs_root", default="/data/REN/runs/streamkv")
    ap.add_argument("--taus", nargs="+", type=float, default=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    args = ap.parse_args()

    base = load(args.base, args.runs_root)
    print("cols: avg bt rt " + " ".join(BACKWARD + REALTIME))
    for name in args.variants:
        var = load(name, args.runs_root)
        keys = sorted(set(base) & set(var))
        tasks = {k: base[k]["task"] for k in keys}
        b = {k: base[k]["correct"] for k in keys}
        v = {k: var[k]["correct"] for k in keys}
        sb = scores(b, tasks)
        print(f"\n=== {name} vs {args.base}  (n={len(keys)})")
        print(f"  base        {fmt(sb)}")
        print(f"  variant     {fmt(scores(v, tasks), sb)}")
        for grp, ts in (("BT", BACKWARD), ("RT", REALTIME)):
            ks = [k for k in keys if tasks[k] in ts]
            win = sum(v[k] > b[k] for k in ks)
            loss = sum(v[k] < b[k] for k in ks)
            print(f"  {grp} flips: +{win} / -{loss}")
        print(f"  oracle gate {fmt(scores({k: max(b[k], v[k]) for k in keys}, tasks), sb)}")
        print(f"  group gate  {fmt(scores({k: v[k] if tasks[k] in BACKWARD else b[k] for k in keys}, tasks), sb)}")
        for tau in args.taus:
            use = {k: max(base[k]["p_opts"]) < tau for k in keys}
            g = {k: v[k] if use[k] else b[k] for k in keys}
            rate_bt = sum(use[k] for k in keys if tasks[k] in BACKWARD) / max(1, sum(tasks[k] in BACKWARD for k in keys))
            rate_rt = sum(use[k] for k in keys if tasks[k] in REALTIME) / max(1, sum(tasks[k] in REALTIME for k in keys))
            print(f"  conf<{tau:<4}   {fmt(scores(g, tasks), sb)}  trigger BT {100 * rate_bt:4.0f}% RT {100 * rate_rt:4.0f}%")


if __name__ == "__main__":
    main()
