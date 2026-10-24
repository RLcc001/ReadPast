"""Compare L1 variants that were run only on a subset of questions (stdlib, host).

Each variant = cap16 with the answers of its subset runs overlaid (questions outside the subset get exactly the
cap16 prompt, so their cap16 answers stand in). Every variant is shown as always-L1, router only and module 1
(router OR L0 conf), with paired video-bootstrap CIs against the cap16 counterpart.

usage: python3 kfs/analyze_overlay.py relevance=qft500_relevance_a,qft500_relevance_b recency=qft500_recency_a,...
"""

import glob
import json
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import BACKWARD, REALTIME, load, scores  # noqa: E402
from analyze_route import RUNS, boot, no_hld, row  # noqa: E402


def main():
    r4, cap = load("r4_ss", RUNS), load("cap16", RUNS)
    route = json.load(open(sorted(glob.glob(f"{RUNS}/*-route_q25/out/route.json"))[-1]))
    keys = sorted(r4)
    tasks = {k: r4[k]["task"] for k in keys}
    video = {k: r4[k]["video"] for k in keys}
    variants = {"cap16": {}}
    for arg in sys.argv[1:]:
        name, runs = arg.split("=")
        variants[name] = {k: v for r in runs.split(",") for k, v in load(r, RUNS).items()}
    rules = {
        "always L1": lambda k: True,
        "router only": lambda k: route[k]["p_past"] >= 0.5,
        "module 1": lambda k: route[k]["p_past"] >= 0.5 or max(r4[k]["p_opts"]) < 0.5,
    }
    sb = scores({k: r4[k]["correct"] for k in keys}, tasks)
    print(" " * 35 + "  avg    bt    rt " + " ".join(f"{c:>5s}" for c in BACKWARD + REALTIME))
    print(row("r4_ss", sb))
    ref = {}
    for rule, use in rules.items():
        for name, over in variants.items():
            c = {k: (over.get(k, cap[k]) if use(k) else r4[k])["correct"] for k in keys}
            s = scores(c, tasks)
            tl = sum((over.get(k, cap[k]).get("timeline_tokens") or 0) for k in keys if use(k)) / max(1, sum(map(use, keys)))
            line = row(f"{rule} | {name}", s, sb) + f"  tl {tl:4.0f}"
            if name == "cap16":
                ref[rule] = c
            else:
                for lab, f in (("avg", lambda s: s["avg"]), ("noHLD", no_hld)):
                    lo, hi, p = boot(keys, tasks, video, ref[rule], c, f)
                    line += f"\n      vs cap16 [{lab:5s}] {f(s) - f(scores(ref[rule], tasks)):+.2f} [{lo:+.2f}, {hi:+.2f}] P(<=0) {p:.3f}"
            print(line)


if __name__ == "__main__":
    main()
