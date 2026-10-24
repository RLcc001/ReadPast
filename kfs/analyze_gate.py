"""Two-pass arbitration between the recent-only run and a history run (stdlib only, runs on the host).

Gate: take the history answer when max p(history) - max p(base) > d, else keep the base answer.
  plain     : as above
  keep-abst : additionally never override a base answer that picks an "Unable to answer" option
              (on OVO only HLD has such an option, so this is effectively "HLD uses the base"; reference only)

Prints the d sweep (tuned on the test set, optimistic) and a 2-fold cross-validation by video
(choose d on one half, evaluate on the other; 20 random splits x 2 folds) as the honest estimate.

usage: python3 kfs/analyze_gate.py clip4_ss mmr16_lr_tag_note ...
"""

import json
import random
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load, mcq, scores  # noqa: E402

RUNS = "/data/REN/runs/streamkv"
ANNO = "/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json"
D = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

annos = json.load(open(ANNO))
abst = {f"{a['task']}:{a['id']}": [i for i, x in enumerate(a.get("options") or []) if "unable" in str(x).lower()] for a in annos}
base = load("r4_ss", RUNS)


def gate(var, keys, d, keep_abst):
    out = {}
    for k in keys:
        b, v = base[k], var[k]
        if keep_abst and mcq(b["response"]) and (ord(mcq(b["response"])) - 65) in abst[k]:
            out[k] = b["correct"]
            continue
        out[k] = v["correct"] if max(v["p_opts"]) - max(b["p_opts"]) > d else b["correct"]
    return out


for name in sys.argv[1:]:
    var = load(name, RUNS)
    keys = sorted(set(base) & set(var))
    tasks = {k: base[k]["task"] for k in keys}
    sb = scores({k: base[k]["correct"] for k in keys}, tasks)
    for ka in (False, True):
        row = [f"{scores(gate(var, keys, d, ka), tasks)['avg'] - sb['avg']:+.2f}" for d in D]
        vids = sorted({base[k]["video"] for k in keys})
        cv = []
        for seed in range(20):
            random.Random(seed).shuffle(vids)
            half = set(vids[: len(vids) // 2])
            A = [k for k in keys if base[k]["video"] in half]
            B = [k for k in keys if base[k]["video"] not in half]
            for tr, te in ((A, B), (B, A)):
                sbe = scores({k: base[k]["correct"] for k in te}, tasks)
                dbest = max(D, key=lambda d: scores(gate(var, tr, d, ka), tasks)["avg"])
                cv.append(scores(gate(var, te, dbest, ka), tasks)["avg"] - sbe["avg"])
        tag = "keep-abst" if ka else "plain    "
        print(f"{name:18s} n={len(keys)} {tag} d={D}: {' '.join(row)} | CV {sum(cv) / len(cv):+.2f} [{min(cv):+.2f},{max(cv):+.2f}]")
