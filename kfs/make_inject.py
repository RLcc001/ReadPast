"""Round 16: insert an "Unable to answer" option into every answerable OVO multiple-choice question.

Answerable set A = RT tasks + EPM + ASI questions that have no abstention option yet (1281 questions). HLD and the one
EPM question that already has an abstention option are left unchanged. The option is inserted (not substituted for a
distractor) at a uniformly random slot among the n+1 slots, seeded by sha1("task:id"); the original options keep their
text and relative order and the gold index shifts accordingly (applied by eval_ovo_kfs.apply_inject).

Outputs (host-side, stdlib only):
  kfs/data/ovo_inject_abst_v1.json   {"option": ..., "positions": {"<task>:<id>": slot}}
  kfs/data/keys_ovo_answerable1281.json  the keys of A

usage: python3 kfs/make_inject.py [--anno_path ...]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random

ABST_OPTION = "Unable to answer"
BACKWARD = ["EPM", "ASI", "HLD"]
REAL_TIME = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]


def is_abst(option: str) -> bool:
    return option.strip().lower().startswith(("unable to answer", "cannot tell"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anno_path", default="/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ovo_inject_abst_v1.json"))
    ap.add_argument("--keys_out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "keys_ovo_answerable1281.json"))
    args = ap.parse_args()

    positions: dict[str, int] = {}
    count = collections.Counter()
    for a in json.load(open(args.anno_path)):
        if a["task"] not in BACKWARD + REAL_TIME or a["task"] == "HLD" or any(is_abst(o) for o in a["options"]):
            continue
        key = f"{a['task']}:{a['id']}"
        seed = int(hashlib.sha1(key.encode()).hexdigest(), 16)
        positions[key] = random.Random(seed).randrange(len(a["options"]) + 1)
        count[a["task"]] += 1
    json.dump({"version": "v1", "option": ABST_OPTION, "positions": positions}, open(args.out, "w"), indent=0)
    json.dump(sorted(positions), open(args.keys_out, "w"))
    slots = collections.Counter(positions.values())
    print(f"{len(positions)} questions", dict(count), "slots", dict(sorted(slots.items())))


if __name__ == "__main__":
    main()
