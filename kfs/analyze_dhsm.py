"""Round 15 (pre-registered, EXPERIMENTS.md): public D-HSM code on OVO backward (Qwen2.5-VL-7B), 2x2 =
history {recent_only, dhsm} x answer prompt {official, uniform_abstention}.

  M_D    = mean(EPM, ASI) of dhsm/official - recent_only/official   (memory gain of D-HSM on memory tasks)
  M_ours = mean(EPM, ASI) of our gate (a) - r4_ss on the same questions (+ always L1, report)
  HLD: instruction effect (recent_only/uniform - recent_only/official), memory effect, both.

usage: python3 kfs/analyze_dhsm.py
"""

import glob
import json
import random
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load, mcq  # noqa: E402
from analyze_route import RUNS  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402

CELLS = {"(1) recent_only / official": "dhsm_recent_official", "(2) recent_only / abstain": "dhsm_recent_abstain",
         "(3) dhsm / official": "dhsm_mem_official", "(4) dhsm / abstain (paper)": "dhsm_mem_abstain"}
PAPER = {"EPM": 52.9, "ASI": 63.5, "HLD": 72.0}
TASKS = ["EPM", "ASI", "HLD"]


def load_dhsm(name):
    """Rows of one cell, merged over its shard runs (<name> or <name>_s<k>); latest result file per run."""
    rows = {}
    for run in sorted(glob.glob(f"{RUNS}/*-{name}")) + sorted(glob.glob(f"{RUNS}/*-{name}_s[0-9]*")):
        paths = sorted(glob.glob(f"{run}/out/hub_and_spoke_results_*.json"))
        if paths:
            rows.update({f"{r['task']}:{r['id']}": r for r in json.load(open(paths[-1]))["backward"]})
    if not rows:
        raise FileNotFoundError(name)
    return rows


def acc(c, keys):
    return 100 * sum(c[k] for k in keys) / len(keys) if keys else float("nan")


def mem_mean(c, keys, task):
    return (acc(c, [k for k in keys if task[k] == "EPM"]) + acc(c, [k for k in keys if task[k] == "ASI"])) / 2


def boot(keys, task, video, a, b, reps=1000):
    vids = sorted({video[k] for k in keys})
    by = {v: [k for k in keys if video[k] == v] for v in vids}
    rng, ds = random.Random(0), []
    for _ in range(reps):
        ks = [k for v in (rng.choice(vids) for _ in vids) for k in by[v]]
        ds.append(mem_mean(b, ks, task) - mem_mean(a, ks, task))
    ds.sort()
    return ds[int(0.025 * reps)], ds[int(0.975 * reps)]


def main():
    runs = {}
    for cell, name in CELLS.items():
        try:
            runs[cell] = load_dhsm(name)
        except FileNotFoundError:
            print(f"{cell}: not ready")
    cfg = BACKBONES["Qwen2.5-VL-7B (dev)"]
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    keys = sorted(set(base) & set(mem) & set.intersection(*(set(r) for r in runs.values())) if runs else set())
    keys = [k for k in keys if base[k]["task"] in TASKS]
    task = {k: base[k]["task"] for k in keys}
    video = {k: base[k]["video"] for k in keys}
    a = gates(cfg, "ovo", {k: mem[k].get("timeline_tokens") or 0 for k in keys})[0]["(a) ours"]
    cor = {cell: {k: int((mcq(r[k]["response"]) or "") == r[k]["ground_truth"]) for k in keys} for cell, r in runs.items()}
    cor["ours r4 (SimpleStream)"] = {k: base[k]["correct"] for k in keys}
    cor["ours always L1"] = {k: mem[k]["correct"] for k in keys}
    cor["ours gate (a)"] = {k: (mem if a[k] else base)[k]["correct"] for k in keys}
    errs = {cell: sum(1 for k in keys if r[k].get("response") is None) for cell, r in runs.items()}
    print(f"n={len(keys)} ({', '.join(f'{t} {sum(task[k] == t for k in keys)}' for t in TASKS)}); missing responses: {errs}")
    print(f"{'':30s}" + "".join(f"{t:>7s}" for t in TASKS) + f"{'EPM/ASI':>9s}{'BT':>7s}")
    for cell, c in cor.items():
        per = {t: acc(c, [k for k in keys if task[k] == t]) for t in TASKS}
        print(f"{cell:30s}" + "".join(f"{per[t]:7.1f}" for t in TASKS) + f"{(per['EPM'] + per['ASI']) / 2:9.2f}{sum(per.values()) / 3:7.2f}")
    print(f"{'paper (D-HSM, Qwen2.5, 4f)':30s}" + "".join(f"{PAPER[t]:7.1f}" for t in TASKS))
    c1, c3 = "(1) recent_only / official", "(3) dhsm / official"
    if c1 in cor and c3 in cor:
        md = mem_mean(cor[c3], keys, task) - mem_mean(cor[c1], keys, task)
        mo = mem_mean(cor["ours gate (a)"], keys, task) - mem_mean(cor["ours r4 (SimpleStream)"], keys, task)
        ml = mem_mean(cor["ours always L1"], keys, task) - mem_mean(cor["ours r4 (SimpleStream)"], keys, task)
        lo, hi = boot(keys, task, video, cor[c1], cor[c3])
        print(f"\nM_D (D-HSM memory gain on EPM/ASI, official prompt) = {md:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        print(f"M_ours (gate (a) - r4) = {mo:+.2f}; always L1 - r4 = {ml:+.2f} (report)")
        v = ("holds: no extra memory gain from the structure (M_D <= M_ours + 1.0)" if md <= mo + 1.0 else
             "structure helps (M_D >= M_ours + 3.0)" if md >= mo + 3.0 else "between thresholds: report only")
        print(f"pre-registered verdict: {v}")
    hld = {cell: acc(c, [k for k in keys if task[k] == "HLD"]) for cell, c in cor.items()}
    if all(c in hld for c in CELLS):
        b = hld["(1) recent_only / official"]
        print(f"HLD vs (1): instruction {hld['(2) recent_only / abstain'] - b:+.1f}, memory {hld['(3) dhsm / official'] - b:+.1f},"
              f" both {hld['(4) dhsm / abstain (paper)'] - b:+.1f}")


if __name__ == "__main__":
    main()
