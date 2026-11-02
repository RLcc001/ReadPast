"""Is the text timeline real memory, and how far back does it reach? (offline, OVO, both backbones)

1. Content control per task: L1 with the video's own timeline vs a foreign timeline (same clips/timestamps, captions of
   another video), always-read and through gate (a). Gains that survive the swap come from memory content.
2. Gain vs distance on the memory tasks (EPM, ASI):
   - elapsed: the query time (how much past exists);
   - evidence distance: query time - end of the caption that best matches the ground-truth answer (content-word
     overlap, ties -> the latest caption). Uses the answer, so it is an analysis tool only. Questions whose answer
     matches no caption are reported separately.

usage: python3 kfs/analyze_memory_reach.py
"""

import json
import os
import re
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import REALTIME, load, scores  # noqa: E402
from analyze_round9 import BACKBONES, gates  # noqa: E402
from analyze_route import RUNS, boot  # noqa: E402

CFG = {
    "Qwen2.5-VL-7B": dict(cfg=BACKBONES["Qwen2.5-VL-7B (dev)"], base="r4_ss", own="cap16", foreign="cap16_foreign",
                          caps="cache/captions_q25_s16"),
    "Qwen3-VL-8B": dict(cfg=BACKBONES["Qwen3-VL-8B (dev)"], base="r4_q3", own="cap16_q3", foreign="cap16_q3_foreign",
                        caps="cache/captions_q3_s16"),
}
ANNO = "/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json"
from answer_match import STOP, words  # noqa: E402,F401


def captions(cap_dir, video):
    p = os.path.join(cap_dir, video.replace("/", "__") + ".json")
    return json.load(open(p))["captions"] if os.path.exists(p) else []


def evidence_distance(caps, anno):
    """Seconds between the query and the end of the best-matching caption, or None."""
    target = words(anno["answer"])
    best, best_t1 = 0, None
    for c in caps:
        if c["t1"] > anno["realtime"]:
            continue
        sc = len(target & words(c["text"]))
        if sc > best or (sc == best and sc > 0):
            best, best_t1 = sc, c["t1"]
    return None if best_t1 is None else anno["realtime"] - best_t1


def acc(c, keys):
    return 100 * sum(c[k] for k in keys) / len(keys) if keys else float("nan")


def main():
    anno = {f"{a['task']}:{a['id']}": a for a in json.load(open(ANNO))}
    for name, d in CFG.items():
        runs = {n: load(d[n], RUNS) for n in ("base", "own", "foreign")}
        keys = sorted(set.intersection(*(set(r) for r in runs.values())))
        tasks = {k: runs["base"][k]["task"] for k in keys}
        video = {k: runs["base"][k]["video"] for k in keys}
        cor = {n: {k: r[k]["correct"] for k in keys} for n, r in runs.items()}
        tl = {k: runs["own"][k].get("timeline_tokens") or 0 for k in keys}
        a = gates(d["cfg"], "ovo", tl)[0]["(a) ours"]
        cor["ours"] = {k: (cor["own"] if a[k] else cor["base"])[k] for k in keys}
        cor["ours_foreign"] = {k: (cor["foreign"] if a[k] else cor["base"])[k] for k in keys}

        print(f"\n=== {name}: 1. content control per task (accuracy; own - foreign with 95% video-bootstrap CI)")
        s = {n: scores(c, tasks) for n, c in cor.items()}
        cols = ["EPM", "ASI", "HLD", "rt"]
        print(f"{'':22s}" + "".join(f"{c:>8s}" for c in cols))
        for n, lab in (("base", "r4"), ("own", "always: own TL"), ("foreign", "always: foreign TL"),
                       ("ours", "gate: own TL"), ("ours_foreign", "gate: foreign TL")):
            print(f"{lab:22s}" + "".join(f"{s[n][c]:8.1f}" for c in cols))
        for pair in (("foreign", "own"), ("ours_foreign", "ours")):
            line = f"{pair[1]} - {pair[0]}:".ljust(22)
            for c in cols:
                lo, hi, _ = boot(keys, tasks, video, cor[pair[0]], cor[pair[1]], lambda x, c=c: x[c], reps=1000)
                line += f"  {s[pair[1]][c] - s[pair[0]][c]:+5.1f} [{lo:+.1f},{hi:+.1f}]"
            print(line)

        print(f"\n=== {name}: 2. memory tasks (EPM+ASI) by distance; gain = always-own TL - r4 (and - foreign TL)")
        mem = [k for k in keys if tasks[k] in ("EPM", "ASI")]
        el = {k: anno[k]["realtime"] for k in mem}
        dist = {k: evidence_distance(captions(d["caps"], anno[k]["video"]), anno[k]) for k in mem}
        for label, val, bins in (("elapsed (s)", el, [0, 60, 120, 240, 1e9]),
                                 ("evidence dist. (s)", dist, [0, 30, 90, 240, 1e9])):
            print(f"  {label}: bin        n    r4   own  foreign  ours | own-r4  own-foreign  ours-r4")
            for lo, hi in zip(bins, bins[1:]):
                ks = [k for k in mem if val[k] is not None and lo <= val[k] < hi]
                if not ks:
                    continue
                r = {n: acc(cor[n], ks) for n in ("base", "own", "foreign", "ours")}
                rng = f"[{lo:.0f},{'inf' if hi > 1e8 else f'{hi:.0f}'})"
                print(f"    {rng:>14s} {len(ks):5d} {r['base']:5.1f} {r['own']:5.1f} {r['foreign']:7.1f} {r['ours']:5.1f} |"
                      f" {r['own'] - r['base']:+6.1f} {r['own'] - r['foreign']:+11.1f} {r['ours'] - r['base']:+8.1f}")
            if label.startswith("evidence"):
                ks = [k for k in mem if dist[k] is None]
                r = {n: acc(cor[n], ks) for n in ("base", "own", "foreign", "ours")}
                print(f"    {'no match':>14s} {len(ks):5d} {r['base']:5.1f} {r['own']:5.1f} {r['foreign']:7.1f} {r['ours']:5.1f} |"
                      f" {r['own'] - r['base']:+6.1f} {r['own'] - r['foreign']:+11.1f} {r['ours'] - r['base']:+8.1f}")


if __name__ == "__main__":
    main()
