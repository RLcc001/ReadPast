"""Round 14 diagnostic (pre-registered, EXPERIMENTS.md): is the information in the timeline but unreadable, or not written?

On the 445 OVO memory questions (EPM, ASI), each backbone reads (always L1, recent 4 frames):
  full    : the whole timeline (existing runs cap16 / cap16_q3)
  oracle1 : only the caption that best matches the ground-truth answer (kfs/answer_match.py; uses the answer)
  random1 : one random other caption of the same timeline (same length scale, wrong content)
  r4      : no timeline (SimpleStream)
Reported on the located subset (some caption shares a content word with the answer):
  H = oracle1 - full    (headroom for reading / retrieval)
  C = oracle1 - random1 (value of the right caption's content)
  S = random1 - full    (effect of a short timeline by itself)
  L = a text-only lexical rule on the oracle caption (pick the option with the most shared content words), reference
      for how much of oracle1 a word-matching shortcut alone could explain.

usage: python3 kfs/analyze_diag_oracle.py
"""

import json
import random
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load, mcq  # noqa: E402
from analyze_route import RUNS  # noqa: E402
from answer_match import words  # noqa: E402

ANNO = "/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json"
CFG = {
    "Qwen2.5-VL-7B": dict(r4="r4_ss", full="cap16", oracle="diag_oracle1_q25", random="diag_random1_q25",
                          caps="cache/captions_q25_s16"),
    "Qwen3-VL-8B": dict(r4="r4_q3", full="cap16_q3", oracle="diag_oracle1_q3", random="diag_random1_q3",
                        caps="cache/captions_q3_s16"),
}


def acc(c, keys):
    return 100 * sum(c[k] for k in keys) / len(keys) if keys else float("nan")


def boot(keys, video, a, b, reps=1000):
    """Video-level bootstrap CI of acc(b) - acc(a)."""
    vids = sorted({video[k] for k in keys})
    by = {v: [k for k in keys if video[k] == v] for v in vids}
    rng, ds = random.Random(0), []
    for _ in range(reps):
        ks = [k for v in (rng.choice(vids) for _ in vids) for k in by[v]]
        ds.append(acc(b, ks) - acc(a, ks))
    ds.sort()
    return ds[int(0.025 * reps)], ds[int(0.975 * reps)]


def lexical(anno, cap_text):
    """Expected accuracy of: pick the option sharing the most content words with the caption (ties at random)."""
    cw = words(cap_text)
    sc = [len(words(o) & cw) for o in anno["options"]]
    top = [i for i, s in enumerate(sc) if s == max(sc)]
    return (anno["gt"] in top) / len(top)


def main():
    anno = {f"{a['task']}:{a['id']}": a for a in json.load(open(ANNO))}
    verdict = {}
    for name, d in CFG.items():
        try:
            runs = {n: load(d[n], RUNS) for n in ("r4", "full", "oracle", "random")}
        except Exception as e:  # noqa: BLE001
            print(f"\n=== {name}: not ready ({e})")
            continue
        keys = sorted(k for k in runs["oracle"] if k in runs["random"] and k in runs["full"] and k in runs["r4"])
        cor = {n: {k: int(mcq(r[k]["response"]) == r[k]["ground_truth"]) for k in keys} for n, r in runs.items()}
        video = {k: runs["oracle"][k]["video"] for k in keys}
        loc = [k for k in keys if runs["oracle"][k]["tl_oracle_idx"] is not None]
        unloc = [k for k in keys if k not in set(loc)]
        single = [k for k in loc if runs["oracle"][k]["n_captions_all"] == 1]
        caps = {}
        for k in loc:
            v = anno[k]["video"]
            if v not in caps:
                caps[v] = json.load(open(f"{d['caps']}/{v.replace('/', '__')}.json"))["captions"]
        lex = {}
        for k in loc:
            t0, t1 = runs["oracle"][k]["tl_sel"]
            text = next(c["text"] for c in caps[anno[k]["video"]] if c["t0"] == t0 and c["t1"] == t1)
            lex[k] = lexical(anno[k], text)
        print(f"\n=== {name}: n={len(keys)}, located {len(loc)} ({100 * len(loc) / len(keys):.0f}%), "
              f"unlocated {len(unloc)}; located with a single caption (oracle = full) {len(single)}")
        print(f"{'subset':12s} {'n':>4s} {'r4':>6s} {'full':>6s} {'oracle1':>8s} {'random1':>8s} {'lexical':>8s}")
        for sub, ks in [("located", loc), ("  EPM", [k for k in loc if anno[k]["task"] == "EPM"]),
                        ("  ASI", [k for k in loc if anno[k]["task"] == "ASI"]), ("unlocated", unloc)]:
            lx = acc(lex, ks) if sub != "unlocated" else float("nan")
            print(f"{sub:12s} {len(ks):4d} {acc(cor['r4'], ks):6.1f} {acc(cor['full'], ks):6.1f} {acc(cor['oracle'], ks):8.1f}"
                  f" {acc(cor['random'], ks):8.1f} {lx:8.1f}")
        h = acc(cor["oracle"], loc) - acc(cor["full"], loc)
        c = acc(cor["oracle"], loc) - acc(cor["random"], loc)
        s = acc(cor["random"], loc) - acc(cor["full"], loc)
        for lab, val, a, b in [("H oracle1-full", h, "full", "oracle"), ("C oracle1-random1", c, "random", "oracle"),
                               ("S random1-full", s, "full", "random")]:
            lo, hi = boot(loc, video, cor[a], cor[b])
            print(f"  {lab:18s} {val:+6.2f} [{lo:+.2f}, {hi:+.2f}]")
        fix = sum(cor["oracle"][k] and not cor["full"][k] for k in loc)
        brk = sum(cor["full"][k] and not cor["oracle"][k] for k in loc)
        print(f"  oracle1 vs full on located: fixes {fix}, breaks {brk}")
        verdict[name] = h
    if len(verdict) == 2:
        hs = list(verdict.values())
        if all(x >= 10 for x in hs):
            v = "reading/retrieval headroom (H >= +10 on both): the information is in the timeline but the full timeline does not let the model use it"
        elif all(x < 5 for x in hs):
            v = "no reading headroom (H < +5 on both): the bottleneck is what is written, not reading"
        else:
            v = "mixed (between the thresholds or backbones disagree): report only"
        print(f"\npre-registered verdict: {v}")


if __name__ == "__main__":
    main()
