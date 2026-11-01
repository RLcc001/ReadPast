"""Round 8 experiment B: do history FRAMES dilute present perception (RT), by frame count rather than tokens?

Both backbones, same configs: r4 (recent 4), cap16 (text timeline, always L1), clip4 (CLIP top-4 history frames),
clip4_lr (same at 1/4 pixels), clip16_lr (16 history frames at 1/4 pixels, about the token count of clip4).
Pre-registered criteria (EXPERIMENTS.md, round 8 B) are checked on Qwen3:
  (i)   RT(clip4) <= RT(r4) - 1.0
  (ii)  RT(clip16_lr) <= RT(clip4)
  (iii) RT drop of cap16 < RT drop of clip4
CIs: paired video bootstrap (1000) against r4.

usage: python3 kfs/analyze_hist.py
"""

import json
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import REALTIME, load, scores  # noqa: E402
from analyze_route import RUNS, boot, no_hld  # noqa: E402

BACKBONES = {
    "Qwen2.5-VL-7B": dict(r4="r4_ss", cap16="cap16", clip4="clip4_ss", clip4_lr="clip4_lr_ss", clip16_lr="clip16_lr_ss"),
    "Qwen3-VL-8B": dict(r4="r4_q3", cap16="cap16_q3", clip4="clip4_ss_q3", clip4_lr="clip4_lr_ss_q3",
                        clip16_lr="clip16_lr_ss_q3"),
}


def vis_tokens(name):
    import glob
    p = sorted(glob.glob(f"{RUNS}/*-{name}/out/summary.json"))[-1]
    m = json.load(open(p))["metrics"]
    return m.get("vis_tokens_mean", 0.0), m.get("timeline_tokens_mean", 0.0)


def main():
    res = {}
    for bb, cfg in BACKBONES.items():
        runs = {k: load(v, RUNS) for k, v in cfg.items()}
        keys = sorted(set.intersection(*(set(r) for r in runs.values())))
        tasks = {k: runs["r4"][k]["task"] for k in keys}
        video = {k: runs["r4"][k]["video"] for k in keys}
        cor = {n: {k: r[k]["correct"] for k in keys} for n, r in runs.items()}
        s = {n: scores(c, tasks) for n, c in cor.items()}
        print(f"\n=== {bb} (n={len(keys)})")
        print(f"{'run':10s} {'vis':>6s} {'text':>6s}   avg    RT   noHLD   EPM   ASI   HLD  " + " ".join(f"{t:>5s}" for t in REALTIME)
              + "   dRT vs r4 [95% CI]")
        for n in cfg:
            v, t = vis_tokens(cfg[n])
            line = (f"{n:10s} {v:6.0f} {t:6.0f}  {s[n]['avg']:5.2f} {s[n]['rt']:5.2f} {no_hld(s[n]):6.2f} "
                    f"{s[n]['EPM']:5.1f} {s[n]['ASI']:5.1f} {s[n]['HLD']:5.1f}  "
                    + " ".join(f"{s[n][x]:5.1f}" for x in REALTIME))
            if n != "r4":
                lo, hi, _ = boot(keys, tasks, video, cor["r4"], cor[n], lambda x: x["rt"], reps=1000)
                line += f"   {s[n]['rt'] - s['r4']['rt']:+5.2f} [{lo:+.2f}, {hi:+.2f}]"
            print(line)
        lo, hi, _ = boot(keys, tasks, video, cor["clip4"], cor["clip16_lr"], lambda x: x["rt"], reps=1000)
        print(f"RT clip16_lr - clip4: {s['clip16_lr']['rt'] - s['clip4']['rt']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lo, hi, _ = boot(keys, tasks, video, cor["clip4"], cor["clip4_lr"], lambda x: x["rt"], reps=1000)
        print(f"RT clip4_lr - clip4:  {s['clip4_lr']['rt'] - s['clip4']['rt']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        res[bb] = s
    s = res["Qwen3-VL-8B"]
    d = {n: s[n]["rt"] - s["r4"]["rt"] for n in s}
    c1 = d["clip4"] <= -1.0
    c2 = s["clip16_lr"]["rt"] <= s["clip4"]["rt"]
    c3 = d["cap16"] > d["clip4"]
    print(f"\nQwen3 pre-registered: (i) {c1} (dRT clip4 {d['clip4']:+.2f})  (ii) {c2}  (iii) {c3} "
          f"(cap16 {d['cap16']:+.2f} vs clip4 {d['clip4']:+.2f})  => O1 {'PASS' if c1 and c2 and c3 else 'FAIL'}")


if __name__ == "__main__":
    main()
