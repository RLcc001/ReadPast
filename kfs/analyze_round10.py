"""Round 10 (pre-registered, EXPERIMENTS.md): timeline after the current view (--tl_pos after) vs before.

Main: mean over the 3 backbones of G(after) - G(before) with gate (a) >= +0.3, and on each backbone gate-(a)
OVO RT(after) >= RT(before) - 0.3. G = (OVO noHLD gain + SB gain) / 2 over r4.
Report only: always-read noHLD BT change per backbone; HLD change.

usage: python3 kfs/analyze_round10.py
"""

import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load, scores  # noqa: E402
from analyze_round9 import BACKBONES, ovo, sb  # noqa: E402
from analyze_route import RUNS  # noqa: E402

AFTER = {
    "Qwen2.5-VL-7B (dev)": ("cap16_after", "261102-0151*-sb_casc_after_s[0-3]"),
    "Qwen3-VL-8B (dev)": ("cap16_q3_after", "261102-0151*-sb_casc_q3_after_s[0-3]"),
    "Qwen3-VL-4B (B3, held-out)": ("cap16_q4_after", "261102-0151*-sb_casc_q4_after_s[0-3]"),
}


def mem_bt(r4, l1):
    b, m = load(r4, RUNS), load(l1, RUNS)
    keys = sorted(set(b) & set(m))
    tasks = {k: b[k]["task"] for k in keys}
    s = scores({k: m[k]["correct"] for k in keys}, tasks)
    return (s["EPM"] + s["ASI"]) / 2, s["HLD"]


def main():
    dg, ok_rt = [], True
    print(f"{'backbone':28s} {'pos':6s} {'OVO avg':>7s} {'dRT':>6s} {'d noHLD':>7s} {'SB d':>6s} {'G':>6s} | always: {'memBT':>6s} {'HLD':>6s}")
    for name, cfg in BACKBONES.items():
        res = {}
        for pos in ("before", "after"):
            c = dict(cfg)
            if pos == "after":
                c["ovo"], c["sb"] = (cfg["ovo"][0], AFTER[name][0]), AFTER[name][1]
            o, _, _ = ovo(c)
            s, _, _ = sb(c)
            x, y = o["(a) ours"], s["(a) ours"]
            g = (x["d_nh"] + y["d"]) / 2
            mb, hld_always = mem_bt(cfg["ovo"][0], c["ovo"][1])
            res[pos] = dict(g=g, rt=x["rt"])
            print(f"{name:28s} {pos:6s} {x['avg']:7.2f} {x['d_rt']:+6.2f} {x['d_nh']:+7.2f} {y['d']:+6.2f} {g:+6.2f} |"
                  f"         {mb:6.2f} {hld_always:6.2f} | always SB d {s['always L1']['d']:+.2f}, always OVO d {o['always L1']['d']:+.2f}")
        dg.append(res["after"]["g"] - res["before"]["g"])
        rt_ok = res["after"]["rt"] >= res["before"]["rt"] - 0.3
        ok_rt &= rt_ok
        print(f"{'':28s} G(after) - G(before) = {dg[-1]:+.2f}; RT(after) - RT(before) = {res['after']['rt'] - res['before']['rt']:+.2f}"
              f" [{'ok' if rt_ok else 'FAIL'}]")
    m = sum(dg) / len(dg)
    print(f"\nmean dG = {m:+.2f} (>= +0.3: {m >= 0.3}); RT ok on all: {ok_rt} -> {'switch to after' if m >= 0.3 and ok_rt else 'keep before'}")


if __name__ == "__main__":
    main()
