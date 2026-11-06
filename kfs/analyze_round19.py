"""Round 19 (EXPERIMENTS.md): efficiency table from kfs/bench_latency.py (SimpleStream / ReadPast) and
kfs/bench_dhsm.py (D-HSM, round-18 settings) runs, plus the pre-set consistency checks. Host-side, stdlib only.

Checks (a timing is only reported for the code path that produced the round-17 / round-18 answers):
  write   generated caption == cached caption (cache/captions_*_s16), every clip
  gate    live p_past (5 decimals) == round-17 route.json
  answer  SimpleStream response == r4 run record; ReadPast response == round-17 composition
          (187 abstention-option questions: *_cinstr runs; others: cap16 if read else r4)
  D-HSM   response == round-18 record (Qwen2.5 backward = round-15 cell (4), realtime = R18 rt runs; Qwen3 = R18 q3)
Warm-up rows (first 5 clips / 5 questions) are excluded from the timing summary.

usage: python3 kfs/analyze_round19.py [--smoke]
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import load as pload  # noqa: E402
from analyze_round17 import raw  # noqa: E402
from analyze_round18_dhsm import CFG as DCFG  # noqa: E402
from analyze_round18_dhsm import load_ovo  # noqa: E402
from analyze_route import RUNS  # noqa: E402

K187 = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
BB = {  # backbone -> (bench run name, D-HSM bench run name, r4 run, cap16 run, cinstr suffix)
    "Qwen2.5-VL-7B (dev)": ("bench_ours_q25", "bench_dhsm_q25", "r4_ss", "cap16", ""),
    "Qwen3-VL-8B (dev)": ("bench_ours_q3", "bench_dhsm_q3", "r4_q3", "cap16_q3", "_q3"),
}
SMOKE = {"Qwen2.5-VL-7B (dev)": ("smoke_bench_q25_r19", "smoke_bench_dhsm_q25_r19")}


def newest(name):
    ds = [d for d in sorted(glob.glob(f"{RUNS}/*-{name}"))
          if json.load(open(f"{d}/meta.json")).get("status") == "done"]
    return ds[-1] if ds else None


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))] if xs else float("nan")


def ours(bb, run, r4n, capn, sfx):
    b = json.load(open(f"{run}/out/bench.json"))
    r4, cap = pload(r4n, RUNS), pload(capn, RUNS)
    cb, cm = raw("r4_cinstr" + sfx), raw("cap16_cinstr" + sfx)
    w = b.get("write", {}).get("rows", [])
    a = b.get("answer", {}).get("rows", [])
    bad_ss = [x["key"] for x in a if x["ss_response"] != r4[x["key"]]["response"]]
    ref = lambda x: ((cm if x["read"] else cb) if x["key"] in K187 else (cap if x["read"] else r4))[x["key"]]  # noqa: E731
    bad_rp = [x["key"] for x in a if x["rp_response"] != ref(x)["response"]]
    print(f"\n=== {bb}  ({run.rsplit('/', 1)[-1]}; {b['env']['gpu']}, weights {b['env']['weights_gb']:.1f} GB)")
    print(f"  checks: captions == cache {sum(x['same_as_cache'] for x in w)}/{len(w)}; "
          f"gate p_past == route.json {sum(x['p_past'] == x['p_past_stored'] for x in a)}/{len(a)}; "
          f"SimpleStream == r4 {len(a) - len(bad_ss)}/{len(a)} {bad_ss[:5]}; "
          f"ReadPast == round 17 {len(a) - len(bad_rp)}/{len(a)} {bad_rp[:5]}")
    wm, am = [x for x in w if not x["warmup"]], [x for x in a if not x["warmup"]]
    if wm:
        print(f"  write (per 16-s clip, n={len(wm)}): total median {med([x['total_s'] for x in wm]):.2f} s / p90 "
              f"{p90([x['total_s'] for x in wm]):.2f} s = RTF {med([x['total_s'] for x in wm]) / 16:.3f} | decode "
              f"{med([x['decode_s'] for x in wm]):.2f}, encode {med([x['encode_s'] for x in wm]):.2f}, generate "
              f"{med([x['generate_s'] for x in wm]):.2f} | out tokens {med([x['out_tokens'] for x in wm]):.0f} | "
              f"peak {max(x['peak_gb'] for x in wm):.1f} GB")
    if am:
        rd = [x for x in am if x["read"]]
        print(f"  answer (per question, n={len(am)}, read {len(rd)}): decode {med([x['decode_s'] for x in am]):.2f} | "
              f"encode {med([x['encode_s'] for x in am]):.2f} | gate {med([x['gate_s'] for x in am]) * 1000:.0f} ms")
        for name, k, pk in (("SimpleStream", "ss_total_s", "peak_gb_ss"), ("ReadPast", "rp_total_s", "peak_gb_rp")):
            print(f"    {name:12s} end-to-end median {med([x[k] for x in am]):.2f} s / p90 {p90([x[k] for x in am]):.2f} s;"
                  f" peak {max(x[pk] for x in am):.1f} GB")
        print(f"    ReadPast answer call: read {med([x['rp_answer_s'] for x in rd]):.2f} s (timeline "
              f"{med([x['timeline_tokens'] for x in rd]):.0f} tok) vs not read "
              f"{med([x['rp_answer_s'] for x in am if not x['read']]):.2f} s; extra over SimpleStream median "
              f"{med([x['rp_total_s'] - x['ss_total_s'] for x in am]):.2f} s / p90 "
              f"{p90([x['rp_total_s'] - x['ss_total_s'] for x in am]):.2f} s; vision tokens "
              f"{med([x['vision_tokens'] for x in am]):.0f}")


def dhsm(bb, run):
    b = json.load(open(f"{run}/out/bench_dhsm.json"))
    anno = json.load(open("kfs/data/bench_anno_r19.json"))
    keyof = {(f"/workspace/datasets/streaming/ovo_bench/chunked_videos/{a['id']}.mp4", a["question"]): f"{a['task']}:{a['id']}"
             for a in anno}
    warm = {f"{a['task']}:{a['id']}" for a in anno[:5]}
    ref, _ = load_ovo(DCFG[bb][0])
    rows = [{**r, "key": keyof[(r["video_path"], r["question"])]} for r in b["rows"]]
    bad = [r["key"] for r in rows if r["response"] != ref[r["key"]]["response"]]
    m = [r for r in rows if r["key"] not in warm]
    mem = [r for r in m if r.get("use_memory")]
    print(f"  D-HSM ({run.rsplit('/', 1)[-1]}): answers == round 18 {len(rows) - len(bad)}/{len(rows)} {bad[:5]}; "
          f"n={len(m)}, memory route {len(mem)}")
    if m:
        print(f"    end-to-end median {med([r['total_s'] for r in m]):.2f} s / p90 {p90([r['total_s'] for r in m]):.2f} s"
              f" | decode {med([r.get('decode_s', 0) for r in m]):.2f} | memory build (memory route) "
              f"{med([r.get('memory_s', 0) for r in mem]):.2f} s, of which captioning {med([r.get('caption_s', 0) for r in mem]):.2f} s "
              f"for {med([r.get('n_captioned', 0) for r in mem]):.0f} chunks | answer {med([r.get('answer_s', 0) for r in m]):.2f} s | "
              f"peak {max(r['peak_gb'] for r in m):.1f} GB")
        per_chunk = [r["caption_s"] / r["n_captioned"] for r in mem if r.get("n_captioned")]
        print(f"    captioning per 1-s chunk {med(per_chunk):.2f} s -> streaming RTF if every chunk were captioned "
              f"{med(per_chunk):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    for bb, (on, dn, r4n, capn, sfx) in BB.items():
        if args.smoke:
            if bb not in SMOKE:
                continue
            on, dn = SMOKE[bb]
        ro, rd = newest(on), newest(dn)
        if ro:
            ours(bb, ro, r4n, capn, sfx)
        else:
            print(f"\n=== {bb}: {on} not done")
        if rd:
            dhsm(bb, rd)


if __name__ == "__main__":
    main()
