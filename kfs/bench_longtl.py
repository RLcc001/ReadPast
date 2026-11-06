"""Round 19 follow-up: answer latency vs timeline length (does the gate save time on long streams?). Report-only.

For each question (first 20 measured keys of kfs/data/bench_keys_r19.json, after 3 warm-up keys) the recent 4 frames
are decoded and encoded once (as in kfs/bench_latency.py); the ReadPast answer prompt is then built with a timeline of
L tokens for L in --lengths (0 = no timeline, i.e. the gate said "now") and generate_segments is timed (synchronized).
Timelines of a given length are real cached captions (cache/captions_*_s16) concatenated across videos with
consecutive synthetic 16-s time stamps, formatted exactly like eval_ovo_kfs.format_timeline, cut at a caption
boundary once the token count reaches L. Answers are meaningless for L > 0 (foreign text); only timing is used.
The gate (route.p_past on the question) is timed once per question. 1 h of stream ~ 225 captions ~ 17-19k tokens.

usage (via rx, exclusive GPU): python kfs/bench_longtl.py --model_path ... --captions cache/captions_q25_s16 \
    --out_dir $RX_RUN_DIR/out [--lengths 0,1000,2000,5000,10000,20000,40000] [--n_q 20]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_latency import RECENT, cpu_timed, timed  # noqa: E402
from eval_ovo_kfs import (  # noqa: E402
    ABST_INSTR,
    _build_sampling_plan,
    _decode_indices,
    _resize_like_fetch_video,
    build_qwen25_prompt,
    build_segments,
    format_timeline,
    is_abst_option,
    to_pil,
    with_timeline,
)
from route import ROUTE_PROMPTS, p_past  # noqa: E402

N_WARM = 3


def caption_pool(cap_dir: str) -> list[str]:
    texts = []
    for f in sorted(os.listdir(cap_dir)):
        if f.endswith(".json"):
            texts += [c["text"] for c in json.load(open(os.path.join(cap_dir, f)))["captions"]]
    return texts


def build_timeline(tok, texts: list[str], target: int) -> tuple[str, int]:
    caps, n = [], 0
    for i, t in enumerate(texts):
        caps.append({"t0": 16 * i, "t1": 16 * (i + 1), "text": t})
        if i % 20 == 19 or i == len(texts) - 1:
            n = len(tok.encode(format_timeline(caps), add_special_tokens=False))
            if n >= target:
                break
    # trim to the first caption boundary at or above target
    while len(caps) > 1 and len(tok.encode(format_timeline(caps[:-1]), add_special_tokens=False)) >= target:
        caps.pop()
    tl = format_timeline(caps)
    return tl, len(tok.encode(tl, add_special_tokens=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--chunked_dir", default="/workspace/datasets/streaming/ovo_bench/chunked_videos")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--keys", default="kfs/data/bench_keys_r19.json")
    ap.add_argument("--lengths", default="0,1000,2000,5000,10000,20000,40000")
    ap.add_argument("--n_q", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch
    from llava_ov import load_qa

    qa = load_qa(args.model_path, "cuda:0", 256)
    tok = qa.processor.tokenizer
    lengths = [int(x) for x in args.lengths.split(",")]
    texts = caption_pool(args.captions)
    tls = {L: build_timeline(tok, texts, L) if L else ("", 0) for L in lengths}
    print({L: n for L, (_, n) in tls.items()}, flush=True)
    annos = {f"{a['task']}:{a['id']}": a for a in json.load(open(args.anno_path))}
    keys = json.load(open(args.keys))
    keys = keys[5 - N_WARM: 5 + args.n_q]  # 3 of the round-19 warm-up keys, then the first n_q measured keys
    words = {w: sorted({tok.encode(s, add_special_tokens=False)[0] for s in (w, w.capitalize())}) for w in ("past", "now")}
    rows = []
    for i, key in enumerate(keys):
        a = annos[key]
        video_path = os.path.join(args.chunked_dir, f"{a['id']}.mp4")
        ele = {"video": video_path, "fps": 1.0}

        def dec():
            plan = _build_sampling_plan(ele, last_nframes=RECENT)
            fr = _decode_indices(video_path, plan.backend, plan.full_indices[-RECENT:])
            return list(_resize_like_fetch_video(fr, ele, plan.sampled_nframes_full, image_patch_size=14))

        frames, t_dec = cpu_timed(dec)
        blocks, t_enc, _ = timed(lambda: [qa.encode_vision([to_pil(f)]) for f in frames])
        segments = build_segments("ss", [], [], blocks, RECENT)
        question = build_qwen25_prompt(a["task"], a)
        if any(is_abst_option(o) for o in a["options"]):
            question = ABST_INSTR["dhsm"] + "\n\n" + question
        _, t_gate, _ = timed(p_past, qa, a["question"], words, ROUTE_PROMPTS[0], with_logodds=True)
        row = {"key": key, "warmup": i < N_WARM, "decode_s": t_dec, "encode_s": t_enc, "gate_s": t_gate, "by_len": {}}
        for L in lengths:
            tl, n = tls[L]
            segs = with_timeline(tl, f"Current view (the latest {RECENT} seconds):\n", segments, "before") if tl else segments
            out, t, m = timed(qa.generate_segments, segs, question, len(a["options"]))
            row["by_len"][str(L)] = {"tokens": n, "answer_s": t, "peak_gb": m,
                                     "out_tokens": len(tok.encode(out["answer"], add_special_tokens=False))}
        rows.append(row)
        print(f"{i} {key}: " + " ".join(f"{L}:{row['by_len'][str(L)]['answer_s']:.2f}s" for L in lengths), flush=True)
        torch.cuda.empty_cache()
    m = [r for r in rows if not r["warmup"]]
    summ = {str(L): {"tokens": tls[L][1],
                     "answer_median_s": statistics.median([r["by_len"][str(L)]["answer_s"] for r in m]),
                     "peak_gb_max": max(r["by_len"][str(L)]["peak_gb"] for r in m),
                     "out_tokens_median": statistics.median([r["by_len"][str(L)]["out_tokens"] for r in m])}
            for L in lengths}
    summ["gate_median_s"] = statistics.median([r["gate_s"] for r in m])
    json.dump({"model": args.model_path, "gpu": torch.cuda.get_device_name(0), "lengths": lengths, "rows": rows,
               "summary": summ}, open(os.path.join(args.out_dir, "bench_longtl.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1), flush=True)


if __name__ == "__main__":
    main()
