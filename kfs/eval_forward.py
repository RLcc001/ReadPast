"""Round 21: OVO Forward yes/no probes (SSR, CRR) with the first-token P(Yes), for label-free calibration of the
"respond now or keep waiting" decision.

Per probe (video chunk `{id}_{index}.mp4`, SimpleStream's prompt from build_ovo_prompt, unchanged):
  L0      recent `--recent` 1-fps frames (= SimpleStream; generated response is scored like SimpleStream)
  L1      caption timeline (clips ended by the probe time, --captions) + recent frames, ReadPast's L1 format
  L0gray  L0 with the frames replaced by mid-gray frames (per-probe "no visual evidence" baseline; report only)
Content-free probes (once per task and condition, written to cf.json):
  B  the question slot (CRR question / SSR step) replaced by "N/A", "", "[MASK]"; 4 mid-gray frames sized like the
     task's first probe; L1 adds the timeline header + "[0s-16s] {cf}"                        [primary]
  A  same content-free prompts without any frames                                              [report only]
p_yes = (P(Yes) + P(yes)) / (P(Yes) + P(yes) + P(No) + P(no)) at the first generated token.

usage: python kfs/eval_forward.py --model_path Qwen/Qwen3-VL-8B-Instruct --captions cache/captions_q3_s16_fw \\
           --result_dir $RX_RUN_DIR/out [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ovo_kfs import (  # noqa: E402
    TIMELINE_HEADER,
    _build_sampling_plan,
    _decode_indices,
    _resize_like_fetch_video,
    format_timeline,
    rxlib,
    timeline_captions,
    to_pil,
    with_timeline,
)

import torch  # noqa: E402
from lib.recent_window_eval import build_ovo_prompt  # noqa: E402
from ovo_constants import score_yesno  # noqa: E402

CAND = ["Yes", "yes", "No", "no"]
CF = ["N/A", "", "[MASK]"]


def p_yes(pc: list[float]) -> float:
    return (pc[0] + pc[1]) / max(sum(pc), 1e-12)


def cf_anno(anno: dict, cf: str) -> dict:
    a = copy.deepcopy(anno)
    if a["task"] == "CRR":
        a["question"] = cf
    else:
        a["test_info"] = [{**a["test_info"][0], "step": cf}]
    return a


def recent_frames(video_path: str, recent: int, fps: float):
    ele = {"video": video_path, "fps": fps}
    plan = _build_sampling_plan(ele, last_nframes=recent)
    full = plan.full_indices
    frames = _decode_indices(video_path, plan.backend, full[-recent:])
    frames = _resize_like_fetch_video(frames, ele, plan.sampled_nframes_full, image_patch_size=14)
    return frames, full[-1] / plan.raw_fps


def ask(qa, segments: list, question: str) -> dict:
    out = qa.generate_segments(segments, question, 1, cand=CAND)
    return {"response": out["answer"], "p_cand": out["p_cand"], "p_yes": p_yes(out["p_cand"]), "generate_time": out["generate_time"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--chunked_dir", default="/workspace/datasets/streaming/ovo_bench/chunked_videos")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--captions", default="", help="caption dir (kfs/caption_ovo.py --bench ovo_fw); empty = no L1")
    ap.add_argument("--tasks", default="SSR,CRR")
    ap.add_argument("--recent", type=int, default=4)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per task (smoke)")
    ap.add_argument("--ids", default="", help="comma-separated annotation ids to keep (smoke)")
    args = ap.parse_args()

    tasks = args.tasks.split(",")
    annos = [a for a in json.load(open(args.anno_path)) if a["task"] in tasks]
    if args.ids:
        keep = {int(i) for i in args.ids.split(",")}
        annos = [a for a in annos if a["id"] in keep]
    if args.limit:
        annos = [a for t in tasks for a in [x for x in annos if x["task"] == t][: args.limit]]
    conds = ["L0"] + (["L1"] if args.captions else []) + ["L0gray"]
    print(json.dumps(vars(args)), {t: sum(a["task"] == t for a in annos) for t in tasks},
          {t: sum(len(a["test_info"]) for a in annos if a["task"] == t) for t in tasks}, conds, flush=True)

    from llava_ov import load_qa

    qa = load_qa(args.model_path, "cuda:0", args.max_new_tokens)
    os.makedirs(args.result_dir, exist_ok=True)
    ckpt = os.path.join(args.result_dir, "probes.jsonl")
    done = {json.loads(l)["_key"] for l in open(ckpt)} if os.path.exists(ckpt) else set()

    # content-free probes (B with gray frames sized like the task's first probe, A without frames)
    cf_path = os.path.join(args.result_dir, "cf.json")
    if not os.path.exists(cf_path):
        cf: dict = {}
        for t in tasks:
            first = next(a for a in annos if a["task"] == t)
            frames, _ = recent_frames(os.path.join(args.chunked_dir, f"{first['id']}_0.mp4"), args.recent, args.fps)
            gray = [qa.encode_vision([to_pil(f)]) for f in torch.full_like(frames, 128.0)]
            for cond in [c for c in conds if c != "L0gray"]:
                for var in ("B", "A"):
                    vals = {}
                    for s in CF:
                        q = build_ovo_prompt(t, cf_anno(first, s), index=0)
                        segs = [gray] if var == "B" else []
                        if cond == "L1":
                            tl = TIMELINE_HEADER + f"[0s-16s] {s}\n"
                            segs = with_timeline(tl, f"Current view (the latest {args.recent} seconds):\n", segs, "before") \
                                if var == "B" else [tl]
                        vals[s] = ask(qa, segs, q)
                    cf.setdefault(t, {}).setdefault(cond, {})[var] = {
                        "p_cf": sum(v["p_yes"] for v in vals.values()) / len(vals),
                        "per_input": {s: {"p_yes": v["p_yes"], "response": v["response"]} for s, v in vals.items()}}
                    print(t, cond, var, round(cf[t][cond][var]["p_cf"], 4), flush=True)
        json.dump(cf, open(cf_path, "w"), indent=1)

    t_start, n = time.perf_counter(), 0
    with open(ckpt, "a") as fh:
        for ai, anno in enumerate(annos):
            for index, info in enumerate(anno["test_info"]):
                keys = {c: f"{anno['task']}:{anno['id']}:{index}:{c}" for c in conds}
                if all(k in done for k in keys.values()):
                    continue
                video_path = os.path.join(args.chunked_dir, f"{anno['id']}_{index}.mp4")
                if not os.path.exists(video_path):
                    print("missing", video_path, flush=True)
                    continue
                frames, q_ts = recent_frames(video_path, args.recent, args.fps)
                blocks = [qa.encode_vision([to_pil(f)]) for f in frames]
                question = build_ovo_prompt(anno["task"], anno, index=index)
                for cond, key in keys.items():
                    if key in done:
                        continue
                    tl_tokens = 0
                    if cond == "L0":
                        segs = [blocks]
                    elif cond == "L0gray":
                        segs = [[qa.encode_vision([to_pil(f)]) for f in torch.full_like(frames, 128.0)]]
                    else:
                        timeline = format_timeline(timeline_captions(args.captions, anno["video"], q_ts))
                        segs = with_timeline(timeline, f"Current view (the latest {args.recent} seconds):\n", [blocks],
                                             "before") if timeline else [blocks]
                        tl_tokens = len(qa.processor.tokenizer.encode(timeline, add_special_tokens=False)) if timeline else 0
                    r = ask(qa, segs, question)
                    rec = {"_key": key, "task": anno["task"], "id": anno["id"], "video": anno["video"], "index": index,
                           "cond": cond, "realtime": info["realtime"], "type": info["type"], "q_ts": round(q_ts, 2),
                           "clue_time": anno.get("clue_time"), "timeline_tokens": tl_tokens,
                           "score": score_yesno(r["response"], info["type"]), **r}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    n += 1
            if (ai + 1) % 5 == 0:
                print(f"[{ai + 1}/{len(annos)}] {n} probes, {(time.perf_counter() - t_start) / max(n, 1):.2f}s/probe", flush=True)

    rows = [json.loads(l) for l in open(ckpt)]
    metrics = {}
    for t in tasks:
        for c in conds:
            sc = [r["score"] for r in rows if r["task"] == t and r["cond"] == c]
            if sc:
                metrics[f"{t}_{c}_raw"] = 100 * sum(sc) / len(sc)
    print(json.dumps({k: round(v, 2) for k, v in metrics.items()}), flush=True)
    rxlib.write_result(metrics, selection="final", n_eval=1, protocol="ovo-forward-yesno")


if __name__ == "__main__":
    main()
