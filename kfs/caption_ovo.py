"""Query-agnostic text memory for OVO-Bench: caption every `seg`-second clip of each source video.

Each source video is captioned once from its longest chunk (the question with the latest
`realtime`), so a question at time T may use every caption whose clip ends at or before T.
Only full clips are captioned. Output per video: {out_dir}/{video with '/' -> '__'}.json
With --bench sb, the StreamingBench Real-Time videos are captioned up to their latest question time stamp.
  {"video", "chunk", "fps", "duration", "seg", "captions": [{"t0", "t1", "text"}, ...]}

Modes:
  plain : each clip is described on its own
  memo  : like plain, but the captioner writes a memory note aimed at later questions (hand-object interactions and
          where objects end up, object states, counts, other people, readable text), at most three sentences
  delta : the first clip is described in full; later clips get the latest `ctx` informative captions as text
          and describe only what changed ("No change." when nothing did; the timeline skips those)

usage (one shard per rx job):
  python kfs/caption_ovo.py --seg 16 --shard 0 --nshard 8 [--mode delta]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ovo_kfs import KFSModel, is_no_change, to_pil  # noqa: E402

import decord  # noqa: E402
import torch  # noqa: E402
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS  # noqa: E402
from qwen_vl_utils.vision_process import smart_resize  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

PROMPT = (
    "These {n} frames are sampled in order from a {seg}-second clip of a video ({t0}s to {t1}s). "
    "In one or two sentences, describe what happens in the clip: the people, the objects that are handled "
    "or clearly visible, the actions, the location, and any readable text. Be concrete and do not speculate."
)
MEMO_PROMPT = (
    "These {n} frames are sampled in order from a {seg}-second clip of a video ({t0}s to {t1}s). Write a short memory "
    "note about this clip that would help answer questions about it later: what the person does with their hands, "
    "which objects are picked up, put down or moved and where they end up, the states of things (on or off, open or "
    "closed, full or empty), how many there are of something, who else appears and what they do, and any readable "
    "text. Mention colors when they help identify objects. Use at most three sentences, be concrete and do not speculate."
)
# delta mode: the latest `ctx` informative captions are shown as text before the frames
DELTA_CONTEXT = "Earlier descriptions of this video:\n{lines}"
DELTA_PROMPT = (
    "These {n} frames are sampled in order from the next {seg}-second clip of the video ({t0}s to {t1}s). "
    "Describe only what is new or has changed compared with the earlier descriptions: objects that are picked up, "
    "put down or moved (say from where to where), people or objects that appear or leave, new actions, a change of "
    "location, and newly readable text. Do not repeat what the earlier descriptions already say. Use one or two "
    "sentences, be concrete and do not speculate. If nothing notable has changed, answer exactly: No change."
)


def video_key(video: str) -> str:
    return video.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--chunked_dir", default="/workspace/datasets/streaming/ovo_bench/chunked_videos")
    ap.add_argument("--bench", choices=["ovo", "sb"], default="ovo")
    ap.add_argument("--sb_anno", default="/workspace/datasets/streaming/streamingbench/questions_real.json")
    ap.add_argument("--sb_video_dir", default="/workspace/datasets/streaming/streamingbench/videos")
    ap.add_argument("--out_dir", default=None, help="default /workspace/streamkv/cache/captions_q25_s{seg}[_delta][_sb]")
    ap.add_argument("--mode", choices=["plain", "delta", "memo"], default="plain")
    ap.add_argument("--max_realtime", type=float, default=0, help="only videos whose latest question is at most this (s)")
    ap.add_argument("--ctx", type=int, default=3, help="delta mode: earlier informative captions shown as context")
    ap.add_argument("--seg", type=int, default=16, help="clip length in seconds")
    ap.add_argument("--nframes", type=int, default=8, help="frames per clip, evenly spaced")
    ap.add_argument("--max_pixels", type=int, default=256 * 28 * 28, help="per-frame pixel budget (~256 visual tokens)")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N videos of this shard (smoke tests)")
    args = ap.parse_args()
    suffix = ("" if args.mode == "plain" else f"_{args.mode}") + ("_sb" if args.bench == "sb" else "")
    out_dir = args.out_dir or f"/workspace/streamkv/cache/captions_q25_s{args.seg}{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    longest: dict[str, dict] = {}  # video -> {"path", "realtime"}: the clip to caption and how far
    if args.bench == "sb":
        for e in json.load(open(args.sb_anno)):
            ts = max(sum(int(x) * 60 ** i for i, x in enumerate(reversed(q["time_stamp"].split(":")))) for q in e["questions"])
            longest[e["video_path"]] = {"path": os.path.join(args.sb_video_dir, e["video_path"]), "realtime": ts}
    else:
        for a in json.load(open(args.anno_path)):
            if a["task"] in BACKWARD_TASKS + REAL_TIME_TASKS and (a["video"] not in longest or a["realtime"] > longest[a["video"]]["realtime"]):
                longest[a["video"]] = {"path": os.path.join(args.chunked_dir, f"{a['id']}.mp4"), "realtime": a["realtime"]}
    # longest first, round-robin over shards so shards get similar total duration
    if args.max_realtime:
        longest = {v: x for v, x in longest.items() if x["realtime"] <= args.max_realtime}
    videos = sorted(longest, key=lambda v: -longest[v]["realtime"])[args.shard::args.nshard]
    if args.limit:
        videos = videos[: args.limit]
    print(json.dumps(vars(args)), f"{len(videos)} videos in shard", flush=True)

    qa = KFSModel(model_name=args.model_path, device="cuda:0", max_new_tokens=args.max_new_tokens)
    stats = defaultdict(float)
    t_start = time.perf_counter()
    for vi, video in enumerate(videos):
        out = os.path.join(out_dir, f"{video_key(video)}.json")
        if os.path.exists(out):
            continue
        chunk = longest[video]["path"]
        vr = decord.VideoReader(chunk)
        fps = float(vr.get_avg_fps())
        duration = len(vr) / fps
        if args.bench == "sb":
            duration = min(duration, float(longest[video]["realtime"]))
        h, w = vr[0].shape[:2]
        hh, ww = smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=args.max_pixels)
        caps = []
        for t0 in range(0, int(duration) - args.seg + 1, args.seg):
            t1 = t0 + args.seg
            ts = [t0 + (i + 0.5) * args.seg / args.nframes for i in range(args.nframes)]
            idx = [min(int(t * fps), len(vr) - 1) for t in ts]
            frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2).float()
            frames = TF.resize(frames, [hh, ww], interpolation=InterpolationMode.BICUBIC, antialias=True)
            block = [qa.encode_vision([to_pil(f)]) for f in frames]
            prev = [c for c in caps if not is_no_change(c["text"])][-args.ctx :]
            if args.mode == "delta" and prev:
                lines = "".join(f"[{c['t0']}s-{c['t1']}s] {c['text']}\n" for c in prev)
                segments = [DELTA_CONTEXT.format(lines=lines), block]
                prompt = DELTA_PROMPT.format(n=args.nframes, seg=args.seg, t0=t0, t1=t1)
            else:
                template = MEMO_PROMPT if args.mode == "memo" else PROMPT
                segments, prompt = [block], template.format(n=args.nframes, seg=args.seg, t0=t0, t1=t1)
            r = qa.generate_segments(segments, prompt, 1)
            caps.append({"t0": t0, "t1": t1, "text": r["answer"]})
            stats["clips"] += 1
            stats["gen_time"] += r["generate_time"]
        rec = {"video": video, "chunk": chunk, "fps": fps, "duration": duration, "seg": args.seg, "mode": args.mode,
               "captions": caps}
        tmp = f"{out}.{os.getpid()}.tmp"
        json.dump(rec, open(tmp, "w"), ensure_ascii=False, indent=0)
        os.replace(tmp, out)
        el = time.perf_counter() - t_start
        print(f"[{vi + 1}/{len(videos)}] {video} {duration:.0f}s {len(caps)} clips | total {int(stats['clips'])} clips "
              f"{el / max(stats['clips'], 1):.2f}s/clip", flush=True)
    print(f"done: {int(stats['clips'])} clips in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
