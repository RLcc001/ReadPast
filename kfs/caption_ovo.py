"""Query-agnostic text memory for OVO-Bench: caption every `seg`-second clip of each source video.

Each source video is captioned once from its longest chunk (the question with the latest
`realtime`), so a question at time T may use every caption whose clip ends at or before T.
Only full clips are captioned. Output per video: {out_dir}/{video with '/' -> '__'}.json
  {"video", "chunk", "fps", "duration", "seg", "captions": [{"t0", "t1", "text"}, ...]}

usage (one shard per rx job):
  python kfs/caption_ovo.py --seg 16 --shard 0 --nshard 8
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
from eval_ovo_kfs import KFSModel, to_pil  # noqa: E402

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


def video_key(video: str) -> str:
    return video.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--chunked_dir", default="/workspace/datasets/streaming/ovo_bench/chunked_videos")
    ap.add_argument("--out_dir", default=None, help="default /workspace/streamkv/cache/captions_q25_s{seg}")
    ap.add_argument("--seg", type=int, default=16, help="clip length in seconds")
    ap.add_argument("--nframes", type=int, default=8, help="frames per clip, evenly spaced")
    ap.add_argument("--max_pixels", type=int, default=256 * 28 * 28, help="per-frame pixel budget (~256 visual tokens)")
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N videos of this shard (smoke tests)")
    args = ap.parse_args()
    out_dir = args.out_dir or f"/workspace/streamkv/cache/captions_q25_s{args.seg}"
    os.makedirs(out_dir, exist_ok=True)

    annos = [a for a in json.load(open(args.anno_path)) if a["task"] in BACKWARD_TASKS + REAL_TIME_TASKS]
    longest: dict[str, dict] = {}
    for a in annos:
        if a["video"] not in longest or a["realtime"] > longest[a["video"]]["realtime"]:
            longest[a["video"]] = a
    # longest first, round-robin over shards so shards get similar total duration
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
        chunk = os.path.join(args.chunked_dir, f"{longest[video]['id']}.mp4")
        vr = decord.VideoReader(chunk)
        fps = float(vr.get_avg_fps())
        duration = len(vr) / fps
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
            r = qa.generate_segments([block], PROMPT.format(n=args.nframes, seg=args.seg, t0=t0, t1=t1), 1)
            caps.append({"t0": t0, "t1": t1, "text": r["answer"]})
            stats["clips"] += 1
            stats["gen_time"] += r["generate_time"]
        rec = {"video": video, "chunk": chunk, "fps": fps, "duration": duration, "seg": args.seg, "captions": caps}
        tmp = f"{out}.{os.getpid()}.tmp"
        json.dump(rec, open(tmp, "w"), ensure_ascii=False, indent=0)
        os.replace(tmp, out)
        el = time.perf_counter() - t_start
        print(f"[{vi + 1}/{len(videos)}] {video} {duration:.0f}s {len(caps)} clips | total {int(stats['clips'])} clips "
              f"{el / max(stats['clips'], 1):.2f}s/clip", flush=True)
    print(f"done: {int(stats['clips'])} clips in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
