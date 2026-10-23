"""Precompute CLIP image features of every 1-fps prefix frame for the OVO backward + real-time questions.

Fills the cache read by eval_ovo_kfs.py (one .npz per question). Decoding dominates, so the
questions are split over `--nproc` worker processes that each hold their own CLIP copy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_ovo_kfs import BACKWARD_TASKS, REAL_TIME_TASKS, ClipScorer, _build_sampling_plan  # noqa: E402


def work(rank: int, args, annos: list[dict]) -> None:
    scorer = ClipScorer(args.clip_path, "cuda:0", args.cache_dir)
    mine = annos[rank :: args.nproc]
    t0 = time.time()
    for i, a in enumerate(mine):
        path = os.path.join(args.chunked_dir, f"{a['id']}.mp4")
        full = _build_sampling_plan({"video": path, "fps": 1.0}, last_nframes=1).full_indices
        scorer.image_feats(f"{a['task']}_{a['id']}", path, full)
        if i % 20 == 0:
            print(f"[rank {rank}] {i}/{len(mine)} {time.time() - t0:.0f}s", flush=True)
    print(f"[rank {rank}] done {len(mine)} in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--chunked_dir", default="/workspace/datasets/streaming/ovo_bench/chunked_videos")
    ap.add_argument("--clip_path", default="/workspace/models/clip-vit-large-patch14-336-st")
    ap.add_argument("--cache_dir", default="/workspace/streamkv/cache/clip_l336_ovo")
    ap.add_argument("--nproc", type=int, default=12)
    args = ap.parse_args()
    annos = [a for a in json.load(open(args.anno_path)) if a["task"] in BACKWARD_TASKS + REAL_TIME_TASKS]
    # longest prefixes first so the slowest questions do not end up as stragglers
    annos.sort(key=lambda a: -float(a["realtime"]))
    print(f"{len(annos)} questions, {args.nproc} procs", flush=True)
    mp.spawn(work, args=(args, annos), nprocs=args.nproc, join=True)


if __name__ == "__main__":
    main()
