"""Surprisal of each caption under the memory written before it (innovation 2: surprise-guided retention).

For every caption file from kfs/caption_ovo.py, the informative captions ("No change." skipped, as in the
timeline) are laid out exactly like the evaluation timeline and scored in one causal text-only forward pass
of the same VLM. A caption's surprisal is the mean negative log-likelihood of its text tokens given the
header and all earlier captions, so it can be computed when the caption is written (streaming-causal).
Low surprisal = the caption was predictable from the existing memory (e.g. copied or repeated content).

Output per video: {out_dir}/{video with '/' -> '__'}.json  {"video", "nll": {"<t0>": float, ...}}

usage: python kfs/surprisal.py --captions cache/captions_q25_s16_delta [--shard 0 --nshard 2]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ovo_kfs import TIMELINE_HEADER, KFSModel, is_no_change, timeline_line  # noqa: E402

import torch  # noqa: E402


@torch.inference_mode()
def score_video(qa: KFSModel, caps: list[dict]) -> dict[str, float]:
    tok = qa.processor.tokenizer
    ids = tok.encode(TIMELINE_HEADER, add_special_tokens=False)
    spans = []  # (t0, start, end) of each caption's text tokens
    for c in caps:
        line = timeline_line(c)
        prefix, text = line.split("] ", 1)
        ids += tok.encode(prefix + "] ", add_special_tokens=False)
        t = tok.encode(text, add_special_tokens=False)
        spans.append((c["t0"], len(ids), len(ids) + len(t)))
        ids += t
    input_ids = torch.tensor([ids], device=qa.model.device)
    logits = qa.model(input_ids=input_ids, use_cache=False).logits[0].float()
    nll = torch.nn.functional.cross_entropy(logits[:-1], input_ids[0, 1:], reduction="none")
    # token j is predicted by position j-1
    return {str(t0): round(float(nll[s - 1:e - 1].mean()), 4) for t0, s, e in spans if e > s}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--out_dir", default=None, help="default {captions}_surprisal")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    args = ap.parse_args()
    out_dir = args.out_dir or args.captions.rstrip("/") + "_surprisal"
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.captions, "*.json")))[args.shard::args.nshard]
    print(json.dumps(vars(args)), f"{len(files)} caption files", flush=True)

    qa = KFSModel(model_name=args.model_path, device="cuda:0", max_new_tokens=1)
    t_start = time.perf_counter()
    for i, f in enumerate(files):
        out = os.path.join(out_dir, os.path.basename(f))
        if os.path.exists(out):
            continue
        rec = json.load(open(f))
        caps = [c for c in rec["captions"] if not is_no_change(c["text"])]
        nll = score_video(qa, caps) if caps else {}
        tmp = f"{out}.{os.getpid()}.tmp"
        json.dump({"video": rec["video"], "nll": nll}, open(tmp, "w"))
        os.replace(tmp, out)
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(files)}] {time.perf_counter() - t_start:.0f}s", flush=True)
    print(f"done: {len(files)} files in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
