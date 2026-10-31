"""Progressive memory access on StreamingBench Real-Time (Qwen2.5-VL), all levels in one pass.

For every question: L0 = SimpleStream recent window (same exact-recent decoder and prompt as
SimpleStream's eval_streamingbench.py, so L0 reproduces its r4 run); L1 = caption timeline + recent window
(only when captions exist before the question, otherwise L1 = L0); L2 = look-back note (kfs/eval_ovo_kfs.py),
run when module 1 sends the question to L1 (p_past >= route_th or L0 max p < tau0) and L1 max p < tau1.
Because L0 and L1 are recorded for every question, r4, cap16 (always L1), module 1 variants and module 1 + L2
are all derived offline from one run (kfs/analyze_sb.py). With --focus_budget N, a focused L1 (response_l1f: the
latest --focus_keep captions + BM25-relevant ones within N tokens, kfs/eval_ovo_kfs.py --retain relevance) is
also recorded whenever the timeline is longer than N; otherwise it equals L1.

usage: python kfs/eval_sb_kfs.py --result_dir OUT --captions cache/captions_q25_s16_sb --route ROUTE_JSON
       [--note_mode desc] [--shard i --nshard n]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
os.environ.setdefault("QWEN_EXACT_RECENT_DECODE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ovo_kfs import (  # noqa: E402
    NOTE_PROMPTS,
    retain_captions,
    SS_ROOT,
    ClipScorer,
    KFSModel,
    format_timeline,
    rxlib,
    timeline_captions,
)

sys.path.insert(0, os.path.join(SS_ROOT, "main_experiments"))
import decord  # noqa: E402
import numpy as np  # noqa: E402
from eval_streamingbench import build_prompt, make_key, timestamp_to_seconds  # noqa: E402
from lib.recent_window_eval import decode_video_to_chunks_qwen, extract_mcq_answer  # noqa: E402
from tqdm import tqdm  # noqa: E402


def recent_block(qa: KFSModel, video_path: str, end: float, n: int):
    """SimpleStream's recent window ending at `end` seconds: one (embeds, grid) per 1-s chunk."""
    chunks, _ = decode_video_to_chunks_qwen(video_path=video_path, chunk_duration=1.0, fps=1.0, recent_frames_only=n,
                                            video_start=max(0.0, end - n), video_end=end + 1e-4)
    return [qa.encode_vision(c.frames) for c in chunks][-n:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/streamingbench/questions_real.json")
    ap.add_argument("--video_dir", default="/workspace/datasets/streaming/streamingbench/videos")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--route", required=True, help="route.json from kfs/route.py --bench sb")
    ap.add_argument("--recent", type=int, default=4)
    ap.add_argument("--route_th", type=float, default=0.5)
    ap.add_argument("--tau0", type=float, default=0.5)
    ap.add_argument("--tau1", type=float, default=0.6)
    ap.add_argument("--lookback_k", type=int, default=4)
    ap.add_argument("--note_mode", choices=list(NOTE_PROMPTS), default="desc")
    ap.add_argument("--note_tokens", type=int, default=128)
    ap.add_argument("--focus_budget", type=int, default=500, help="0 disables the focused L1")
    ap.add_argument("--focus_keep", type=int, default=2)
    ap.add_argument("--clip_path", default="/workspace/models/clip-vit-large-patch14-336-st")
    ap.add_argument("--cache_dir", default="/workspace/streamkv/cache/clip_l336_sb")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="first N videos of this shard (smoke tests)")
    args = ap.parse_args()

    entries = json.load(open(args.anno_path))[args.shard::args.nshard]
    if args.limit:
        entries = entries[: args.limit]
    routes = json.load(open(args.route))
    print(json.dumps(vars(args)), f"{len(entries)} videos", flush=True)
    qa = KFSModel(model_name=args.model_path, device="cuda:0", max_new_tokens=args.max_new_tokens)
    qa.per_frame_blocks = False  # SimpleStream's StreamingBench script puts the window in one block for every backbone
    scorer = ClipScorer(args.clip_path, "cuda:0", args.cache_dir)

    os.makedirs(args.result_dir, exist_ok=True)
    ckpt = os.path.join(args.result_dir, "results_incremental.jsonl")
    done = {json.loads(line)["_key"] for line in open(ckpt)} if os.path.exists(ckpt) else set()
    frame_idx: dict[str, list[int]] = {}
    with open(ckpt, "a") as fh:
        for e in tqdm(entries, desc="videos", mininterval=30):
            video_path = os.path.join(args.video_dir, e["video_path"])
            questions = sorted(e["questions"], key=lambda q: timestamp_to_seconds(q["time_stamp"]))
            hist_sec = list(range(max(timestamp_to_seconds(q["time_stamp"]) for q in questions)))
            for q in questions:
                key = make_key(os.path.basename(video_path), q)
                if key in done:
                    continue
                ts = float(timestamp_to_seconds(q["time_stamp"]))
                prompt, n_opt = build_prompt(q), len(q["options"])
                gt = extract_mcq_answer(str(q["answer"])) or str(q["answer"]).strip().upper()
                rec = {"_key": key, "video": e["video_path"], "task_type": q["task_type"], "time_stamp": q["time_stamp"],
                       "temporal_clue_type": q.get("temporal_clue_type", ""), "question": q["question"], "answer_gt": gt,
                       "p_past": routes[q["question_id"]]["p_past"], "level": 0, "note": "", "lookback_ts": []}
                t0 = time.perf_counter()
                try:
                    block = recent_block(qa, video_path, ts, args.recent)
                except Exception as err:  # undecodable window: SimpleStream also records no answer (counted wrong)
                    print(f"{key}: {err!r:.200}", flush=True)
                    rec.update({f"{f}_{lv}" if lv else f: None for f in ("response", "p_opts") for lv in ("l0", "l1", "l1f", "")})
                    rec.update({f"correct_{lv}" if lv else "correct": False for lv in ("l0", "l1", "l1f", "")}, timeline_tokens=0)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    continue
                out = qa.generate_segments([block], prompt, n_opt)
                rec.update(response_l0=out["answer"], p_opts_l0=out["p_opts"])
                caps = timeline_captions(args.captions, e["video_path"], ts)
                timeline = format_timeline(caps)
                cur = f"Current view (the latest {args.recent} seconds):\n"
                if timeline:
                    out1 = qa.generate_segments([timeline + cur, block], prompt, n_opt)
                else:
                    out1 = out
                rec.update(response_l1=out1["answer"], p_opts_l1=out1["p_opts"],
                           timeline_tokens=len(qa.processor.tokenizer.encode(timeline, add_special_tokens=False)) if timeline else 0)
                out1f = out1
                if args.focus_budget and timeline:
                    focus = argparse.Namespace(budget=args.focus_budget, keep_recent=args.focus_keep, retain="relevance")
                    fcaps = retain_captions(focus, qa.processor.tokenizer, e["video_path"], caps, q["question"] + " " + " ".join(q["options"]))
                    if len(fcaps) < len(caps):
                        out1f = qa.generate_segments([format_timeline(fcaps) + cur, block], prompt, n_opt)
                rec.update(response_l1f=out1f["answer"], p_opts_l1f=out1f["p_opts"])
                to_l1 = bool(timeline) and (rec["p_past"] >= args.route_th or max(out["p_opts"]) < args.tau0)
                hist = [s for s in hist_sec if s < ts - args.recent]
                final = out1 if to_l1 else out
                if to_l1:
                    rec["level"] = 1
                if to_l1 and max(out1["p_opts"]) < args.tau1 and hist:
                    if video_path not in frame_idx:  # 1-fps history frames (mid-second) for CLIP, cached per video
                        vr = decord.VideoReader(video_path)
                        fps, n = vr.get_avg_fps(), len(vr)
                        frame_idx[video_path] = [min(int((s + 0.5) * fps), n - 1) for s in hist_sec]
                    feats = scorer.image_feats(e["video_path"].replace("/", "__"), video_path, frame_idx[video_path])
                    sims = feats[hist] @ scorer.text_feat(q["question"])
                    chosen = sorted(hist[i] for i in np.argsort(-sims)[: args.lookback_k])
                    segs: list = []
                    for s in chosen:
                        segs += [f"[{s}s] ", recent_block(qa, video_path, float(s), 1), "\n"]
                    max_new, qa.max_new_tokens = qa.max_new_tokens, args.note_tokens
                    note = qa.generate_segments(segs, NOTE_PROMPTS[args.note_mode].format(question=q["question"].strip()), n_opt)["answer"]
                    qa.max_new_tokens = max_new
                    stamp = ", ".join(f"{s}s" for s in chosen)
                    look = f"A closer look at frames retrieved from earlier ({stamp}): {note.strip()}\n"
                    final = qa.generate_segments([timeline + look + cur, block], prompt, n_opt)
                    rec.update(level=2, note=note, lookback_ts=chosen, response_l2=final["answer"], p_opts_l2=final["p_opts"])
                rec.update(response=final["answer"], p_opts=final["p_opts"], time=time.perf_counter() - t0)
                for lv in ("l0", "l1", "l1f", ""):
                    r = rec.get(f"response_{lv}" if lv else "response")
                    rec[f"correct_{lv}" if lv else "correct"] = extract_mcq_answer(r) == gt
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

    rows = [json.loads(line) for line in open(ckpt)]
    metrics = {f"acc_{lv}": 100 * float(np.mean([r[f"correct_{lv}"] for r in rows])) for lv in ("l0", "l1")}
    metrics["acc_cascade"] = 100 * float(np.mean([r["correct"] for r in rows]))
    metrics.update({f"l{i}_rate": float(np.mean([r["level"] == i for r in rows])) for i in (1, 2)})
    metrics["n"] = len(rows)
    print(json.dumps(metrics, indent=1), flush=True)
    rxlib.write_result(metrics, selection="final", n_eval=1, protocol="sb-rt")


if __name__ == "__main__":
    main()
