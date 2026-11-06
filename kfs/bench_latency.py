"""Round 19 (EXPERIMENTS.md): latency / memory of SimpleStream and ReadPast on one exclusive GPU. Report-only.

Two stages, one model load (Qwen2.5-VL / Qwen3-VL through llava_ov.load_qa, same classes as the evals):

  write   ReadPast's timeline writer, exactly kfs/caption_ovo.py (plain mode, 16 s clips, 8 frames, 256*28*28 px,
          max_new_tokens 96): per clip decode (get_batch + resize), vision encode (8 frames), caption generation.
          Clips: 10 OVO videos spread over caption-count deciles, the first 20 clips of each (fewer if shorter).
          Check: every generated caption must equal the cached one (cache/captions_*_s16) -> same code path.
  answer  per question, exactly kfs/eval_ovo_kfs.py evaluate_one with k=0, layout ss, max_new_tokens 256:
          decode the recent 4 frames (SimpleStream exact-recent decoder) | encode them |
          SimpleStream answer (no timeline, no instruction)                          = r4 run record
          gate: route.p_past on the question only (prompt 0), read iff p_past >= p_cf (calib.json of --route_dir)
          ReadPast answer: timeline if read; D-HSM instruction iff an option is an abstention option
                                                                                      = round-17 composition
          Questions: kfs/data/bench_keys_r19.json (5 warm-up + 100 = BT 50 + RT 50, task-stratified, seed 19).
Every timed call is bracketed by torch.cuda.synchronize(); peak memory = max_memory_allocated, reset per call.
The first 5 clips / questions are warm-up and excluded from the summary (they are still checked).

usage (via rx, exclusive GPU):  python kfs/bench_latency.py --model_path ... --captions cache/captions_q25_s16 \
    --route_dir <rx run>/out --out_dir $RX_RUN_DIR/out
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from caption_ovo import PROMPT  # noqa: E402
from eval_ovo_kfs import (  # noqa: E402
    ABST_INSTR,
    DECODE_THREADS,
    _build_sampling_plan,
    _decode_indices,
    _resize_like_fetch_video,
    build_qwen25_prompt,
    build_segments,
    format_timeline,
    is_abst_option,
    timeline_captions,
    to_pil,
    with_timeline,
)
from route import ROUTE_PROMPTS, p_past  # noqa: E402

import decord  # noqa: E402
import torch  # noqa: E402
from qwen_vl_utils.vision_process import smart_resize  # noqa: E402
from torchvision.transforms import InterpolationMode  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

N_WARM = 5
SEG, NFRAMES, CAP_PIXELS, CAP_TOKENS, ANS_TOKENS, RECENT = 16, 8, 256 * 28 * 28, 96, 256, 4


def timed(fn, *a, **k):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    out = fn(*a, **k)
    torch.cuda.synchronize()
    return out, time.perf_counter() - t, torch.cuda.max_memory_allocated() / 2**30


def cpu_timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


def pick_write_clips(cap_dir: str, n_videos: int = 10, per_video: int = 20) -> list[dict]:
    """Videos spread over caption-count deciles (deterministic), plus one extra video for warm-up."""
    recs = []
    for f in sorted(os.listdir(cap_dir)):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(cap_dir, f)))
            if r["captions"]:
                recs.append(r)
    recs.sort(key=lambda r: (len(r["captions"]), r["video"]))
    idx = [round((i + 0.5) * (len(recs) - 1) / n_videos) for i in range(n_videos)]
    warm = recs[len(recs) // 2 + 1]
    clips = [dict(video=warm["video"], chunk=warm["chunk"], **c) for c in warm["captions"][:N_WARM]]
    for i in idx:
        r = recs[i]
        clips += [dict(video=r["video"], chunk=r["chunk"], **c) for c in r["captions"][:per_video]]
    return clips


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    p90 = xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]
    return {"median": statistics.median(xs), "p90": p90, "mean": statistics.mean(xs), "n": len(xs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--chunked_dir", default="/workspace/datasets/streaming/ovo_bench/chunked_videos")
    ap.add_argument("--captions", required=True)
    ap.add_argument("--route_dir", required=True, help="out dir of the round-17 route.py run (route.json, calib.json)")
    ap.add_argument("--keys", default="kfs/data/bench_keys_r19.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stages", default="write,answer")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N clips and first N questions")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from llava_ov import load_qa

    qa = load_qa(args.model_path, "cuda:0", ANS_TOKENS)
    tok = qa.processor.tokenizer
    torch.cuda.synchronize()
    weights_gb = torch.cuda.memory_allocated() / 2**30
    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "decode_threads": DECODE_THREADS,
           "omp_threads": os.environ.get("OMP_NUM_THREADS"), "weights_gb": weights_gb, "model": args.model_path}
    print(json.dumps(env), flush=True)
    out = {"env": env, "config": vars(args)}

    if "write" in args.stages:
        qa.max_new_tokens = CAP_TOKENS
        rows, readers = [], {}
        clips = pick_write_clips(args.captions)
        for i, c in enumerate(clips[: args.limit] if args.limit else clips):
            if c["chunk"] not in readers:
                (vr, t_open) = cpu_timed(decord.VideoReader, c["chunk"], num_threads=DECODE_THREADS)
                readers = {c["chunk"]: (vr, t_open)}
            vr, t_open = readers[c["chunk"]]
            fps = float(vr.get_avg_fps())
            h, w = vr[0].shape[:2]
            hh, ww = smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=CAP_PIXELS)
            t0, t1 = c["t0"], c["t1"]
            ts = [t0 + (j + 0.5) * SEG / NFRAMES for j in range(NFRAMES)]
            idx = [min(int(t * fps), len(vr) - 1) for t in ts]

            def dec():
                fr = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2).float()
                return TF.resize(fr, [hh, ww], interpolation=InterpolationMode.BICUBIC, antialias=True)

            frames, t_dec = cpu_timed(dec)
            block, t_enc, m_enc = timed(lambda: [qa.encode_vision([to_pil(f)]) for f in frames])
            r, t_gen, m_gen = timed(qa.generate_segments, [block], PROMPT.format(n=NFRAMES, seg=SEG, t0=t0, t1=t1), 1)
            rows.append({"video": c["video"], "t0": t0, "t1": t1, "warmup": i < N_WARM, "decode_s": t_dec,
                         "encode_s": t_enc, "generate_s": t_gen, "total_s": t_dec + t_enc + t_gen, "open_s": t_open,
                         "peak_gb": max(m_enc, m_gen), "vision_tokens": r["num_vision_tokens"],
                         "out_tokens": len(tok.encode(r["answer"], add_special_tokens=False)),
                         "same_as_cache": r["answer"] == c["text"], "caption": r["answer"]})
            print(f"write {i}: {rows[-1]['total_s']:.2f}s same={rows[-1]['same_as_cache']}", flush=True)
        m = [x for x in rows if not x["warmup"]]
        out["write"] = {"rows": rows, "summary": {
            **{k: stats([x[k] for x in m]) for k in ("decode_s", "encode_s", "generate_s", "total_s", "out_tokens", "peak_gb")},
            "rtf": stats([x["total_s"] / SEG for x in m]), "same_as_cache": sum(x["same_as_cache"] for x in rows),
            "n_rows": len(rows)}}
        json.dump(out, open(os.path.join(args.out_dir, "bench.json"), "w"), indent=1)

    if "answer" in args.stages:
        qa.max_new_tokens = ANS_TOKENS
        annos = {f"{a['task']}:{a['id']}": a for a in json.load(open(args.anno_path))}
        keys = json.load(open(args.keys))
        keys = keys[: args.limit] if args.limit else keys
        calib = json.load(open(os.path.join(args.route_dir, "calib.json")))
        stored = json.load(open(os.path.join(args.route_dir, "route.json")))
        p_cf = calib["p_cf"]
        words = {w: sorted({tok.encode(s, add_special_tokens=False)[0] for s in (w, w.capitalize())}) for w in ("past", "now")}
        rows = []
        for i, key in enumerate(keys):
            a = annos[key]
            video_path = os.path.join(args.chunked_dir, f"{a['id']}.mp4")
            ele = {"video": video_path, "fps": 1.0}

            def dec():
                plan = _build_sampling_plan(ele, last_nframes=RECENT)
                full = plan.full_indices
                fr = _decode_indices(video_path, plan.backend, full[-RECENT:])
                return plan, full, list(_resize_like_fetch_video(fr, ele, plan.sampled_nframes_full, image_patch_size=14))

            (plan, full, frames), t_dec = cpu_timed(dec)
            blocks, t_enc, m_enc = timed(lambda: [qa.encode_vision([to_pil(f)]) for f in frames])
            segments = build_segments("ss", [], [], blocks, RECENT)
            question = build_qwen25_prompt(a["task"], a)
            n = len(a["options"])
            ss, t_ss, m_ss = timed(qa.generate_segments, segments, question, n)

            (pp, mass, lo), t_gate, m_gate = timed(p_past, qa, a["question"], words, ROUTE_PROMPTS[0], with_logodds=True)
            read = round(pp, 5) >= p_cf

            def assemble():
                caps = timeline_captions(args.captions, a["video"], full[-1] / plan.raw_fps) if read else []
                tl = format_timeline(caps)
                segs = with_timeline(tl, f"Current view (the latest {RECENT} seconds):\n", segments, "before") if tl else segments
                q = question
                if any(is_abst_option(o) for o in a["options"]):
                    q = ABST_INSTR["dhsm"] + "\n\n" + q
                return tl, segs, q

            (tl, segs, q2), t_asm = cpu_timed(assemble)
            rp, t_rp, m_rp = timed(qa.generate_segments, segs, q2, n)
            q_tok = len(tok.encode(question, add_special_tokens=False))
            rows.append({
                "key": key, "task": a["task"], "warmup": i < N_WARM, "query_ts": round(full[-1] / plan.raw_fps, 2),
                "decode_s": t_dec, "encode_s": t_enc, "ss_answer_s": t_ss, "gate_s": t_gate, "assemble_s": t_asm,
                "rp_answer_s": t_rp,
                "ss_total_s": t_dec + t_enc + t_ss, "rp_total_s": t_dec + t_enc + t_gate + t_asm + t_rp,
                "peak_gb_ss": max(m_enc, m_ss), "peak_gb_rp": max(m_enc, m_gate, m_rp),
                "vision_tokens": ss["num_vision_tokens"], "question_tokens": q_tok,
                "timeline_tokens": len(tok.encode(tl, add_special_tokens=False)) if tl else 0,
                "p_past": round(pp, 5), "p_past_stored": stored[key]["p_past"], "read": read,
                "instr": q2 != question,
                "ss_response": ss["answer"], "ss_p_opts": ss["p_opts"], "rp_response": rp["answer"], "rp_p_opts": rp["p_opts"],
                "ss_out_tokens": len(tok.encode(ss["answer"], add_special_tokens=False)),
                "rp_out_tokens": len(tok.encode(rp["answer"], add_special_tokens=False)),
            })
            print(f"answer {i} {key}: ss {rows[-1]['ss_total_s']:.2f}s rp {rows[-1]['rp_total_s']:.2f}s read={read}", flush=True)
        m = [x for x in rows if not x["warmup"]]
        cols = ("decode_s", "encode_s", "ss_answer_s", "gate_s", "assemble_s", "rp_answer_s", "ss_total_s", "rp_total_s",
                "peak_gb_ss", "peak_gb_rp", "timeline_tokens")
        summ = {k: stats([x[k] for x in m]) for k in cols}
        summ["rp_extra_s"] = stats([x["rp_total_s"] - x["ss_total_s"] for x in m])
        summ["rp_answer_read_s"] = stats([x["rp_answer_s"] for x in m if x["read"]])
        summ["rp_answer_noread_s"] = stats([x["rp_answer_s"] for x in m if not x["read"]])
        summ["read_rate"] = sum(x["read"] for x in m) / len(m)
        summ["gate_matches_stored"] = sum(x["p_past"] == x["p_past_stored"] for x in rows)
        out["answer"] = {"rows": rows, "summary": summ, "p_cf": p_cf}
        json.dump(out, open(os.path.join(args.out_dir, "bench.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
