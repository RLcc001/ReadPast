"""Temporal-scope router (module 1): does a question need memory of the earlier stream, or only the current view?

The same VLM reads only the question text (no options, no frames: the "Unable to answer" option would leak HLD) and
the next-token distribution over "past" / "now" gives p_past. One fixed prompt, no tuning.

Contextual calibration (Zhao et al., 2021): p_past of content-free questions ("N/A", "", "[MASK]"), averaged, is
written to {out_dir}/calib.json; the calibrated rule "past iff p_past >= p_cf" needs no threshold tuning.

Output: {out_dir}/route.json  {"<task>:<id>" (OVO) or "<question_id>" (--bench sb): {"p_past": float, "mass": float}}

usage: python kfs/route.py --out_dir $RX_RUN_DIR/out
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ovo_kfs import BACKWARD_TASKS, REAL_TIME_TASKS, KFSModel  # noqa: E402

import torch  # noqa: E402

ROUTE_PROMPTS = [
    (
        "You are watching a live video stream. At the current moment someone asks a question about it.\n"
        "Question: {question}\n"
        "To answer it, do you need to remember what happened earlier in the stream, or is what is visible right now "
        "enough? Reply with one word: past or now."
    ),
    # paraphrases for the prompt-sensitivity check
    (
        "Question about a live video: {question}\n"
        "Can this question be answered from the current frame alone, or does it require information from earlier in "
        "the video? Answer with one word: now or past."
    ),
    (
        "Here is a question asked during a live video stream: {question}\n"
        "Does it ask about something that happened before the current moment? Reply with one word: past or now."
    ),
]
ROUTE_PROMPT = ROUTE_PROMPTS[0]


@torch.inference_mode()
def p_past(qa: KFSModel, question: str, words: dict[str, list[int]], prompt: str = ROUTE_PROMPT) -> tuple[float, float]:
    tok = qa.processor.tokenizer
    ids = [qa._im_start_id] + tok.encode("user\n" + prompt.format(question=question.strip()), add_special_tokens=False)
    ids += [qa._im_end_id] + tok.encode("\n", add_special_tokens=False)
    ids += [qa._im_start_id] + tok.encode("assistant\n", add_special_tokens=False)
    logits = qa.model(input_ids=torch.tensor([ids], device=qa.model.device), use_cache=False).logits[0, -1].float()
    p = torch.softmax(logits, dim=-1)
    past = float(sum(p[i] for i in words["past"]))
    now = float(sum(p[i] for i in words["now"]))
    return past / max(past + now, 1e-12), past + now


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--bench", choices=["ovo", "sb"], default="ovo")
    ap.add_argument("--sb_anno", default="/workspace/datasets/streaming/streamingbench/questions_real.json")
    ap.add_argument("--prompt", type=int, default=0, help="index into ROUTE_PROMPTS")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    qa = KFSModel(model_name=args.model_path, device="cuda:0", max_new_tokens=1)
    tok = qa.processor.tokenizer
    # first token of each spelling; all distinct single tokens for Qwen2.5
    words = {w: sorted({tok.encode(s, add_special_tokens=False)[0] for s in (w, w.capitalize())}) for w in ("past", "now")}
    print(json.dumps(vars(args)), words, flush=True)

    if args.bench == "sb":
        qs = [(q["question_id"], q["question"]) for e in json.load(open(args.sb_anno)) for q in e["questions"]]
    else:
        qs = [(f"{a['task']}:{a['id']}", a["question"]) for a in json.load(open(args.anno_path))
              if a["task"] in BACKWARD_TASKS + REAL_TIME_TASKS]
    out, t0 = {}, time.perf_counter()
    for key, question in qs:
        pp, mass = p_past(qa, question, words, ROUTE_PROMPTS[args.prompt])
        out[key] = {"p_past": round(pp, 5), "mass": round(mass, 5)}
    json.dump(out, open(os.path.join(args.out_dir, "route.json"), "w"), indent=0)
    cf = {q: p_past(qa, q, words, ROUTE_PROMPTS[args.prompt])[0] for q in ("N/A", "", "[MASK]")}
    json.dump({"content_free": cf, "p_cf": sum(cf.values()) / len(cf)}, open(os.path.join(args.out_dir, "calib.json"), "w"), indent=1)
    print("calibration", cf, flush=True)
    print(f"done: {len(out)} questions in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
