"""Answer priors without visual evidence (round 12): the option-letter distribution of the same VLM when it reads only the
OVO multiple-choice prompt (exact SimpleStream question + options + instruction), with no frames and no timeline.

Two text-only conditions per question:
  q : the real question and options          -> question-only prior p_Q (what the text alone suggests)
  na: question replaced by "N/A", same options -> content-free prior p_N (Zhao et al., 2021), report only

Output: {out_dir}/prior.json  {"<task>:<id>": {"lp_q": [log p over letters], "lp_na": [...], "mass_q": float}}

usage: python kfs/answer_prior.py --out_dir $RX_RUN_DIR/out [--model_path Qwen/Qwen3-VL-8B-Instruct] [--inject_abst kfs/data/ovo_inject_abst_v1.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ovo_kfs import BACKWARD_TASKS, REAL_TIME_TASKS, KFSModel, apply_inject, build_qwen25_prompt  # noqa: E402

import torch  # noqa: E402


@torch.inference_mode()
def letter_logprobs(qa: KFSModel, question: str, n_options: int) -> tuple[list[float], float]:
    """Same chat layout as KFSModel.generate_segments with no vision/timeline segments; first-step letter log-probs."""
    tok = qa.processor.tokenizer
    ids = [qa._im_start_id] + tok.encode("user" + getattr(qa, "role_sep", "\n"), add_special_tokens=False)
    ids += tok.encode("\n", add_special_tokens=False) + tok.encode(question, add_special_tokens=False)
    ids += [qa._im_end_id] + tok.encode(getattr(qa, "end_sep", "\n"), add_special_tokens=False)
    ids += [qa._im_start_id] + tok.encode("assistant\n", add_special_tokens=False)
    logits = qa.model(input_ids=torch.tensor([ids], device=qa.model.device), use_cache=False).logits[0, -1].float()
    letter_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(n_options)]
    mass = float(torch.softmax(logits, dim=-1)[letter_ids].sum())
    return [round(float(x), 6) for x in torch.log_softmax(logits[letter_ids], dim=-1)], mass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--inject_abst", default="", help="round 16: priors for the questions with an inserted abstention option")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from llava_ov import load_qa

    qa = load_qa(args.model_path, "cuda:0", 1)
    print(json.dumps(vars(args)), flush=True)
    annos = [a for a in json.load(open(args.anno_path)) if a["task"] in BACKWARD_TASKS + REAL_TIME_TASKS]
    if args.inject_abst:
        print("inject_abst:", apply_inject(annos, args.inject_abst), "questions", flush=True)
    out, t0 = {}, time.perf_counter()
    for a in annos:
        n = len(a["options"])
        lp_q, mass = letter_logprobs(qa, build_qwen25_prompt(a["task"], a), n)
        lp_na, _ = letter_logprobs(qa, build_qwen25_prompt(a["task"], {**a, "question": "N/A"}), n)
        out[f"{a['task']}:{a['id']}"] = {"lp_q": lp_q, "lp_na": lp_na, "mass_q": round(mass, 5), "abst_idx": a.get("abst_idx")}
    json.dump(out, open(os.path.join(args.out_dir, "prior.json"), "w"), indent=0)
    print(f"done: {len(out)} questions in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
