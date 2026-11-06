"""Round 19: time the public D-HSM code (third_party/D-HSM @04579a0) with exactly the round-18 settings
(kfs/run_dhsm.sh: dhsm memory + uniform_abstention, 4 recent frames, 1 fps) on the round-19 question sample.

D-HSM's code is NOT modified: this wrapper imports experiments/evaluate_ovo.py, wraps a few functions / methods with
synchronized timers, then runs its own main() with the round-18 arguments. Per question it records
  decode_s     decode_video_to_chunks_qwen (all 1-s chunks up to the query time)
  memory_s     build_memory_from_chunks: query-time memory build (D-HSM's OVO protocol rebuilds the memory for every
               question from <= max_extraction_chunks=20 history chunks sampled evenly) = captioning + entity update
  caption_s    of which VLM captioning (_pre_caption_chunks + _caption_chunk), n_captioned chunks
  answer_s     answer_with_memory_mcq (retrieval + VLM answer)
  total_s      the whole _build_and_answer call; peak_gb = max_memory_allocated over it
D-HSM's own generate_time = memory_s + answer_s (it excludes decoding), as in its result rows.
Output: {result_dir}/bench_dhsm.json; D-HSM's own result files land in the same dir (answers are checked against
round 18 on the host).

usage (via rx, run from the D-HSM dir like kfs/run_dhsm.sh):  bash kfs/run_bench_dhsm.sh
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
import evaluate_ovo as E  # noqa: E402
import torch  # noqa: E402
from dhsm.memory_selection import evaluator_class_for  # noqa: E402

REC: list[dict] = []
CUR: dict | None = None


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def add(field, dt, n=0):
    if CUR is not None:
        CUR[field] = CUR.get(field, 0.0) + dt
        if n:
            CUR["n_captioned"] = CUR.get("n_captioned", 0) + n


def wrap_fn(fn, field, count=None):
    def inner(*a, **k):
        sync()
        t = time.perf_counter()
        out = fn(*a, **k)
        sync()
        add(field, time.perf_counter() - t, count(a, k) if count else 0)
        return out
    return inner


orig_build_answer = E._build_and_answer


def build_answer(question, options, query_cutoff, video_path, *a, **k):
    global CUR
    CUR = {"video_path": video_path, "question": question}
    sync()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    resp, meta = orig_build_answer(question, options, query_cutoff, video_path, *a, **k)
    sync()
    CUR.update(total_s=time.perf_counter() - t, peak_gb=torch.cuda.max_memory_allocated() / 2**30,
               response=resp, generate_time=meta.get("generate_time"), use_memory=meta.get("route_use_memory"),
               num_hist_chunks=meta.get("num_hist_chunks"), error=meta.get("error"))
    REC.append(CUR)
    CUR = None
    return resp, meta


def main():
    E._build_and_answer = build_answer
    E.decode_video_to_chunks_qwen = wrap_fn(E.decode_video_to_chunks_qwen, "decode_s")
    cls = evaluator_class_for("entity_resolved")
    cls.build_memory_from_chunks = wrap_fn(cls.build_memory_from_chunks, "memory_s")
    cls._pre_caption_chunks = wrap_fn(cls._pre_caption_chunks, "caption_s", count=lambda a, k: len(a[1]))
    cls._caption_chunk = wrap_fn(cls._caption_chunk, "caption_s")
    cls.answer_with_memory_mcq = wrap_fn(cls.answer_with_memory_mcq, "answer_s")
    result_dir = sys.argv[sys.argv.index("--result_dir") + 1]
    try:
        E.main()
    finally:
        os.makedirs(result_dir, exist_ok=True)
        json.dump({"rows": REC, "gpu": torch.cuda.get_device_name(0), "argv": sys.argv},
                  open(os.path.join(result_dir, "bench_dhsm.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
