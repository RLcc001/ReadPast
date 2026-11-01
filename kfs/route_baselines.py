"""Question-only routers from prior work, scored with the reader VLM itself (for the router comparison).

Per question (text only, no options, no frames) this records:
  ss_zs / ss_fs10  ShallowStream's query-choice gate (arXiv 2609.02780, github.com/CURRENTF/ShallowStream,
                   src/shallowstream/task_gate.py): the released zero-shot (retrieval_ab_v1) and 10-shot
                   (retrieval_ab_fewshot10_v1) prompts verbatim, chat template + generation prompt, score =
                   logit(A: retrieve) - logit(B: recent only). Thresholds are applied offline (kfs/analyze_routers.py).
  fb_s1 / fb_s2    FlashBack's structured temporal-evidence router (arXiv 2610.01192, appendix A.4): the stage-1
                   (5 states) and stage-2 (4 Current sub-states) prompts, representation-balanced over cyclic views
                   (every state once at each position and with each code); per view a log-softmax over the code
                   tokens, averaged per state. Stage 2 only runs when stage 1 picks CURRENT_PERCEPTION.
  fb_recall        FlashBack's fixed mapping: Historical / Interaction / Aggregation -> Recall, Predictive -> Native;
                   Current -> stage 2: Persistent / Outcome -> Recall, Boundary / Ongoing -> Native.
  p_past           our router (kfs/route.py prompt 0), so the synthetic set has every router's score.

Benches: ovo (keys "<task>:<id>"), sb (question_id), synth (kfs/data/route_synth_v1.json, keys "synth:<i>").
Output: {out_dir}/scores_<bench>.json

usage: python kfs/route_baselines.py --model_path M --out_dir $RX_RUN_DIR/out [--bench ovo sb synth]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ovo_kfs import BACKWARD_TASKS, REAL_TIME_TASKS, KFSModel  # noqa: E402
from route import ROUTE_PROMPT, p_past  # noqa: E402

import torch  # noqa: E402

# ---- ShallowStream (verbatim from src/shallowstream/task_gate.py) ----
SS_FEWSHOT10 = (
    ("What object is the person holding in the latest visible moment?", "B"),
    ("Where did the person leave the keys near the beginning of the video?", "A"),
    ("What word is currently displayed on the screen?", "B"),
    ("How many times did the bell ring over the course of the video?", "A"),
    ("Which item did the cook pick up first?", "A"),
    ("What color is the vehicle visible in the latest scene?", "B"),
    ("What was inside the box before it was emptied?", "A"),
    ("What action is the person performing right now?", "B"),
    ("Which object is immediately to the left of the cup in the latest frame?", "B"),
    ("How has the room changed compared with its appearance earlier in the video?", "A"),
)
SS_HEAD = ("You are deciding whether answering a video question requires retrieving older "
           "video memory. The system already has the latest video segment immediately "
           "available. Do not answer the video question itself.\n\n")
SS_OPTIONS = ("A. Answering reliably requires inspecting earlier video memory.\n"
              "B. The latest video segment is sufficient; older video memory is unnecessary.\n\n")


def ss_prompt(question: str, fewshot: bool) -> str:
    if fewshot:
        demos = "\n\n".join(f"Example {i}:\nVideo question: {q}\nCorrect routing option: {a}"
                            for i, (q, a) in enumerate(SS_FEWSHOT10, start=1))
        return (SS_HEAD + "Routing options:\n" + SS_OPTIONS + "Here are labeled routing examples:\n\n" + demos
                + "\n\nNow route this video question:\n<video_question>\n" + question + "\n</video_question>\n\n"
                "Output only A or B.\nRouting answer:")
    return (SS_HEAD + "<video_question>\n" + question + "\n</video_question>\n\n"
            "Choose exactly one routing option:\n" + SS_OPTIONS + "Output only A or B.\nRouting answer:")


# ---- FlashBack (appendix A.4 canonical views; emphasis removed as the paper notes it is typographic only) ----
FB1_HEAD = (
    "Classify the temporal proof form requested by the visual question.\n"
    "Classify only the request semantics, without assuming facts that are not stated in the request.\n\n"
    "Apply this priority: prefix-wide quantification or coverage is GLOBAL_AGGREGATION; an explicitly unobserved "
    "future target is PREDICTIVE_CONSEQUENCE; an observed order, change, interaction, or cause is "
    "TEMPORAL_INTERACTION; one explicitly earlier bounded fact is HISTORICAL_STATE; otherwise a query-boundary or "
    "latest-current target is CURRENT_PERCEPTION.\n\n"
    "A word such as count, compare, before, after, first, last, or never has no class by itself; use the time "
    "domain over which the question applies it.\n\n"
    "Contrast the following proof forms: a property requested at an explicit earlier time is HISTORICAL_STATE; "
    "the same property requested at the query boundary is CURRENT_PERCEPTION; an observed transition is "
    "TEMPORAL_INTERACTION; a statistic over the observed prefix is GLOBAL_AGGREGATION; and an unobserved later "
    "target is PREDICTIVE_CONSEQUENCE.\n\n"
    "Choose exactly one routing code.\n"
)
FB1_STATES = [
    ("CURRENT_PERCEPTION", "The request targets a configuration, state, or activity at the query boundary, including "
     "a latest state treated as current. It is not an explicitly earlier fact, an event relation, a prefix-wide "
     "aggregate, or an unobserved future target."),
    ("HISTORICAL_STATE", "The request targets one explicitly earlier, completed, bounded event or state. It does not "
     "ask for a relation among events or for coverage over multiple moments."),
    ("TEMPORAL_INTERACTION", "The request targets an observed order, transition, change, interaction, or causal "
     "relation inside one bounded episode."),
    ("GLOBAL_AGGREGATION", "The request targets multiple moments or the observed prefix as a set: aggregation, "
     "repetition, frequency, cross-period comparison, coverage, exclusion, first/last/unique, ever/never, or absence "
     "over time."),
    ("PREDICTIVE_CONSEQUENCE", "The requested outcome lies after the query boundary and has not yet been observed; the "
     "request asks for a forecast rather than an already realized result."),
]
FB1_CODES = ["A", "B", "C", "D", "E"]
FB1_RECALL = {"HISTORICAL_STATE": True, "TEMPORAL_INTERACTION": True, "GLOBAL_AGGREGATION": True,
              "PREDICTIVE_CONSEQUENCE": False}
FB2_HEAD = (
    "Classify the proof form of this CURRENT_PERCEPTION request.\n"
    "Classify only the target semantics, without assuming facts that are not stated in the request.\n\n"
    "Explicit wording controls the distinction: currently performing or moving is ONGOING_PROCESS; explicitly "
    "visible, present, or configured at this moment is SIMULTANEOUS_BOUNDARY_CONFIGURATION; latest-established, "
    "latest-known, or based on a prior update is PERSISTENT_CURRENT_STATE.\n\n"
    "Apply this priority: an already-realized result is REALIZED_BOUNDED_OUTCOME; an activity still underway is "
    "ONGOING_PROCESS; a state that semantically persists until updated is PERSISTENT_CURRENT_STATE; otherwise an "
    "instantaneous configuration is SIMULTANEOUS_BOUNDARY_CONFIGURATION.\n\n"
    "Choose exactly one routing code.\n"
)
FB2_STATES = [
    ("SIMULTANEOUS_BOUNDARY_CONFIGURATION", "The request explicitly targets what is visually present, visible, or "
     "configured right at the query boundary: appearance, visible text, a local quantity, or a simultaneous spatial "
     "relation. Use this for an explicit current observation even if that observed property could physically persist."),
    ("ONGOING_PROCESS", "The target is an action, motion, gesture, or process still underway at the query boundary; it "
     "does not ask for its completed result."),
    ("PERSISTENT_CURRENT_STATE", "The request explicitly asks for the latest established or carried state determined "
     "by a prior update, handoff, or observation, such as latest-known location or possession. Do not choose this "
     "merely because a property visible at the boundary could persist."),
    ("REALIZED_BOUNDED_OUTCOME", "The target is the already-realized completion, continuation result, or outcome of a "
     "bounded process that began earlier; it is neither ongoing nor an unobserved future forecast."),
]
FB2_CODES = ["S", "D", "L", "T"]
FB2_RECALL = {"SIMULTANEOUS_BOUNDARY_CONFIGURATION": False, "ONGOING_PROCESS": False,
              "PERSISTENT_CURRENT_STATE": True, "REALIZED_BOUNDED_OUTCOME": True}


def fb_views(head: str, states: list[tuple[str, str]], codes: list[str], tail: str, question: str):
    """Cyclic views: view v puts state (i + v) % n at position i with code codes[i]."""
    n = len(states)
    for v in range(n):
        order = [states[(i + v) % n] for i in range(n)]
        body = "".join(f"{codes[i]}. {name}: {desc}\n" for i, (name, desc) in enumerate(order))
        yield head + body + f"\nVisual request:\n{question}\n\n{tail}", {name: codes[i] for i, (name, _) in enumerate(order)}


@torch.inference_mode()
def next_logits(qa: KFSModel, prompt: str) -> torch.Tensor:
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = qa.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = qa.processor.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(qa.model.device)
    return qa.model(input_ids=ids, use_cache=False).logits[0, -1].float()


def single_token(tok, s: str) -> int:
    ids = tok.encode(s, add_special_tokens=False)
    assert len(ids) == 1, (s, ids)
    return ids[0]


def fb_stage(qa, tok, head, states, codes, tail, question) -> dict[str, float]:
    code_ids = [single_token(tok, c) for c in codes]
    acc = {name: 0.0 for name, _ in states}
    for prompt, code_of in fb_views(head, states, codes, tail, question):
        lp = torch.log_softmax(next_logits(qa, prompt)[code_ids], dim=-1)
        for name in acc:
            acc[name] += float(lp[codes.index(code_of[name])]) / len(states)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--sb_anno", default="/workspace/datasets/streaming/streamingbench/questions_real.json")
    ap.add_argument("--synth", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/route_synth_v1.json"))
    ap.add_argument("--bench", nargs="+", default=["synth", "ovo", "sb"])
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    qa = KFSModel(model_name=args.model_path, device="cuda:0", max_new_tokens=1)
    tok = qa.processor.tokenizer
    a_id, b_id = single_token(tok, "A"), single_token(tok, "B")
    words = {w: sorted({tok.encode(s, add_special_tokens=False)[0] for s in (w, w.capitalize())}) for w in ("past", "now")}
    print(json.dumps(vars(args)), flush=True)

    for bench in args.bench:
        if bench == "synth":
            qs = [(f"synth:{i}", it["q"]) for i, it in enumerate(json.load(open(args.synth))["items"])]
        elif bench == "sb":
            qs = [(q["question_id"], q["question"]) for e in json.load(open(args.sb_anno)) for q in e["questions"]]
        else:
            qs = [(f"{a['task']}:{a['id']}", a["question"]) for a in json.load(open(args.anno_path))
                  if a["task"] in BACKWARD_TASKS + REAL_TIME_TASKS]
        out, t0 = {}, time.perf_counter()
        for i, (key, question) in enumerate(qs):
            question = question.strip()
            rec = {}
            for name, fs in (("ss_zs", False), ("ss_fs10", True)):
                lg = next_logits(qa, ss_prompt(question, fs))
                rec[name] = round(float(lg[a_id] - lg[b_id]), 4)
            s1 = fb_stage(qa, tok, FB1_HEAD, FB1_STATES, FB1_CODES, "Question type:", question)
            top1 = max(s1, key=s1.get)
            rec["fb_s1"] = {k: round(v, 4) for k, v in s1.items()}
            if top1 == "CURRENT_PERCEPTION":
                s2 = fb_stage(qa, tok, FB2_HEAD, FB2_STATES, FB2_CODES, "CURRENT proof form:", question)
                top2 = max(s2, key=s2.get)
                rec["fb_s2"] = {k: round(v, 4) for k, v in s2.items()}
                rec["fb_state"], rec["fb_recall"] = top2, FB2_RECALL[top2]
            else:
                rec["fb_state"], rec["fb_recall"] = top1, FB1_RECALL[top1]
            rec["p_past"] = round(p_past(qa, question, words, ROUTE_PROMPT)[0], 5)
            out[key] = rec
            if i % 500 == 0:
                print(f"{bench} {i}/{len(qs)} {time.perf_counter() - t0:.0f}s {key} {rec}", flush=True)
        json.dump(out, open(os.path.join(args.out_dir, f"scores_{bench}.json"), "w"), indent=0)
        print(f"{bench}: {len(out)} questions in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
