"""Keyframe selection on top of SimpleStream (Qwen2.5-VL, OVO-Bench backward + real-time).

Per question the VLM sees the SimpleStream window (last `recent` 1-fps frames of the causal
prefix) plus `k` history keyframes picked from the frames before that window. All frames are
sampled and resized exactly like SimpleStream's exact-recent decoder, so `--k 0 --layout ss`
reproduces the SimpleStream recent-N baseline.

Selectors:  none | uniform (evenly spaced over history) | clip (CLIP text-image top-k)
            clip_mmr (CLIP relevance with MMR redundancy penalty)
Layouts:    ss      all frames in one vision block, chronological (SimpleStream's prompt)
            blocks  one vision block per frame, no text
            tagged  history frames each with a timestamp, then "current view" + recent block
History frames can be shrunk (--hist_scale, fraction of the recent frames' pixels) and the
question can be prefixed with an evidence-skeptical note about the history (--hist_note).
Text memory (--captions DIR from kfs/caption_ovo.py): a timeline of the captions of clips that ended
by the query time is put before the recent block (ss layout only); --cap_note adds an abstention hint.
Timeline-indexed recall (--recall): the model first reads the numbered timeline + current view and names the
clip that holds the evidence (0 = none); the CLIP top-`recall_k` frames inside that clip are then shown between
the timeline and the current view. Questions answered with 0 get exactly the --captions prompt.
Bounded memory (--budget N): when the caption lines exceed N tokens, the latest `keep_recent` captions stay
and older ones are admitted newest-first (--retain recency), by surprisal under the earlier memory
(--retain surprise, scores from kfs/surprisal.py) or by BM25 relevance to the question and its options
(--retain relevance; ties newest-first), then shown in time order.
Control (--foreign_timeline): same clips and timestamps, but the caption texts come from another source video.
Memory flush (--flush): at query time the partial clip after the last complete caption clip (up to the query) is
captioned with the same captioner prompt and appended as the newest timeline entry, so memory + recent window cover
the whole past (clips shorter than --flush_min seconds are skipped).
Progressive memory access (--cascade ROUTE_JSON, needs --captions):
  L0  SimpleStream (recent frames only), always.
  L1  + caption timeline, when the question is about the past (p_past from kfs/route.py >= --route_th) or L0 is
      unsure (max option prob < --tau0).                                                  [module 1]
  L2  + look-back evidence, when L1 is unsure (< --tau1): the CLIP top-`lookback_k` history frames are written
      into a short text note by the same model (frames never enter the answering context), the note is placed
      between the timeline and the current view, and the question is answered again.      [module 2]
  --note_mode desc: neutral description (the question is not shown); qdesc: description focused on the question.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zlib
from typing import Any

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

SS_ROOT = os.environ.get("SS_ROOT", "/workspace/streamkv/third_party/SimpleStream")
sys.path.insert(0, SS_ROOT)
sys.path.insert(0, "/workspace/exptools")

import rxlib  # noqa: E402
from lib.qwen_exact_recent_decoder import (  # noqa: E402
    _build_sampling_plan,
    _decode_indices,
    _resize_like_fetch_video,
)
from lib.recent_window_eval import calculate_ovo_scores, load_jsonl_results  # noqa: E402
from lib.recent_window_eval_qwen25 import RecentWindowQAModel, build_qwen25_prompt  # noqa: E402
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS  # noqa: E402
from qwen_vl_utils.vision_process import smart_resize  # noqa: E402

HIST_NOTE = (
    "The earlier keyframes were retrieved automatically and may be unrelated to the question. "
    "Use them only if they clearly contain the needed evidence; otherwise answer from the current view.\n"
)

CAP_NOTE = (
    "The timeline was generated automatically and may miss details. Use it for what happened earlier and the "
    "current view for what is happening now. If the question asks about something that appears in neither, "
    "it cannot be answered from the video.\n"
)

NOTE_PROMPTS = {
    "desc": (
        "These frames were taken from earlier in the video at the times shown. Describe what they show: the people, "
        "the objects and where they are, what the hands are doing, and any visible text. Be concrete and brief "
        "(at most three sentences)."
    ),
    "qdesc": (
        "These frames were taken from earlier in the video at the times shown. Describe what they show that is "
        "relevant to this question: {question}\nMention the objects, where they are, and the actions. If the frames "
        "do not show what the question asks about, say so. Do not guess (at most three sentences)."
    ),
}

LOCATE_PROMPT = (
    "Question: {question}\nOptions: {options}\nDo not answer the question yet. Which numbered clip of the timeline "
    "shows the evidence needed to answer it? Reply with the clip number only. Reply 0 if the current view alone is "
    "enough, or if no clip in the timeline is related to the question."
)

VisionBlock = list[tuple[torch.Tensor, torch.Tensor]]  # [(embeds, grid_thw row)] for one <vision_start> block


class KFSModel(RecentWindowQAModel):
    """Qwen2.5-VL or Qwen3-VL (the base class picks the model class from the name).

    per_frame_blocks: every frame gets its own <vision_start> block, as SimpleStream's Qwen3 OVO script does; the
    Qwen2.5 scripts and SimpleStream's StreamingBench script (all backbones) put a window in one block.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.per_frame_blocks = "qwen3" in self.model_name.lower()

    @torch.inference_mode()
    def generate_segments(self, segments: list[str | VisionBlock], question: str, n_options: int) -> dict[str, Any]:
        """Build `<im_start>user\\n{segments}\\n{question}<im_end>\\n<im_start>assistant\\n` and generate.

        With a single vision block this is token-for-token SimpleStream's Qwen2.5 prompt (per-frame blocks: its Qwen3
        OVO prompt). Also returns the first-step distribution over the option letters (raw logits) as a confidence signal.
        """
        tok = self.processor.tokenizer
        text_device = self._get_text_input_device()
        ids: list[int] = [self._im_start_id] + tok.encode("user\n", add_special_tokens=False)
        embeds: list[torch.Tensor] = []
        block_grids: list[torch.Tensor] = []
        for seg in segments:
            if isinstance(seg, str):
                ids += tok.encode(seg, add_special_tokens=False)
                continue
            for blk in ([[f] for f in seg] if self.per_frame_blocks else [seg]):
                n_tok = sum(int(e.shape[0]) for e, _ in blk)
                ids += [self._vision_start_id] + [self.image_token_id] * n_tok + [self._vision_end_id]
                embeds += [e for e, _ in blk]
                # get_rope_index consumes one grid row per vision block (SimpleStream relies on the same behaviour)
                block_grids.append(blk[0][1])
        ids += tok.encode("\n", add_special_tokens=False) + tok.encode(question, add_special_tokens=False)
        ids += [self._im_end_id] + tok.encode("\n", add_special_tokens=False)
        ids += [self._im_start_id] + tok.encode("assistant\n", add_special_tokens=False)

        input_ids = torch.tensor([ids], dtype=torch.long, device=text_device)
        attention_mask = torch.ones_like(input_ids)
        text_model = self._get_text_model()
        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        vis = torch.cat(embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        mask = (input_ids == self.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(mask, vis)
        grids = torch.cat(block_grids, dim=0).to(inputs_embeds.device)
        position_ids, _ = self._get_multimodal_model().get_rope_index(
            input_ids=input_ids, image_grid_thw=grids, video_grid_thw=None, attention_mask=attention_mask
        )

        t0 = time.perf_counter()
        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            output_logits=True,
            return_dict_in_generate=True,
        )
        gen_time = time.perf_counter() - t0
        seq = out.sequences[0]
        new = seq[input_ids.shape[1]:] if seq.shape[0] > input_ids.shape[1] else seq
        answer = self.processor.batch_decode([new], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        logits = out.logits[0][0].float()
        letter_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(n_options)]
        p_full = torch.softmax(logits, dim=-1)
        p_opts = torch.softmax(logits[letter_ids], dim=-1)
        return {
            "answer": answer,
            "p_opts": [round(float(p), 5) for p in p_opts],
            "p_letters_mass": round(float(p_full[letter_ids].sum()), 5),
            "num_vision_tokens": int(vis.shape[0]),
            "generate_time": gen_time,
        }


class ClipScorer:
    """CLIP text-image relevance over the 1-fps prefix frames, image features cached per question."""

    def __init__(self, path: str, device: str, cache_dir: str, decode_short_side: int = 336, batch: int = 64):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(path)
        self.device = device
        self.cache_dir = cache_dir
        self.short = decode_short_side
        self.batch = batch
        os.makedirs(cache_dir, exist_ok=True)

    @torch.inference_mode()
    def image_feats(self, key: str, video_path: str, indices: list[int]) -> np.ndarray:
        cache = os.path.join(self.cache_dir, f"{key}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            if z["idx"].tolist() == list(indices):
                return z["feat"].astype(np.float32)
        import decord

        h, w = decord.VideoReader(video_path).get_batch([0]).shape[1:3]
        s = self.short / min(h, w)
        reader = decord.VideoReader(video_path, width=int(round(w * s)), height=int(round(h * s)))
        feats = []
        for i in range(0, len(indices), self.batch):
            frames = reader.get_batch(indices[i:i + self.batch]).asnumpy()
            px = self.processor(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
            f = self.model.get_image_features(pixel_values=px)
            feats.append(torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy())
        feat = np.concatenate(feats, 0) if feats else np.zeros((0, self.model.config.projection_dim), np.float32)
        tmp = f"{cache}.{os.getpid()}.tmp.npz"
        np.savez(tmp, idx=np.asarray(indices), feat=feat.astype(np.float16))
        os.replace(tmp, cache)
        return feat

    @torch.inference_mode()
    def text_feat(self, text: str) -> np.ndarray:
        t = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        f = self.model.get_text_features(**t)
        return torch.nn.functional.normalize(f.float(), dim=-1)[0].cpu().numpy()


def select_history(args, scorer: ClipScorer | None, anno: dict, video_path: str, full: list[int], hist: list[int]) -> tuple[list[int], list[float] | None]:
    """Return chosen history positions (indices into `hist`, chronological) and their scores."""
    k = args.k
    if k <= 0 or not hist:
        return [], None
    if len(hist) <= k:
        return list(range(len(hist))), None
    if args.select == "uniform":
        return [int((i + 0.5) * len(hist) / k) for i in range(k)], None
    img = scorer.image_feats(f"{anno['task']}_{anno['id']}", video_path, full)[: len(hist)]
    sims = img @ scorer.text_feat(anno["question"])
    if args.select == "clip":
        top = np.argsort(-sims)[:k].tolist()
    elif args.select == "clip_mmr":
        rel = (sims - sims.min()) / max(float(sims.max() - sims.min()), 1e-6)
        top = []
        for _ in range(k):
            red = (img @ img[top].T).max(1) if top else np.zeros_like(rel)
            score = (1 - args.mmr_lambda) * rel - args.mmr_lambda * red
            score[top] = -np.inf
            top.append(int(np.argmax(score)))
    else:
        raise ValueError(args.select)
    top = sorted(top)
    return top, [round(float(sims[i]), 4) for i in top]


def to_pil(frame: torch.Tensor) -> Image.Image:
    return Image.fromarray(frame.clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy())


def build_segments(layout: str, hist_blocks: VisionBlock, hist_ts: list[float], recent_block: VisionBlock, n_recent: int) -> list:
    if layout == "ss":
        return [hist_blocks + recent_block]
    if layout == "blocks":
        return [[b] for b in hist_blocks + recent_block]
    if layout == "tagged":
        segs: list = []
        if hist_blocks:
            segs.append("Keyframes from earlier in the video:\n")
            for b, ts in zip(hist_blocks, hist_ts):
                segs += [f"[{ts:.0f}s] ", [b], "\n"]
        segs += [f"Current view (the latest {n_recent} seconds):\n", recent_block]
        return segs
    raise ValueError(layout)


_captions: dict[str, list[dict]] = {}
_surprisal: dict[str, dict[str, float]] = {}
_foreign_loaded: set[str] = set()
TIMELINE_HEADER = "Timeline of the video so far (automatic captions of earlier clips):\n"


def is_no_change(text: str) -> bool:
    """Delta captions answer "No change." for clips without news; the timeline skips them."""
    return text.strip().lower().rstrip(".") == "no change"


def timeline_captions(cap_dir: str, video: str, query_ts: float) -> list[dict]:
    """Informative captions of the clips that ended by `query_ts`."""
    if video not in _captions:
        path = os.path.join(cap_dir, video.replace("/", "__") + ".json")
        _captions[video] = json.load(open(path))["captions"] if os.path.exists(path) else []
    return [c for c in _captions[video] if c["t1"] <= query_ts + 1e-6 and not is_no_change(c["text"])]


def flush_caption(args, qa: KFSModel, video_path: str, raw_fps: float, caps: list[dict], q_ts: float) -> dict | None:
    """Caption [end of the last complete clip in `caps` (causal), query time] with kfs/caption_ovo.py's prompt."""
    from caption_ovo import PROMPT  # lazy: caption_ovo imports this module

    import decord

    t0 = max((c["t1"] for c in caps), default=0)
    span = q_ts - t0
    if span < args.flush_min:
        return None
    n = max(2, min(8, int(span)))
    vr = decord.VideoReader(video_path)
    idx = [min(int((t0 + (i + 0.5) * span / n) * raw_fps), len(vr) - 1) for i in range(n)]
    frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2).float()
    h, w = frames.shape[-2:]
    hh, ww = smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=256 * 28 * 28)
    frames = TF.resize(frames, [hh, ww], interpolation=InterpolationMode.BICUBIC, antialias=True)
    block = [qa.encode_vision([to_pil(f)]) for f in frames]
    max_new, qa.max_new_tokens = qa.max_new_tokens, 96
    text = qa.generate_segments([block], PROMPT.format(n=n, seg=round(span), t0=t0, t1=round(q_ts)), 1)["answer"]
    qa.max_new_tokens = max_new
    return {"t0": t0, "t1": round(q_ts), "text": text}


def foreign_captions(cap_dir: str, video: str, caps: list[dict]) -> list[dict]:
    """Control: keep the clips and timestamps of `caps` but take the texts from another source video that has at
    least as many informative captions (chosen deterministically from the video name)."""
    if not caps:
        return caps
    if cap_dir not in _foreign_loaded:
        for f in sorted(os.listdir(cap_dir)):
            rec = json.load(open(os.path.join(cap_dir, f)))
            _captions.setdefault(rec["video"], rec["captions"])
        _foreign_loaded.add(cap_dir)
    informative = {v: [c for c in cs if not is_no_change(c["text"])] for v, cs in _captions.items() if v != video}
    donors = sorted(v for v, cs in informative.items() if len(cs) >= len(caps))
    if donors:
        texts = informative[donors[zlib.crc32(video.encode()) % len(donors)]]
    else:  # longer than every other video: chain the longest ones
        texts = [c for v in sorted(informative, key=lambda v: -len(informative[v])) for c in informative[v]]
    return [{**c, "text": d["text"]} for c, d in zip(caps, texts)]


def timeline_line(c: dict, num: int | None = None) -> str:
    return f"{f'[{num}] ' if num else ''}[{c['t0']}s-{c['t1']}s] {c['text'].strip()}\n"


def format_timeline(caps: list[dict], numbered: bool = False) -> str:
    if not caps:
        return ""
    return TIMELINE_HEADER + "".join(timeline_line(c, i + 1 if numbered else None) for i, c in enumerate(caps))


_STOP = set("a an the of to in on at for with and or is are was were be been it its this that these those i my me you "
             "he she they his her their what which who whom where when how why did do does from by as into out up down "
             "over after before while then there here some any one not no yes can could would should will".split())


def _terms(text: str) -> list[str]:
    return [w.rstrip("s") for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2]


def bm25(query: str, docs: list[str], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """Okapi BM25 of each doc for the query; IDF over `docs` (the captions written so far)."""
    toks = [_terms(d) for d in docs]
    avg = sum(map(len, toks)) / max(1, len(toks))
    q = set(_terms(query))
    df = {w: sum(w in set(t) for t in toks) for w in q}
    idf = {w: np.log(1 + (len(toks) - df[w] + 0.5) / (df[w] + 0.5)) for w in q}
    out = []
    for t in toks:
        tf = {w: t.count(w) for w in q}
        out.append(float(sum(idf[w] * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / max(avg, 1e-6))) for w in q)))
    return out


def retain_captions(args, tok, video: str, caps: list[dict], query: str = "") -> list[dict]:
    """Keep the caption lines within `budget` tokens: the latest `keep_recent` always stay, older ones are
    admitted newest-first (recency) or most-surprising-first (surprise) while they fit."""
    lens = [len(tok.encode(timeline_line(c), add_special_tokens=False)) for c in caps]
    if sum(lens) <= args.budget or len(caps) <= args.keep_recent:
        return caps
    n_old = len(caps) - args.keep_recent
    keep = set(range(n_old, len(caps)))
    used = sum(lens[i] for i in keep)
    if args.retain == "recency":
        order = list(range(n_old - 1, -1, -1))
    elif args.retain == "relevance":
        rel = bm25(query, [c["text"] for c in caps])
        order = sorted(range(n_old - 1, -1, -1), key=lambda i: -rel[i])  # stable: ties stay newest-first
    else:
        if video not in _surprisal:
            _surprisal[video] = json.load(open(os.path.join(args.surprisal, video.replace("/", "__") + ".json")))["nll"]
        nll = _surprisal[video]
        order = sorted(range(n_old), key=lambda i: -nll[str(caps[i]["t0"])])
    for i in order:
        if used + lens[i] <= args.budget:
            keep.add(i)
            used += lens[i]
    return [caps[i] for i in sorted(keep)]


def locate_and_recall(args, qa: KFSModel, scorer: ClipScorer, anno: dict, raw_fps: float, video_path: str,
                      full: list[int], hist: list[int], caps: list[dict], recent_block: VisionBlock) -> tuple[int, list[int], str]:
    """Ask the model which timeline clip holds the evidence (0 = none); return it and the CLIP top frames inside it."""
    segs = [format_timeline(caps, numbered=True) + f"Current view (the latest {args.recent} seconds):\n", recent_block]
    opts = "; ".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(anno["options"])) + ";"
    max_new, qa.max_new_tokens = qa.max_new_tokens, 8
    out = qa.generate_segments(segs, LOCATE_PROMPT.format(question=anno["question"], options=opts), len(anno["options"]))
    qa.max_new_tokens = max_new
    m = re.search(r"\d+", out["answer"])
    k = int(m.group()) if m else 0
    if not 1 <= k <= len(caps):
        return 0, [], out["answer"]
    c = caps[k - 1]
    cand = [p for p, i in enumerate(hist) if c["t0"] <= i / raw_fps < c["t1"]]
    if not cand:
        return k, [], out["answer"]
    img = scorer.image_feats(f"{anno['task']}_{anno['id']}", video_path, full)
    sims = img[cand] @ scorer.text_feat(anno["question"])
    top = sorted(cand[j] for j in np.argsort(-sims)[: args.recall_k])
    return k, [hist[p] for p in top], out["answer"]


def evaluate_one(args, qa: KFSModel, scorer: ClipScorer | None, anno: dict) -> dict[str, Any]:
    video_path = os.path.join(args.chunked_dir, f"{anno['id']}.mp4")
    ele = {"video": video_path, "fps": float(args.fps)}
    plan = _build_sampling_plan(ele, last_nframes=args.recent)
    full = plan.full_indices
    recent, hist = full[-args.recent:], full[:-args.recent]

    t0 = time.perf_counter()
    pos, scores = select_history(args, scorer, anno, video_path, full, hist)
    select_time = time.perf_counter() - t0
    chosen = [hist[p] for p in pos]

    frames = _decode_indices(video_path, plan.backend, chosen + recent)
    frames = _resize_like_fetch_video(frames, ele, plan.sampled_nframes_full, image_patch_size=14)
    frame_list = list(frames)
    if chosen and args.hist_scale < 1.0:
        h, w = frames.shape[-2:]
        hh, ww = smart_resize(h, w, factor=28, min_pixels=4 * 28 * 28, max_pixels=int(h * w * args.hist_scale))
        for i in range(len(chosen)):
            frame_list[i] = TF.resize(frame_list[i], [hh, ww], interpolation=InterpolationMode.BICUBIC, antialias=True)
    blocks: VisionBlock = []
    for fr in frame_list:
        emb, grid = qa.encode_vision([to_pil(fr)])
        blocks.append((emb, grid))

    hist_ts = [i / plan.raw_fps for i in chosen]
    segments = build_segments(args.layout, blocks[:len(chosen)], hist_ts, blocks[len(chosen):], len(recent))
    question = build_qwen25_prompt(anno["task"], anno)
    if chosen and args.hist_note:
        question = HIST_NOTE + question
    caps = timeline_captions(args.captions, anno["video"], full[-1] / plan.raw_fps) if args.captions else []
    n_caps_all = len(caps)
    if args.foreign_timeline:
        caps = foreign_captions(args.captions, anno["video"], caps)
    flush = None
    if args.flush and args.captions:
        flush = flush_caption(args, qa, video_path, plan.raw_fps, caps, full[-1] / plan.raw_fps)
        if flush:
            caps = caps + [flush]
    if args.budget:
        query = anno["question"] + " " + " ".join(anno["options"])
        caps = retain_captions(args, qa.processor.tokenizer, anno["video"], caps, query)
    timeline = format_timeline(caps)
    rclip, recalled, locate_answer, locate_time = 0, [], "", 0.0
    if args.recall and caps:
        t0 = time.perf_counter()
        rclip, recalled, locate_answer = locate_and_recall(
            args, qa, scorer, anno, plan.raw_fps, video_path, full, hist, caps, blocks[len(chosen):])
        locate_time = time.perf_counter() - t0
    if recalled:
        rframes = _decode_indices(video_path, plan.backend, recalled)
        rframes = _resize_like_fetch_video(rframes, ele, plan.sampled_nframes_full, image_patch_size=14)
        rblock: VisionBlock = [qa.encode_vision([to_pil(fr)]) for fr in rframes]
        c = caps[rclip - 1]
        segments = [timeline + f"Frames recalled from the timeline clip [{c['t0']}s-{c['t1']}s]:\n", rblock,
                    f"\nCurrent view (the latest {len(recent)} seconds):\n"] + segments
    elif timeline:
        segments = [timeline + f"Current view (the latest {len(recent)} seconds):\n"] + segments
    if timeline and args.cap_note:
        question = CAP_NOTE + question
    out = qa.generate_segments(segments, question, len(anno["options"]))
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "response": out["answer"],
        "ground_truth": chr(65 + anno["gt"]),
        "p_opts": out["p_opts"],
        "p_letters_mass": out["p_letters_mass"],
        "n_hist_candidates": len(hist),
        "hist_ts": [round(t, 2) for t in hist_ts],
        "hist_scores": scores,
        "query_ts": round(full[-1] / plan.raw_fps, 2),
        "num_frames": len(blocks) + len(recalled),
        "num_vision_tokens": out["num_vision_tokens"],
        "n_captions": len(caps),
        "n_captions_all": n_caps_all,
        "timeline_tokens": len(qa.processor.tokenizer.encode(timeline, add_special_tokens=False)) if timeline else 0,
        "flush": flush["text"] if flush else "",
        "recall_clip": rclip,
        "recall_ts": [round(i / plan.raw_fps, 2) for i in recalled],
        "locate_answer": locate_answer,
        "locate_time": locate_time,
        "select_time": select_time,
        "generate_time": out["generate_time"],
    }


_routes: dict[str, dict] = {}


def evaluate_cascade(args, qa: KFSModel, scorer: ClipScorer, anno: dict) -> dict[str, Any]:
    """L0 -> L1 -> L2 for one question (see the module docstring); returns the record of the final level."""
    if not _routes:
        _routes.update(json.load(open(args.cascade)))
    key = f"{anno['task']}:{anno['id']}"
    video_path = os.path.join(args.chunked_dir, f"{anno['id']}.mp4")
    ele = {"video": video_path, "fps": float(args.fps)}
    plan = _build_sampling_plan(ele, last_nframes=args.recent)
    full = plan.full_indices
    recent, hist = full[-args.recent:], full[:-args.recent]
    frames = _resize_like_fetch_video(_decode_indices(video_path, plan.backend, recent), ele, plan.sampled_nframes_full, image_patch_size=14)
    recent_block: VisionBlock = [qa.encode_vision([to_pil(fr)]) for fr in frames]
    question = build_qwen25_prompt(anno["task"], anno)
    n_opt = len(anno["options"])
    cur = f"Current view (the latest {len(recent)} seconds):\n"
    rec: dict[str, Any] = {"p_past": _routes[key]["p_past"], "level": 0, "note": "", "lookback_ts": []}

    out = qa.generate_segments([recent_block], question, n_opt)
    rec.update(response_l0=out["answer"], p_opts_l0=out["p_opts"])
    caps = timeline_captions(args.captions, anno["video"], full[-1] / plan.raw_fps)
    timeline = format_timeline(caps)
    if timeline and (rec["p_past"] >= args.route_th or max(out["p_opts"]) < args.tau0):
        rec["level"] = 1
        out = qa.generate_segments([timeline + cur, recent_block], question, n_opt)
        rec.update(response_l1=out["answer"], p_opts_l1=out["p_opts"])
        if max(out["p_opts"]) < args.tau1 and hist:
            pos, _ = select_history(argparse.Namespace(k=args.lookback_k, select="clip"), scorer, anno, video_path, full, hist)
            chosen = [hist[p] for p in pos]
            lb = _resize_like_fetch_video(_decode_indices(video_path, plan.backend, chosen), ele, plan.sampled_nframes_full, image_patch_size=14)
            ts = [i / plan.raw_fps for i in chosen]
            segs: list = []
            for fr, t in zip(lb, ts):
                segs += [f"[{t:.0f}s] ", [qa.encode_vision([to_pil(fr)])], "\n"]
            max_new, qa.max_new_tokens = qa.max_new_tokens, args.note_tokens
            note = qa.generate_segments(segs, NOTE_PROMPTS[args.note_mode].format(question=anno["question"].strip()), n_opt)["answer"]
            qa.max_new_tokens = max_new
            stamp = ", ".join(f"{t:.0f}s" for t in ts)
            look = f"A closer look at frames retrieved from earlier ({stamp}): {note.strip()}\n"
            out = qa.generate_segments([timeline + look + cur, recent_block], question, n_opt)
            rec.update(level=2, note=note, lookback_ts=[round(t, 2) for t in ts])
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "response": out["answer"],
        "ground_truth": chr(65 + anno["gt"]),
        "p_opts": out["p_opts"],
        "query_ts": round(full[-1] / plan.raw_fps, 2),
        "n_captions": len(caps),
        "timeline_tokens": len(qa.processor.tokenizer.encode(timeline, add_special_tokens=False)) if rec["level"] else 0,
        "num_vision_tokens": out["num_vision_tokens"],
        "select_time": 0.0,
        **rec,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
    ap.add_argument("--chunked_dir", default="/workspace/datasets/streaming/ovo_bench/chunked_videos")
    ap.add_argument("--result_dir", required=True)
    ap.add_argument("--recent", type=int, default=4)
    ap.add_argument("--k", type=int, default=0, help="number of history keyframes")
    ap.add_argument("--select", choices=["none", "uniform", "clip", "clip_mmr"], default="none")
    ap.add_argument("--mmr_lambda", type=float, default=0.5)
    ap.add_argument("--layout", choices=["ss", "blocks", "tagged"], default="ss")
    ap.add_argument("--hist_scale", type=float, default=1.0, help="history frame pixels as a fraction of recent frames")
    ap.add_argument("--hist_note", action="store_true", help="prefix the question with an evidence-skeptical note")
    ap.add_argument("--captions", default="", help="caption dir from kfs/caption_ovo.py (text memory)")
    ap.add_argument("--cap_note", action="store_true", help="prefix the question with an abstention hint when a timeline is shown")
    ap.add_argument("--recall", action="store_true", help="timeline-indexed recall of frames (needs --captions)")
    ap.add_argument("--recall_k", type=int, default=2, help="frames recalled from the located clip")
    ap.add_argument("--budget", type=int, default=0, help="token budget of the caption lines (0 = unbounded)")
    ap.add_argument("--keep_recent", type=int, default=4, help="latest captions always kept under a budget")
    ap.add_argument("--retain", choices=["recency", "surprise", "relevance"], default="recency")
    ap.add_argument("--surprisal", default="", help="dir from kfs/surprisal.py (default {captions}_surprisal)")
    ap.add_argument("--foreign_timeline", action="store_true", help="control: caption texts from another video")
    ap.add_argument("--flush", action="store_true", help="caption the partial clip up to the query time")
    ap.add_argument("--flush_min", type=float, default=2.0)
    ap.add_argument("--cascade", default="", help="route.json from kfs/route.py: progressive memory access")
    ap.add_argument("--route_th", type=float, default=0.5, help="L1 when p_past >= route_th")
    ap.add_argument("--tau0", type=float, default=0.5, help="L1 when L0 max option prob < tau0")
    ap.add_argument("--tau1", type=float, default=0.6, help="L2 when L1 max option prob < tau1")
    ap.add_argument("--lookback_k", type=int, default=4)
    ap.add_argument("--note_mode", choices=list(NOTE_PROMPTS), default="desc")
    ap.add_argument("--note_tokens", type=int, default=128)
    ap.add_argument("--keys", default="", help="json list of '<task>:<id>' to evaluate (subset runs)")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--clip_path", default="/workspace/models/clip-vit-large-patch14-336-st")
    ap.add_argument("--cache_dir", default="/workspace/streamkv/cache/clip_l336_ovo")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per task group (smoke tests)")
    args = ap.parse_args()
    assert not args.captions or args.layout == "ss", "--captions is implemented for the ss layout only"
    assert not args.recall or (args.captions and args.k == 0), "--recall needs --captions and no history keyframes"
    assert not args.budget or args.captions, "--budget needs --captions"
    assert not args.foreign_timeline or args.captions, "--foreign_timeline needs --captions"
    assert not args.cascade or (args.captions and args.k == 0), "--cascade needs --captions and no history keyframes"
    args.surprisal = args.surprisal or args.captions.rstrip("/") + "_surprisal"

    annos = json.load(open(args.anno_path))
    groups = {
        "backward": [a for a in annos if a["task"] in BACKWARD_TASKS],
        "realtime": [a for a in annos if a["task"] in REAL_TIME_TASKS],
    }
    if args.limit:
        groups = {g: v[: args.limit] for g, v in groups.items()}
    if args.keys:
        keep = set(json.load(open(args.keys)))
        groups = {g: [a for a in v if f"{a['task']}:{a['id']}" in keep] for g, v in groups.items()}
    print(json.dumps(vars(args)), {g: len(v) for g, v in groups.items()}, flush=True)

    qa = KFSModel(model_name=args.model_path, device="cuda:0", max_new_tokens=args.max_new_tokens)
    need_clip = args.select.startswith("clip") or args.recall or args.cascade
    scorer = ClipScorer(args.clip_path, "cuda:0", args.cache_dir) if need_clip else None

    os.makedirs(args.result_dir, exist_ok=True)
    ckpt = os.path.join(args.result_dir, "results_incremental.jsonl")
    records, _ = load_jsonl_results(ckpt)
    done = {f"{r['task']}:{r['id']}" for r in records}
    results = {g: [r for r in records if (r["task"] in BACKWARD_TASKS) == (g == "backward")] for g in groups}
    with open(ckpt, "a") as fh:
        for g, items in groups.items():
            for anno in tqdm(items, desc=g, mininterval=30):
                key = f"{anno['task']}:{anno['id']}"
                if key in done:
                    continue
                r = (evaluate_cascade if args.cascade else evaluate_one)(args, qa, scorer, anno)
                results[g].append(r)
                fh.write(json.dumps({**r, "_key": key}, ensure_ascii=False) + "\n")
                fh.flush()

    summary = calculate_ovo_scores(results["backward"], results["realtime"], [])
    bt = float(np.mean([v["accuracy"] for v in summary["backward"].values()]))
    rt = float(np.mean([v["accuracy"] for v in summary["realtime"].values()]))
    metrics = {"bt": bt, "rt": rt, "avg": (bt + rt) / 2}
    for sec in ("backward", "realtime"):
        metrics.update({t: v["accuracy"] for t, v in summary[sec].items()})
    allr = results["backward"] + results["realtime"]
    metrics["vis_tokens_mean"] = float(np.mean([r["num_vision_tokens"] for r in allr]))
    metrics["select_time_mean"] = float(np.mean([r["select_time"] for r in allr]))
    metrics["timeline_tokens_mean"] = float(np.mean([r.get("timeline_tokens", 0) for r in allr]))
    if args.cascade:
        metrics.update({f"l{i}_rate": float(np.mean([r["level"] == i for r in allr])) for i in (1, 2)})
    print(json.dumps({k: round(v, 2) for k, v in metrics.items()}, indent=1), flush=True)
    json.dump({"config": vars(args), "metrics": metrics, "summary": summary}, open(os.path.join(args.result_dir, "summary.json"), "w"), indent=1)
    rxlib.write_result(metrics, selection="final", n_eval=1, protocol="ovo-rtbt")


if __name__ == "__main__":
    main()
