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
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")

import numpy as np
import torch
from PIL import Image
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

VisionBlock = list[tuple[torch.Tensor, torch.Tensor]]  # [(embeds, grid_thw row)] for one <vision_start> block


class KFSModel(RecentWindowQAModel):
    @torch.inference_mode()
    def generate_segments(self, segments: list[str | VisionBlock], question: str, n_options: int) -> dict[str, Any]:
        """Build `<im_start>user\\n{segments}\\n{question}<im_end>\\n<im_start>assistant\\n` and generate.

        With a single vision block this is token-for-token SimpleStream's Qwen2.5 prompt. Also returns
        the first-step distribution over the option letters (raw logits) as a confidence signal.
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
            n_tok = sum(int(e.shape[0]) for e, _ in seg)
            ids += [self._vision_start_id] + [self.image_token_id] * n_tok + [self._vision_end_id]
            embeds += [e for e, _ in seg]
            # get_rope_index consumes one grid row per vision block (SimpleStream relies on the same behaviour)
            block_grids.append(seg[0][1])
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
    blocks: VisionBlock = []
    for fr in frames:
        emb, grid = qa.encode_vision([to_pil(fr)])
        blocks.append((emb, grid))

    hist_ts = [i / plan.raw_fps for i in chosen]
    segments = build_segments(args.layout, blocks[:len(chosen)], hist_ts, blocks[len(chosen):], len(recent))
    out = qa.generate_segments(segments, build_qwen25_prompt(anno["task"], anno), len(anno["options"]))
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
        "num_frames": len(blocks),
        "num_vision_tokens": out["num_vision_tokens"],
        "select_time": select_time,
        "generate_time": out["generate_time"],
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
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--clip_path", default="/workspace/models/clip-vit-large-patch14-336-st")
    ap.add_argument("--cache_dir", default="/workspace/streamkv/cache/clip_l336_ovo")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N questions per task group (smoke tests)")
    args = ap.parse_args()

    annos = json.load(open(args.anno_path))
    groups = {
        "backward": [a for a in annos if a["task"] in BACKWARD_TASKS],
        "realtime": [a for a in annos if a["task"] in REAL_TIME_TASKS],
    }
    if args.limit:
        groups = {g: v[: args.limit] for g, v in groups.items()}
    print(json.dumps(vars(args)), {g: len(v) for g, v in groups.items()}, flush=True)

    qa = KFSModel(model_name=args.model_path, device="cuda:0", max_new_tokens=args.max_new_tokens)
    scorer = ClipScorer(args.clip_path, "cuda:0", args.cache_dir) if args.select.startswith("clip") else None

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
                r = evaluate_one(args, qa, scorer, anno)
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
    print(json.dumps({k: round(v, 2) for k, v in metrics.items()}, indent=1), flush=True)
    json.dump({"config": vars(args), "metrics": metrics, "summary": summary}, open(os.path.join(args.result_dir, "summary.json"), "w"), indent=1)
    rxlib.write_result(metrics, selection="final", n_eval=1, protocol="ovo-rtbt")


if __name__ == "__main__":
    main()
