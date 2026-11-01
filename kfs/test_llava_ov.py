"""Check kfs/llava_ov.py against the HF processor + model forward on one OVO question (4 recent frames).

1. token ids of generate_segments([block], q) == processor(apply_chat_template(video + text)) input_ids;
2. per-frame encode_vision == one 4-frame video through the processor;
3. first-step letter distribution of generate_segments == full-model forward logits.

usage: python kfs/test_llava_ov.py [--model_path llava-hf/llava-onevision-qwen2-7b-ov-hf]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ovo_kfs import _build_sampling_plan, _decode_indices, _resize_like_fetch_video, build_qwen25_prompt, to_pil  # noqa: E402
from llava_ov import LlavaOVModel  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model_path", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
ap.add_argument("--anno_path", default="/workspace/datasets/streaming/ovo_bench/ovo_bench_new.json")
ap.add_argument("--chunked_dir", default="/workspace/datasets/streaming/ovo_bench/chunked_videos")
args = ap.parse_args()

qa = LlavaOVModel(args.model_path, max_new_tokens=1)
annos = [a for a in json.load(open(args.anno_path)) if a["task"] in ("EPM", "OCR", "HLD")][:3]
for anno in annos:
    vp = os.path.join(args.chunked_dir, f"{anno['id']}.mp4")
    ele = {"video": vp, "fps": 1.0}
    plan = _build_sampling_plan(ele, last_nframes=4)
    recent = plan.full_indices[-4:]
    frames = [to_pil(f) for f in _resize_like_fetch_video(_decode_indices(vp, plan.backend, recent), ele,
                                                          plan.sampled_nframes_full, image_patch_size=14)]
    q = build_qwen25_prompt(anno["task"], anno)
    block = [qa.encode_vision([f]) for f in frames]
    out = qa.generate_segments([block], q, len(anno["options"]))

    msgs = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": q}]}]
    text = qa.processor.apply_chat_template(msgs, add_generation_prompt=True)
    inp = qa.processor(text=text, videos=[np.stack([np.asarray(f) for f in frames])], return_tensors="pt").to(qa.model.device)
    ref_ids = inp["input_ids"][0].tolist()
    same_ids = ref_ids == qa._last_ids
    with torch.inference_mode():
        joint = qa.model.model.get_video_features(inp["pixel_values_videos"].to(qa.model.dtype), qa.cfg.vision_feature_layer,
                                                  qa.cfg.vision_feature_select_strategy)[0]
        sep = torch.cat([e for e, _ in block], 0)
        logits = qa.model(**inp, use_cache=False).logits[0, -1].float()
    tok = qa.processor.tokenizer
    letters = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(len(anno["options"]))]
    p_ref = torch.softmax(logits[letters], -1).tolist()
    print(json.dumps({"key": f"{anno['task']}:{anno['id']}", "template": text[:60] + " ... " + text[-40:], "ids_equal": same_ids,
                      "n_ids": [len(ref_ids), len(qa._last_ids)], "frame_feat_maxdiff": float((joint - sep).abs().max()), "frame_feat_relerr": float((joint - sep).float().norm() / joint.float().norm()),
                      "p_mine": out["p_opts"], "p_ref": [round(p, 5) for p in p_ref], "answer": out["answer"],
                      "gt": chr(65 + anno["gt"]), "vis_tokens": out["num_vision_tokens"]}), flush=True)

# 4. Is the per-frame vs joint difference bf16 noise? Compare in float32 and joint-bf16 vs joint-fp32.
with torch.inference_mode():
    px = inp["pixel_values_videos"]
    mm = qa.model.model
    f = lambda dtype: mm.get_video_features(px.to(dtype), qa.cfg.vision_feature_layer, qa.cfg.vision_feature_select_strategy)[0].float()  # noqa: E731
    one = lambda dtype: torch.cat([mm.get_video_features(px[:, i:i + 1].to(dtype), qa.cfg.vision_feature_layer,  # noqa: E731
                                                          qa.cfg.vision_feature_select_strategy)[0].float() for i in range(px.shape[1])])
    jb = f(torch.bfloat16)
    mm.vision_tower.float(); mm.multi_modal_projector.float()
    j32, s32 = f(torch.float32), one(torch.float32)
    mm.vision_tower.to(torch.bfloat16); mm.multi_modal_projector.to(torch.bfloat16)
    rel = lambda a, b: float((a - b).norm() / b.norm())  # noqa: E731
    print(json.dumps({"joint_bf16_vs_joint_fp32": rel(jb, j32), "perframe_fp32_vs_joint_fp32": rel(s32, j32)}), flush=True)
