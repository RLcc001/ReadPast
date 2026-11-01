"""LLaVA-OneVision backend with the KFSModel interface (kfs/eval_ovo_kfs.py), for the fourth backbone.

Same interface as KFSModel: encode_vision(frames) -> (embeds, n_frames), generate_segments(segments, question,
n_options), processor.tokenizer, model, _im_start_id / _im_end_id, max_new_tokens. Text-only scripts (route.py,
route_baselines.py, answer_prior.py) use `role_sep` (" " here, "\\n" for Qwen) after the role name and `end_sep`
("" here, "\\n" for Qwen) after <|im_end|>.

Vision follows the HF video path of llava-hf/llava-onevision-qwen2-7b-ov-hf (the path ShallowStream uses for its
SimpleStream rerun): each frame -> SigLIP (384x384) -> projector -> 2x2 bilinear pooling = 196 tokens; a vision
block (list of frames) is one "video": its frames' tokens followed by one image_newline token. With a single block
and no text segments, generate_segments reproduces the processor's chat template token for token:
"<|im_start|>user <video>\\n{question}<|im_end|><|im_start|>assistant\\n" (checked by kfs/test_llava_ov.py).
Frames arrive already resized by the shared (Qwen) decoding path; the video processor resizes them to 384x384.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from PIL import Image


class LlavaOVModel:
    role_sep = " "
    end_sep = ""  # HF template: "<|im_end|><|im_start|>assistant\n" (no newline in between)
    per_frame_blocks = False

    def __init__(self, model_name: str, device: str = "cuda:0", max_new_tokens: int = 256) -> None:
        from transformers import LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.processor = LlavaOnevisionProcessor.from_pretrained(model_name)
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device).eval()
        tok = self.processor.tokenizer
        self._im_start_id = tok.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.video_token_id = self.model.config.video_token_index
        self.cfg = self.model.config

    @torch.inference_mode()
    def encode_vision(self, frames: list[Image.Image]) -> tuple[torch.Tensor, int]:
        """Frames of one block -> (n_frames * 196, d) pooled features (no newline; added per block)."""
        px = self.processor.video_processor(videos=[np.stack([np.asarray(f.convert("RGB")) for f in frames])],
                                            return_tensors="pt")["pixel_values_videos"]
        feats = self.model.model.get_video_features(
            px.to(self.model.device, self.model.dtype),
            vision_feature_layer=self.cfg.vision_feature_layer,
            vision_feature_select_strategy=self.cfg.vision_feature_select_strategy,
        )
        return feats[0], len(frames)

    @torch.inference_mode()
    def generate_segments(self, segments: list, question: str, n_options: int) -> dict[str, Any]:
        tok = self.processor.tokenizer
        ids: list[int] = [self._im_start_id] + tok.encode("user" + self.role_sep, add_special_tokens=False)
        vis: list[torch.Tensor] = []
        newline = self.model.model.image_newline[None].to(self.model.dtype)
        for seg in segments:
            if isinstance(seg, str):
                ids += tok.encode(seg, add_special_tokens=False)
                continue
            block = torch.cat([e for e, _ in seg] + [newline.to(seg[0][0].device)], dim=0)
            ids += [self.video_token_id] * int(block.shape[0])
            vis.append(block)
        ids += tok.encode("\n" + question, add_special_tokens=False)
        ids += [self._im_end_id] + tok.encode(self.end_sep, add_special_tokens=False)
        ids += [self._im_start_id] + tok.encode("assistant\n", add_special_tokens=False)

        self._last_ids = ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.model.device)
        attention_mask = torch.ones_like(input_ids)
        embeds = self.model.get_input_embeddings()(input_ids)
        n_vis = 0
        if vis:
            v = torch.cat(vis, dim=0).to(embeds.device, embeds.dtype)
            n_vis = int(v.shape[0])
            mask = (input_ids == self.video_token_id).unsqueeze(-1).expand_as(embeds)
            embeds = embeds.masked_scatter(mask, v)

        t0 = time.perf_counter()
        out = self.model.generate(inputs_embeds=embeds, attention_mask=attention_mask, max_new_tokens=self.max_new_tokens,
                                  do_sample=False, output_logits=True, return_dict_in_generate=True,
                                  pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else self._im_end_id)
        gen_time = time.perf_counter() - t0
        seq = out.sequences[0]
        answer = self.processor.batch_decode([seq], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        logits = out.logits[0][0].float()
        letter_ids = [tok.encode(chr(65 + i), add_special_tokens=False)[0] for i in range(n_options)]
        p_full = torch.softmax(logits, dim=-1)
        p_opts = torch.softmax(logits[letter_ids], dim=-1)
        return {
            "answer": answer,
            "p_opts": [round(float(p), 5) for p in p_opts],
            "p_letters_mass": round(float(p_full[letter_ids].sum()), 5),
            "num_vision_tokens": n_vis,
            "generate_time": gen_time,
        }


def load_qa(model_path: str, device: str = "cuda:0", max_new_tokens: int = 256):
    """KFSModel for Qwen-VL checkpoints, LlavaOVModel for LLaVA-OneVision."""
    if "llava" in model_path.lower():
        return LlavaOVModel(model_path, device=device, max_new_tokens=max_new_tokens)
    from eval_ovo_kfs import KFSModel

    return KFSModel(model_name=model_path, device=device, max_new_tokens=max_new_tokens)
