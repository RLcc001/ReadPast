# Third-party code

`third_party/` is not tracked (see `.gitignore`). The experiments used the upstream repositories below at these exact commits.
Only SimpleStream carries a local change; the others are unmodified.

| Repo | Upstream | Commit | Local change |
|---|---|---|---|
| SimpleStream | https://github.com/EvolvingLMMs-Lab/SimpleStream | `be227ee8ce90a258b286a34161688c2a59eeeadb` | `simplestream_ren_dev.patch` (branch `ren/dev`, commit `9440b09`) |
| D-HSM | https://github.com/Oshikaka/D-HSM | `04579a052f0a435b1eec52dab00cd7de759fd599` | none |
| ShallowStream | https://github.com/CURRENTF/ShallowStream | `097550663fd60d7486dc99ea0a7ba251eae79f77` | none (read only; paper values used) |
| HERMES | https://github.com/haowei-freesky/HERMES | `8d699b16a6bedb9086c1b39ec4253c6a1d1ce789` | none |
| InfiniPot-V | https://github.com/aiha-lab/InfiniPot-V | `81a4dbe2e74660a8148cc2a6cee1148a9a268479` | none |
| LiveVLM | https://github.com/sjtu-zhao-lab/LiveVLM | `8b84be84ea3c21c9cdda4f2bc1009a54c51e0fc0` | none |
| rekv | https://github.com/becomebright/rekv | `1fd9a3dbf5dbff7f27069ae2f4463674c495e830` | none |

## SimpleStream patch

Upstream `main_experiments/eval_streamingbench.py` keys StreamingBench results by video + timestamp + the first 80 characters of the question. Questions that share all three (e.g. Counting) collide, so 139 questions are silently skipped as already done. The patch keys results by `question_id` when it is present. All StreamingBench numbers in `RESULTS.md` use the patched script.

```bash
git clone https://github.com/EvolvingLMMs-Lab/SimpleStream third_party/SimpleStream
cd third_party/SimpleStream
git checkout be227ee8ce90a258b286a34161688c2a59eeeadb
git checkout -b ren/dev
git am ../../patches/simplestream_ren_dev.patch
```

The official SimpleStream scripts also need `FORCE_QWENVL_VIDEO_READER=decord QWEN_EXACT_RECENT_DECODE=1`.

## Other repositories

```bash
git clone https://github.com/Oshikaka/D-HSM third_party/D-HSM && git -C third_party/D-HSM checkout 04579a052f0a435b1eec52dab00cd7de759fd599
git clone https://github.com/CURRENTF/ShallowStream third_party/ShallowStream && git -C third_party/ShallowStream checkout 097550663fd60d7486dc99ea0a7ba251eae79f77
git clone https://github.com/haowei-freesky/HERMES third_party/HERMES && git -C third_party/HERMES checkout 8d699b16a6bedb9086c1b39ec4253c6a1d1ce789
git clone https://github.com/aiha-lab/InfiniPot-V third_party/InfiniPot-V && git -C third_party/InfiniPot-V checkout 81a4dbe2e74660a8148cc2a6cee1148a9a268479
git clone https://github.com/sjtu-zhao-lab/LiveVLM third_party/LiveVLM && git -C third_party/LiveVLM checkout 8b84be84ea3c21c9cdda4f2bc1009a54c51e0fc0
git clone https://github.com/becomebright/rekv third_party/rekv && git -C third_party/rekv checkout 1fd9a3dbf5dbff7f27069ae2f4463674c495e830
```
