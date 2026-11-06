# 运行环境冻结记录（2026-11-06）

主表所有 run 都在这套环境里跑。环境有改动（重建 venv、升级包、换权重）时重新导出并在 `STATUS.md` 决定记录里写一行。

## 容器与硬件
- 容器 `ren_stream_all`，镜像 `ren/stream:latest`，image id `sha256:6231134b2390d4b38eba02a8a488f5ad51bb900bc4959e54239acefa2be77047`（Dockerfile：`/data/REN/docker/stream`）。
- CUDA 12.8（nvcc `cuda_12.8.r12.8/compiler.35583870_0`），驱动 595.71.05，NVIDIA H100 80GB HBM3。
- 挂载：`/data/REN` → `/workspace`，`/data/dataset` → `/data/dataset`。

## venv（`envs/`，包列表见同目录 `<venv>.txt`，导出命令见下）
| venv | Python | 关键包 | 用于 |
|---|---|---|---|
| simplestream-qwen25 | 3.11.17 | torch 2.7.1+cu128, transformers 4.52.4, flash-attn 2.8.3, decord 0.6.0, qwen-vl-utils 0.0.14 | Qwen2.5-VL：SimpleStream、ReadPast 全部 |
| simplestream-qwen3 | 3.12.15 | torch 2.9.1+cu128, transformers 4.57.6, flash-attn 2.8.3（本地 wheel） | Qwen3-VL-8B / 4B、LLaVA-OV（`kfs/llava_ov.py`）；含 `zz_kfs_conv3d_fix.pth`（Conv3d 提速修补） |
| dhsm | 3.12.15 | transformers 5.5.0, sentence-transformers 6.1.0；**torch 等通过 `_qwen3_base.pth` 借用 simplestream-qwen3 的 site-packages** | D-HSM 同协议重跑 |
| hermes-qwen / hermes-llava | 3.12.15 | 见 txt | 早期 HERMES 复现，不进主表 |

导出：`docker exec -u ren ren_stream_all bash -c 'cd /workspace/streamkv && uv pip freeze --python envs/<venv>/bin/python > envs/locks/<venv>.txt'`

## 模型权重（HF cache `/data/REN/.cache/huggingface/hub`，revision = snapshot hash）
| 模型 | revision |
|---|---|
| Qwen/Qwen2.5-VL-7B-Instruct | cc594898137f460bfe9f0759e9844b3ce807cfb5 |
| Qwen/Qwen3-VL-8B-Instruct | 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b |
| Qwen/Qwen3-VL-4B-Instruct | ebb281ec70b05090aa6165b016eac8ec08e71b17 |
| llava-hf/llava-onevision-qwen2-7b-ov-hf | 0d50680527681998e456c7b78950205bedd8a068 |
| BAAI/bge-small-en-v1.5（D-HSM 检索） | 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a |
| openai/clip-vit-large-patch14-336（早期选帧轮次） | ce19dc912ca5cd21c8a653c79e251e808ccabcd1 |

## 数据（`/data/REN/datasets/streaming`，sha256 前 16 位）
| 文件 | sha256 |
|---|---|
| ovo_bench/ovo_bench_new.json | dc4a90daf26764dc |
| streamingbench/questions_real.json（由 `scripts/build_sb_questions.py` 从官方 CSV 生成） | 3e1b39360bf0ce58 |
