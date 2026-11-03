#!/bin/bash
# Round 18: run the public D-HSM code (third_party/D-HSM, commit 04579a0) on StreamingBench Real-Time inside the
# container, with the settings of its scripts/run_eval.sh (COMMON args, SB profile auto-selected from the model
# name; single process). Annotation = our local questions_real.json (or a per-video shard of it).
# usage (via rx): ANNO=kfs/data/sb_....json MODEL_PATH=... bash kfs/run_dhsm_sb.sh [extra evaluate_streamingbench.py args]
set -e
source /workspace/streamkv/envs/dhsm/bin/activate
export FORCE_QWENVL_VIDEO_READER=torchcodec TORCHCODEC_NUM_THREADS=${TORCHCODEC_NUM_THREADS:-2} MIN_PIXELS=50176 MAX_PIXELS=262144
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /workspace/streamkv/third_party/D-HSM
python experiments/evaluate_streamingbench.py \
  --model_path "${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}" --embed_model BAAI/bge-small-en-v1.5 \
  --attn_implementation flash_attention_2 --recent_frames_only 4 --chunk_duration 1.0 --fps 1.0 --max_qa_tokens 256 \
  --extract_every_n_chunks 1 --max_extraction_chunks 20 --sim_threshold 0.25 --top_k dynamic --caption_batch_size 4 \
  --routing keyword --memory_mode entity_resolved \
  --anno_path "/workspace/streamkv/${ANNO:-kfs/data/sb_smoke_r18.json}" \
  --video_dir /workspace/datasets/streaming/streamingbench/videos \
  --result_dir "${RX_RUN_DIR:-/workspace/streamkv/results/dhsm_sb_manual}/out" "$@"
