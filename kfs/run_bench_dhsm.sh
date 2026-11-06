#!/bin/bash
# Round 19: D-HSM timing with the round-18 settings (same arguments as kfs/run_dhsm.sh dhsm uniform_abstention),
# through the kfs/bench_dhsm.py timer wrapper (D-HSM code unchanged). usage (via rx): bash kfs/run_bench_dhsm.sh
set -e
source /workspace/streamkv/envs/dhsm/bin/activate
export FORCE_QWENVL_VIDEO_READER=torchcodec TORCHCODEC_NUM_THREADS=${TORCHCODEC_NUM_THREADS:-2} MIN_PIXELS=50176 MAX_PIXELS=262144
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
cd /workspace/streamkv/third_party/D-HSM
python /workspace/streamkv/kfs/bench_dhsm.py \
  --model_path "${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}" --embed_model BAAI/bge-small-en-v1.5 \
  --attn_implementation flash_attention_2 --recent_frames_only 4 --chunk_duration 1.0 --fps 1.0 --max_qa_tokens 256 \
  --extract_every_n_chunks 1 --max_extraction_chunks 20 --sim_threshold 0.25 --top_k dynamic --caption_batch_size 4 \
  --routing keyword --memory_mode entity_resolved \
  --anno_path "/workspace/streamkv/${ANNO:-kfs/data/bench_anno_r19.json}" \
  --chunked_dir /workspace/datasets/streaming/ovo_bench/chunked_videos \
  --result_dir "${RX_RUN_DIR:-/workspace/streamkv/results/bench_dhsm_manual}/out" \
  --history_mode dhsm --mcq_prompt_policy uniform_abstention --splits backward,realtime "$@"
