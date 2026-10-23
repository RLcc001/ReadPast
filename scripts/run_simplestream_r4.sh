#!/bin/bash
# SimpleStream repro: Qwen2.5-VL-7B, recent 4 frames, 1fps. OVO on host GPU4 (container 0), StreamingBench-RT on host GPU5 (container 1).
set -e
cd /workspace/streamkv/third_party/SimpleStream
source /workspace/streamkv/envs/simplestream-qwen25/bin/activate
LOG=/workspace/streamkv/logs; RES=/workspace/streamkv/results; D=/workspace/datasets/streaming
mkdir -p $LOG $RES
M=Qwen/Qwen2.5-VL-7B-Instruct
# decord + exact-tail decode: same frames/resolution as full-prefix fetch_video, without decoding from t=0
export FORCE_QWENVL_VIDEO_READER=decord QWEN_EXACT_RECENT_DECODE=1
python -c "from huggingface_hub import snapshot_download as s; print(s('$M'))" 2>&1 | tail -1 | tee $LOG/dl_qwen25vl.log
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 --main_process_port 29511 main_experiments/eval_qwen25vl_ovo.py \
  --model_path $M --anno_path $D/ovo_bench/ovo_bench_new.json --chunked_dir $D/ovo_bench/chunked_videos \
  --result_dir $RES/ovo_qwen25vl_recent4 --recent_frames_only 4 --chunk_duration 1.0 --fps 1.0 \
  > $LOG/ss_ovo_qwen25vl_r4.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main_experiments/eval_streamingbench.py \
  --qa-model $M --anno-path $D/streamingbench/questions_real.json --video-dir $D/streamingbench/videos \
  --top-k 0 --recent-frames-only 4 --chunk-duration 1.0 --fps 1.0 \
  --output-dir $RES/sb_qwen25vl_recent4 > $LOG/ss_sb_qwen25vl_r4.log 2>&1 &
wait
echo "==== both done $(date)"
