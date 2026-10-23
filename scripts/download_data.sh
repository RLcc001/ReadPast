#!/bin/bash
# 下载并解压流式视频基准（在 ren_stream 容器里运行）。
#   bash /workspace/streamkv/scripts/download_data.sh [ovo|sb]   不带参数则两个都做
# 下载可断点续传：重跑会跳过已下载完的文件。压缩包解压后保留在 _archives/，确认无误后再手动删除。
set -euo pipefail

ROOT=/workspace/datasets/streaming
HF=/workspace/streamkv/envs/simplestream-qwen25/bin/hf

ovo() {
  local d=$ROOT/ovo_bench
  mkdir -p $d/_archives
  # 标注在 GitHub 仓库里，HF 上只有视频
  curl -sSfL -o $d/ovo_bench_new.json https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/main/data/ovo_bench_new.json
  # chunked_videos：SimpleStream 用；src_videos：HERMES / ReKV 用（整段原视频）
  $HF download JoeLeelyf/OVO-Bench --repo-type dataset --local-dir $d/_archives --max-workers 8
  for p in chunked_videos src_videos; do
    if [ ! -f $d/.${p}_extracted ]; then
      echo "extracting $p ..."
      cat $d/_archives/$p.tar.part* | tar -x -C $d
      touch $d/.${p}_extracted
    fi
  done
}

sb() {
  local d=$ROOT/streamingbench
  mkdir -p $d/_archives $d/videos
  # 只下实时视觉理解部分（500 个视频，SimpleStream / HERMES 都只评这一部分）和全部标注 csv
  $HF download mjuicem/StreamingBench --repo-type dataset --local-dir $d/_archives --max-workers 8 \
    --include "Real-Time Visual Understanding_*.zip" "StreamingBench/*.csv"
  cp $d/_archives/StreamingBench/*.csv $d/
  for z in "$d/_archives/Real-Time Visual Understanding_"*.zip; do
    local tag=$d/.$(basename "$z" .zip | tr ' ' '_')_extracted
    if [ ! -f "$tag" ]; then
      echo "extracting $(basename "$z") ..."
      unzip -q -o "$z" -d $d/videos
      touch "$tag"
    fi
  done
}

for t in ${@:-ovo sb}; do
  echo "==== $t start $(date '+%F %T')"
  $t
  echo "==== $t done  $(date '+%F %T')"
done
