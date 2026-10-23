#!/bin/bash
# Run the OVO keyframe-selection eval inside ren_stream; outputs go to the rx run dir.
# usage (via rx): bash kfs/run.sh [eval_ovo_kfs.py args...]
set -e
source /workspace/streamkv/envs/simplestream-qwen25/bin/activate
export FORCE_QWENVL_VIDEO_READER=decord
cd /workspace/streamkv
python kfs/eval_ovo_kfs.py --result_dir "${RX_RUN_DIR:-results/kfs/manual}/out" "$@"
