"""Frames for the ReadPast demo (CPU, container): timeline thumbnails (segment midpoints) + the recent-4 view.
usage: docker exec ren_stream_all /workspace/streamkv/envs/simplestream-qwen3/bin/python /workspace/streamkv/demo/extract_frames.py"""
import json, os
import decord
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CHUNKS = "/workspace/datasets/streaming/ovo_bench/chunked_videos"
data = json.load(open(os.path.join(HERE, "data.json")))
os.makedirs(os.path.join(HERE, "frames"), exist_ok=True)
for ex in data["examples"]:
    vr = decord.VideoReader(os.path.join(CHUNKS, f"{ex['id']}.mp4"), num_threads=2)
    fps, n = vr.get_avg_fps(), len(vr)
    for kind, ts, width in (("tl", ex["thumbs"], 400), ("now", ex["recent"], 560)):
        for t in ts:
            im = Image.fromarray(vr[min(int(round(t * fps)), n - 1)].asnumpy())
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            im.save(os.path.join(HERE, "frames", f"{ex['task']}_{ex['id']}_{kind}_{t:g}.jpg"), quality=78, optimize=True)
    print(ex["key"], f"{n / fps:.1f}s", len(ex["thumbs"]), "thumbs")
