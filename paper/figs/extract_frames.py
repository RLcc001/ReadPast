"""Extract example frames for figure 2 (CPU only): python extract_frames.py <video> <out_dir> <t1,t2,...> [width]"""
import os
import sys

import decord
from PIL import Image

video, out, ts = sys.argv[1], sys.argv[2], [float(t) for t in sys.argv[3].split(",")]
width = int(sys.argv[4]) if len(sys.argv) > 4 else 320
vr = decord.VideoReader(video, num_threads=2)
fps = vr.get_avg_fps()
os.makedirs(out, exist_ok=True)
for t in ts:
    im = Image.fromarray(vr[min(int(round(t * fps)), len(vr) - 1)].asnumpy())
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(os.path.join(out, f"t{t:06.2f}.jpg"), quality=92)
    print(t, im.size)
