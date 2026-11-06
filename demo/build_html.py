"""Embed demo/data.json + demo/frames/*.jpg into demo/template.html -> demo/readpast_demo.html (single file).
usage (host): python3 demo/build_html.py"""
import base64, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
imgs = {f: "data:image/jpeg;base64," + base64.b64encode(open(os.path.join(HERE, "frames", f), "rb").read()).decode()
        for f in sorted(os.listdir(os.path.join(HERE, "frames"))) if f.endswith(".jpg")}
html = open(os.path.join(HERE, "template.html")).read()
html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__IMG__", json.dumps(imgs))
out = os.path.join(HERE, "readpast_demo.html")
open(out, "w").write(html)
print(out, f"{len(html) / 1e6:.2f} MB", len(imgs), "images")
