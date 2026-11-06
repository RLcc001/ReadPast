"""Round 19: fixed question sample for the latency benchmark (ours and D-HSM use the same keys).
5 warm-up questions + 100 measured = BT 50 + RT 50, each split over its tasks proportionally (largest remainder),
random.Random(19). Writes kfs/data/bench_keys_r19.json (list of "<task>:<id>") and kfs/data/bench_anno_r19.json
(the same questions as OVO annotation dicts, for D-HSM's --anno_path).
usage: python3 kfs/make_bench_keys_r19.py
"""
import json, random
from collections import defaultdict

BT = ["EPM", "ASI", "HLD"]
RT = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
annos = [a for a in json.load(open("/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json")) if a["task"] in BT + RT]
by = defaultdict(list)
for a in annos:
    by[a["task"]].append(a)
rng = random.Random(19)
for t in by:
    rng.shuffle(by[t])

def alloc(tasks, n):
    tot = sum(len(by[t]) for t in tasks)
    q = {t: n * len(by[t]) / tot for t in tasks}
    k = {t: int(q[t]) for t in tasks}
    for t in sorted(tasks, key=lambda t: -(q[t] - k[t]))[: n - sum(k.values())]:
        k[t] += 1
    return k

take = {**alloc(BT, 50), **alloc(RT, 50)}
measured = [a for t in BT + RT for a in by[t][: take[t]]]
rng.shuffle(measured)
used = {id(a) for a in measured}
warm = [a for a in rng.sample(annos, 20) if id(a) not in used][:5]
sel = warm + measured
json.dump([f"{a['task']}:{a['id']}" for a in sel], open("kfs/data/bench_keys_r19.json", "w"), indent=0)
json.dump(sel, open("kfs/data/bench_anno_r19.json", "w"), indent=0)
print(take, len(sel), [f"{a['task']}:{a['id']}" for a in warm])
