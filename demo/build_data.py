"""ReadPast demo data (Qwen3-VL-8B, OVO-Bench): rebuild the final method per question from the run records and write
demo/data.json for the 6 showcase questions + the summary numbers. Host stdlib; run from /data/REN/streamkv.

ReadPast = SimpleStream recent-4 view + 16 s self-written caption timeline, read only when the calibrated gate says
"past" (p_past >= p_cf, route_q3_cal_p0_lo) + conditional abstention instruction (cinstr runs, abstention-option Qs).
"""
import glob, json, math, sys
sys.path.insert(0, "kfs")
from analyze_pilot import load, scores, BACKWARD, REALTIME  # noqa: E402

RUNS = "/data/REN/runs/streamkv"
CAPS = "cache/captions_q3_s16"
ANNO = "/data/REN/datasets/streaming/ovo_bench/ovo_bench_new.json"
EXAMPLES = ["EPM:173", "ASI:562", "ASI:568", "OCR:1359", "ATR:1030", "EPM:253", "EPM:82"]


def raw(name):
    p = sorted(glob.glob(f"{RUNS}/*-{name}/out/results_incremental.jsonl"))[-1]
    return {(r := json.loads(l))["_key"]: r for l in open(p)}


def popts(r):
    return r["p_opts"] if isinstance(r["p_opts"], list) else json.loads(r["p_opts"])


def argmax_ok(r):
    p = popts(r)
    return int(chr(65 + max(range(len(p)), key=p.__getitem__)) == r["ground_truth"])


base, mem = load("r4_q3", RUNS), load("cap16_q3", RUNS)
cb, cm = raw("r4_cinstr_q3"), raw("cap16_cinstr_q3")
rdir = sorted(glob.glob(f"{RUNS}/*-route_q3_cal_p0_lo/out"))[-1]
route, p_cf = json.load(open(f"{rdir}/route.json")), json.load(open(f"{rdir}/calib.json"))["p_cf"]
special = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
keys = sorted(base)
tasks = {k: base[k]["task"] for k in keys}
ga = {k: route[k]["p_past"] >= p_cf for k in keys}


def ours_rec(k):
    if k in special:
        return cm[k] if ga[k] else cb[k]
    return mem[k] if ga[k] else base[k]


ss = {k: base[k]["correct"] for k in keys}
always = {k: mem[k]["correct"] for k in keys}
gated = {k: (mem if ga[k] else base)[k]["correct"] for k in keys}
ours = {k: argmax_ok(ours_rec(k)) if k in special else gated[k] for k in keys}
always_ci = {k: argmax_ok(cm[k]) if k in special else always[k] for k in keys}  # always read + cond. instruction


def summ(c):
    s = scores(c, tasks)
    return {"avg": round(s["avg"], 2), "bt": round(s["bt"], 2), "rt": round(s["rt"], 2)}


bt = [k for k in keys if tasks[k] in BACKWARD]
rt = [k for k in keys if tasks[k] in REALTIME]
summary = {
    "n": len(keys),
    "simplestream": summ(ss), "always": summ(always_ci), "readpast": summ(ours),
    "read_bt": round(100 * sum(ga[k] for k in bt) / len(bt), 1),
    "read_rt": round(100 * sum(ga[k] for k in rt) / len(rt), 1),
    "lo_cf": round(math.log(p_cf / (1 - p_cf)), 2),
}
print(json.dumps(summary))

annos = {f"{a['task']}:{a['id']}": a for a in json.load(open(ANNO))}
out = []
for k in EXAMPLES:
    a, r = annos[k], base[k]
    q = r["query_ts"]
    caps = json.load(open(f"{CAPS}/" + a["video"].replace("/", "__") + ".json"))["captions"]
    tl = [{"t0": c["t0"], "t1": c["t1"], "text": c["text"].strip()} for c in caps if c["t1"] <= q + 1e-6]
    letter = lambda rec: rec["response"].strip()[:1]  # noqa: E731
    out.append({
        "key": k, "task": a["task"], "id": a["id"], "video": a["video"], "query_ts": round(q, 2),
        "question": a["question"].strip(), "options": a["options"], "gt": a["gt"],
        "lo": route[k]["lo"], "gate_past": ga[k], "special": k in special,
        "ss": {"p": [round(x, 4) for x in popts(base[k])], "pick": letter(base[k]), "ok": ss[k]},
        "always": {"p": [round(x, 4) for x in popts(mem[k])], "pick": letter(mem[k]), "ok": always[k]},
        "ours": {"p": [round(x, 4) for x in popts(ours_rec(k))], "ok": ours[k]},
        "timeline": tl,
        "thumbs": [round((c["t0"] + c["t1"]) / 2, 2) for c in tl],
        "recent": [max(0, math.floor(q) - 3 + i) for i in range(4)],
    })
json.dump({"summary": summary, "examples": out}, open("demo/data.json", "w"), indent=1, ensure_ascii=False)
print("wrote demo/data.json", [(e["key"], e["ss"]["ok"], e["always"]["ok"], e["ours"]["ok"], e["gate_past"]) for e in out])
