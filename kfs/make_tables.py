"""Canonical ReadPast result tables: the single source of truth for current numbers (11-06).

Recomputes every cell from raw runs and writes
  results/main_tables.json  (all backbones x methods x prompts x benchmarks, per task)
  RESULTS.md                (human-readable tables; generated, do not edit by hand)
and checks the anchors already recorded in EXPERIMENTS.md (fails loudly on drift).

Protocol (decisions in STATUS.md):
  OVO standard = (RT + BT) / 2, RT = mean of 6 real-time tasks, BT = mean(EPM, ASI, HLD).
  OVO w/o HLD  = (RT + (EPM + ASI) / 2) / 2.
  Prompt "shared" = conditional abstention instruction (D-HSM text) only on the 187 OVO MCQs that offer an
  abstention option (186 HLD + 1 EPM); "official" = SimpleStream prompt. SB has no abstention option -> identical.
  ReadPast = 16 s caption timeline + recent 4 frames + gate (a) (content-free calibrated p_past >= p_cf).
  D-HSM = public code rerun on our data/scorer (its default uniform instruction on every MCQ), round 18.
  Paper rows (D-HSM, ShallowStream) = values copied from the papers, own protocols; not comparable cell by cell.

Inputs: ONLY the run ids listed in kfs/data/canonical_runs.json (no name globbing). Replacing a run there is a
recorded decision (STATUS.md). A newer finished run with the same name as a listed one is reported as a warning.

usage (repo root, host python, stdlib only): python3 kfs/make_tables.py
"""
from __future__ import annotations

import datetime
import glob
import json
import math
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "kfs")
from analyze_pilot import mcq, scores  # noqa: E402

MANIFEST = json.load(open("kfs/data/canonical_runs.json"))
RUNS = MANIFEST["runs_root"]
BB = {  # short id -> (unused, unused, display name); run ids come from MANIFEST["ours"][id]
    "q25": (None, None, "Qwen2.5-VL-7B"),
    "q3": (None, None, "Qwen3-VL-8B"),
    "q4": (None, None, "Qwen3-VL-4B"),
    "lov": (None, None, "LLaVA-OV-7B"),
}
OVO_RT = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
OVO_BT = ["EPM", "ASI", "HLD"]
SB_TASKS = {"OP": "Object Perception", "CR": "Causal Reasoning", "CS": "Clips Summarize", "ATP": "Attribute Perception",
            "EU": "Event Understanding", "TR": "Text-Rich Understanding", "PR": "Prospective Reasoning",
            "SU": "Spatial Understanding", "ACP": "Action Perception", "CT": "Counting"}
SB_ABBR = {v: k for k, v in SB_TASKS.items()}
RULES = ["SimpleStream", "ReadPast", "always read", "fixed 0.5", "ShallowStream gate + cf"]
ABST_KEYS = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))

PAPER = [  # (backbone, method, ovo dict, sb dict); copied from the papers (RELATED.md, EXPERIMENTS.md 11-05)
    ("q25", "D-HSM (paper)", dict(OCR=94.0, ACR=70.7, ATR=86.2, STU=66.9, FPD=75.3, OJR=81.0, EPM=52.9, ASI=63.5, HLD=72.0),
     dict(OP=85.3, CR=64.0, CS=88.8, ATP=87.9, EU=79.1, TR=90.0, PR=87.6, SU=79.7, ACP=79.6, CT=47.8, overall=82.5)),
    ("q3", "D-HSM (paper)", dict(OCR=95.0, ACR=82.6, ATR=84.5, STU=72.5, FPD=71.3, OJR=83.7, EPM=54.6, ASI=62.2, HLD=67.2),
     dict(OP=86.9, CR=64.8, CS=92.1, ATP=89.2, EU=74.7, TR=90.7, PR=85.7, SU=77.6, ACP=79.6, CT=54.5, overall=83.1)),
    ("q3", "ShallowStream (paper)", dict(RT=80.9, EPM=52.5, ASI=72.3, HLD=49.5), dict(overall=78.2)),
    ("lov", "ShallowStream (paper)", dict(RT=75.4, EPM=50.2, ASI=50.7, HLD=46.2), dict(overall=75.5)),
]
ANCHORS = {  # (bb, method, prompt, field) -> value recorded in EXPERIMENTS.md (rounds 17-18, 11-05)
    **{(b, "ReadPast", "shared", "Avg"): v for b, v in zip(BB, (70.78, 73.34, 72.60, 61.60))},
    **{(b, "SimpleStream", "shared", "Avg"): v for b, v in zip(BB, (68.88, 71.87, 71.48, 60.32))},
    **{(b, "ReadPast", "official", "Avg"): v for b, v in zip(BB, (67.02, 70.39, 68.08, 60.07))},
    **{(b, "SimpleStream", "official", "Avg"): v for b, v in zip(BB, (64.75, 67.66, 67.42, 59.15))},
    **{(b, "ReadPast", "shared", "SB"): v for b, v in zip(BB, (79.56, 81.64, 81.40, 74.92))},
    ("q25", "D-HSM", "own default", "Avg"): 67.73, ("q3", "D-HSM", "own default", "Avg"): 71.14,
    ("q25", "D-HSM", "own default", "SB"): 80.51, ("q3", "D-HSM", "own default", "SB"): 81.91,
}


def ovo_summary(per: dict) -> dict:
    out = dict(per)
    if "RT" not in out:
        out["RT"] = sum(per[t] for t in OVO_RT) / len(OVO_RT)
    out["BT"] = sum(per[t] for t in OVO_BT) / 3
    out["BT_noHLD"] = (per["EPM"] + per["ASI"]) / 2
    out["Avg"] = (out["RT"] + out["BT"]) / 2
    out["Avg_noHLD"] = (out["RT"] + out["BT_noHLD"]) / 2
    return out


def ovo_from_correct(c: dict, tasks: dict) -> dict:
    s = scores(c, tasks)
    return ovo_summary({t: s[t] for t in OVO_RT + OVO_BT})


def odir(run_id):
    return f"{RUNS}/{run_id}/out"


def raw(run_id):
    """Per-question records of one run (keyed by _key)."""
    return {(r := json.loads(line))["_key"]: r for line in open(f"{odir(run_id)}/results_incremental.jsonl")}


def load_gen(run_id):
    """Like analyze_pilot.load: correctness from the generated answer."""
    rows = raw(run_id)
    for r in rows.values():
        r["correct"] = int(mcq(r["response"]) == r["ground_truth"])
    return rows


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def gates(m, bench):
    """Same rules as analyze_round9.gates: (a) p_past >= p_cf; (b) ShallowStream 10-shot logit gate, cf-calibrated."""
    route = json.load(open(f"{odir(m['route_' + bench])}/route.json"))
    p_cf = json.load(open(f"{odir(m['route_' + bench])}/calib.json"))["p_cf"]
    sc = json.load(open(f"{odir(m['routers'])}/scores_{bench}.json"))
    cf = json.load(open(f"{odir(m['routers'])}/scores_cf.json"))
    p_ss = sum(sigmoid(v["ss_fs10"]) for v in cf.values()) / len(cf)
    return dict(a={k: v["p_past"] >= p_cf for k, v in route.items()},
                b={k: sigmoid(v["ss_fs10"]) >= p_ss for k, v in sc.items()},
                fixed={k: v["p_past"] >= 0.5 for k, v in route.items()}), p_cf


def argmax_ok(r):
    p = r["p_opts"] if isinstance(r["p_opts"], list) else json.loads(r["p_opts"])
    return int(chr(65 + max(range(len(p)), key=p.__getitem__)) == r["ground_truth"])


def ours_ovo(m):
    base, mem = load_gen(m["ovo_recent"]), load_gen(m["ovo_timeline"])
    cb, cm = raw(m["ovo_recent_instr"]), raw(m["ovo_timeline_instr"])
    keys = sorted(set(base) & set(mem))
    tasks = {k: base[k]["task"] for k in keys}
    gs, p_cf = gates(m, "ovo")
    info = {"p_cf": p_cf}
    dec = {"SimpleStream": {k: False for k in keys}, "ReadPast": gs["a"], "always read": {k: True for k in keys},
           "fixed 0.5": gs["fixed"], "ShallowStream gate + cf": gs["b"]}
    out = {}
    for rn in RULES:
        g = dec[rn]
        for prompt in ("official", "shared"):
            c = {k: (argmax_ok(cm[k]) if g[k] else argmax_ok(cb[k])) if (prompt == "shared" and k in ABST_KEYS)
                 else (mem if g[k] else base)[k]["correct"] for k in keys}
            s = ovo_from_correct(c, tasks)
            s["read_RT"] = 100 * sum(g[k] for k in keys if tasks[k] in OVO_RT) / sum(tasks[k] in OVO_RT for k in keys)
            s["read_BT"] = 100 * sum(g[k] for k in keys if tasks[k] in OVO_BT) / sum(tasks[k] in OVO_BT for k in keys)
            s["n"] = len(keys)
            out[(rn, prompt)] = s
    return out, info["p_cf"]


def sb_summary(rows, ok):
    per = {}
    for ab, full in SB_TASKS.items():
        v = [ok(r) for r in rows if r["task_type"] == full]
        per[ab] = 100 * sum(v) / len(v)
    per["overall"] = 100 * sum(ok(r) for r in rows) / len(rows)
    per["n"] = len(rows)
    return per


def ours_sb(m):
    rows = {}
    for run_id in m["sb"]:  # shards, merged by _key in listed order
        rows.update(raw(run_id))
    rows = list(rows.values())
    qid = {r["_key"]: r["_key"].split("_", 1)[1] for r in rows}
    gs, _ = gates(m, "sb")
    dec = {"SimpleStream": None, "ReadPast": gs["a"], "always read": "all",
           "fixed 0.5": gs["fixed"], "ShallowStream gate + cf": gs["b"]}
    out = {}
    for rn in RULES:
        d = dec[rn]
        if d is None:
            read = lambda r: False  # noqa: E731
        elif d == "all":
            read = lambda r: True  # noqa: E731
        else:
            read = lambda r, d=d: r["timeline_tokens"] > 0 and d[qid[r["_key"]]]  # noqa: E731
        s = sb_summary(rows, lambda r, read=read: int(r["correct_l1"] if read(r) else r["correct_l0"]))
        s["read"] = 100 * sum(bool(read(r)) and r["timeline_tokens"] > 0 for r in rows) / len(rows) if d != "all" else \
            100 * sum(r["timeline_tokens"] > 0 for r in rows) / len(rows)
        out[rn] = s
    return out


def dhsm(m):
    """D-HSM rerun (same loaders as analyze_round18_dhsm, explicit run ids)."""
    rows = {}
    for run_id in m["ovo"]:
        res = json.load(open(sorted(glob.glob(f"{odir(run_id)}/hub_and_spoke_results_*.json"))[-1]))
        for split in ("backward", "realtime"):
            for r in res.get(split, []):
                rows[f"{r['task']}:{r['id']}"] = r
    keys = [k for k in rows if rows[k]["task"] in OVO_RT + OVO_BT]
    c = {k: int((mcq(rows[k].get("response")) or "") == rows[k]["ground_truth"]) for k in keys}
    ovo = ovo_from_correct(c, {k: rows[k]["task"] for k in keys})
    ovo["n"] = len(keys)
    sbr = {}
    for run_id in m["sb"]:
        for r in json.load(open(sorted(glob.glob(f"{odir(run_id)}/hub_and_spoke_streamingbench_*.json"))[-1]))["results"]:
            sbr[(r["video_path"], r["time_stamp"], r["task_type"], r["question"])] = r
    sb = sb_summary(list(sbr.values()), lambda r: int((mcq(r.get("response")) or "") == r["answer_gt"]))
    return ovo, sb


def check_manifest():
    """Every listed run must be finished; warn when a newer finished run with the same name exists."""
    ids = [v for grp in ("ours", "dhsm") for d in MANIFEST[grp].values() for x in d.values()
           for v in (x if isinstance(x, list) else [x])]
    warn = []
    for run_id in ids:
        meta = json.load(open(f"{RUNS}/{run_id}/meta.json"))
        if meta.get("status") != "done":
            sys.exit(f"manifest run not done: {run_id} ({meta.get('status')})")
        for d in sorted(glob.glob(f"{RUNS}/*-{meta['name']}")):
            other = os.path.basename(d)
            if other > run_id and json.load(open(f"{d}/meta.json")).get("status") == "done":
                warn.append(f"{run_id}: newer finished run with the same name: {other}")
    for w in warn:
        print("WARNING", w)
    return len(ids)


def build():
    recs = []
    for b, (_, _, disp) in BB.items():
        m = MANIFEST["ours"][b]
        ovo, p_cf = ours_ovo(m)
        sb = ours_sb(m)
        for rn in RULES:
            for prompt in ("official", "shared"):
                recs.append(dict(bb=b, backbone=disp, method=rn, prompt=prompt, source="ours",
                                 ablation=rn not in ("SimpleStream", "ReadPast"), ovo=ovo[(rn, prompt)], sb=sb[rn],
                                 p_cf=p_cf if rn == "ReadPast" else None))
        if b in MANIFEST["dhsm"]:
            o, s = dhsm(MANIFEST["dhsm"][b])
            recs.append(dict(bb=b, backbone=disp, method="D-HSM", prompt="own default", source="rerun (round 18)",
                             ablation=False, ovo=o, sb=s))
    for b, m, o, s in PAPER:
        recs.append(dict(bb=b, backbone=BB[b][2], method=m, prompt="paper", source="paper", ablation=False,
                         ovo=ovo_summary(o), sb=s))
    return recs


def get(recs, b, m, prompt):
    for r in recs:
        if r["bb"] == b and r["method"] == m and r["prompt"] == prompt:
            return r
    return None


def check_anchors(recs):
    bad = []
    for (b, m, prompt, f), v in ANCHORS.items():
        r = get(recs, b, m, prompt)
        x = r["sb"]["overall"] if f == "SB" else r["ovo"][f]
        if round(x, 2) != v:
            bad.append(f"{b} {m} {prompt} {f}: {x:.2f} != {v}")
    if bad:
        sys.exit("ANCHOR DRIFT:\n  " + "\n  ".join(bad))
    return len(ANCHORS)


def f2(x, ref=None, nd=2):
    if x is None:
        return "–"
    s = f"{x:.{nd}f}"
    return s if ref is None else f"{s} ({x - ref:+.2f})"


def write_md(recs, n_anchor):
    L = []
    w = L.append
    w("# ReadPast 结果总表（自动生成，勿手改）\n")
    w(f"由 `python3 kfs/make_tables.py` 从原始 run 重算生成（{datetime.datetime.now():%Y-%m-%d %H:%M}）；"
      f"机器可读版 `results/main_tables.json`。输入只取 `kfs/data/canonical_runs.json` 列出的 run；{n_anchor} 个已记录锚点（EXPERIMENTS.md 第十七、十八轮）逐位一致。"
      "口径与决定见 `STATUS.md`。\n")
    w("- OVO 标准 = (RT + BT) / 2，BT = (EPM + ASI + HLD) / 3；不含 HLD = (RT + (EPM + ASI) / 2) / 2。SB = StreamingBench 实时 2500 题准确率。")
    w("- 共用提示 = 只在带弃答选项的 187 道 OVO 题（186 HLD + 1 EPM）加 D-HSM 的弃答指令；官方提示 = SimpleStream 原提示。SB 无弃答选项，两种提示数字相同；不含 HLD 的列基本不受提示影响。")
    w("- ‡ D-HSM = 公开代码同协议重跑（它默认每道选择题都加弃答指令）。† 论文值，各自协议，复现的 SimpleStream 不同，不宜逐格比较。\n")

    w("## 表 1 主表（共用提示；括号内为相对同一提示 SimpleStream）\n")
    w("| 骨干 | 方法 | SB Real-Time | OVO Real-Time | OVO Backward | OVO Backward 不含 HLD | OVO Avg | OVO Avg 不含 HLD |")
    w("|---|---|---|---|---|---|---|---|")
    for b, (_, _, disp) in BB.items():
        ss = get(recs, b, "SimpleStream", "shared")
        rows = [("SimpleStream", ss, False), ("D-HSM‡", get(recs, b, "D-HSM", "own default"), False),
                ("ShallowStream†", get(recs, b, "ShallowStream (paper)", "paper"), False),
                ("**ReadPast**", get(recs, b, "ReadPast", "shared"), True)]
        first = True
        for name, r, delta in rows:
            if r is None:
                continue
            o, s = r["ovo"], r["sb"]
            ref = lambda f, src="ovo": (ss[src][f] if delta else None)  # noqa: E731
            w(f"| {disp if first else ''} | {name} | {f2(s['overall'], ref('overall', 'sb'))} | {f2(o['RT'], ref('RT'))} | "
              f"{f2(o['BT'], ref('BT'))} | {f2(o['BT_noHLD'], ref('BT_noHLD'))} | {f2(o['Avg'], ref('Avg'))} | "
              f"{f2(o['Avg_noHLD'], ref('Avg_noHLD'))} |")
            first = False

    w("\n## 表 2 官方提示（不加弃答指令）\n")
    w("| 骨干 | 方法 | SB Real-Time | OVO Real-Time | OVO Backward | OVO Backward 不含 HLD | OVO Avg | OVO Avg 不含 HLD |")
    w("|---|---|---|---|---|---|---|---|")
    for b, (_, _, disp) in BB.items():
        ss = get(recs, b, "SimpleStream", "official")
        for i, m in enumerate(("SimpleStream", "ReadPast")):
            r = get(recs, b, m, "official")
            o, s = r["ovo"], r["sb"]
            ref = lambda f, src="ovo": (ss[src][f] if i else None)  # noqa: E731
            w(f"| {disp if i == 0 else ''} | {m} | {f2(s['overall'], ref('overall', 'sb'))} | {f2(o['RT'], ref('RT'))} | "
              f"{f2(o['BT'], ref('BT'))} | {f2(o['BT_noHLD'], ref('BT_noHLD'))} | {f2(o['Avg'], ref('Avg'))} | "
              f"{f2(o['Avg_noHLD'], ref('Avg_noHLD'))} |")

    w("\n## 表 3 OVO 逐子任务（共用提示；D-HSM 为其默认提示；论文值行一位小数）\n")
    cols = OVO_RT + ["RT"] + OVO_BT + ["BT", "BT_noHLD", "Avg", "Avg_noHLD"]
    w("| 骨干 | 方法 | " + " | ".join(c.replace("_noHLD", " 不含HLD") for c in cols) + " |")
    w("|---|---|" + "---|" * len(cols))
    for b, (_, _, disp) in BB.items():
        rows = [("SimpleStream", get(recs, b, "SimpleStream", "shared")), ("D-HSM‡", get(recs, b, "D-HSM", "own default")),
                ("D-HSM†", get(recs, b, "D-HSM (paper)", "paper")),
                ("ShallowStream†", get(recs, b, "ShallowStream (paper)", "paper")),
                ("ReadPast", get(recs, b, "ReadPast", "shared"))]
        first = True
        for name, r in rows:
            if r is None:
                continue
            nd = 1 if r["source"] == "paper" else 2
            w(f"| {disp if first else ''} | {name} | " + " | ".join(f2(r["ovo"].get(c), nd=nd) for c in cols) + " |")
            first = False

    w("\n## 表 4 StreamingBench 实时逐子任务\n")
    cols = list(SB_TASKS) + ["overall"]
    w("| 骨干 | 方法 | " + " | ".join(cols) + " |")
    w("|---|---|" + "---|" * len(cols))
    for b, (_, _, disp) in BB.items():
        rows = [("SimpleStream", get(recs, b, "SimpleStream", "shared")), ("D-HSM‡", get(recs, b, "D-HSM", "own default")),
                ("D-HSM†", get(recs, b, "D-HSM (paper)", "paper")),
                ("ShallowStream†", get(recs, b, "ShallowStream (paper)", "paper")),
                ("ReadPast", get(recs, b, "ReadPast", "shared"))]
        first = True
        for name, r in rows:
            if r is None:
                continue
            nd = 1 if r["source"] == "paper" else 2
            w(f"| {disp if first else ''} | {name} | " + " | ".join(f2(r["sb"].get(c), nd=nd) for c in cols) + " |")
            first = False

    w("\n## 表 5 门的消融（共用提示；括号内为相对 SimpleStream；读记忆比例 = 门判为读的题占比）\n")
    w("| 骨干 | 读记忆规则 | OVO Avg | OVO Avg 不含 HLD | OVO RT | SB | OVO 读记忆 BT / RT | SB 读记忆 |")
    w("|---|---|---|---|---|---|---|---|")
    for b, (_, _, disp) in BB.items():
        ss = get(recs, b, "SimpleStream", "shared")
        for i, m in enumerate(RULES):
            r = get(recs, b, m, "shared")
            o, s = r["ovo"], r["sb"]
            ref = lambda f, src="ovo": (ss[src][f] if i else None)  # noqa: E731
            name = "ReadPast 门 (a)" if m == "ReadPast" else ("从不读（SimpleStream）" if m == "SimpleStream" else m)
            w(f"| {disp if i == 0 else ''} | {name} | {f2(o['Avg'], ref('Avg'))} | {f2(o['Avg_noHLD'], ref('Avg_noHLD'))} | "
              f"{f2(o['RT'], ref('RT'))} | {f2(s['overall'], ref('overall', 'sb'))} | "
              f"{o['read_BT']:.0f}% / {o['read_RT']:.0f}% | {s['read']:.0f}% |")
    w("\n消融的显著性（按视频重抽样）见 EXPERIMENTS.md 11-05：只有“时间线”清楚，门 ≈ 总是读（OVO），SB 上门更差。")
    open("RESULTS.md", "w").write("\n".join(L) + "\n")


def main():
    n_runs = check_manifest()
    recs = build()
    n = check_anchors(recs)
    os.makedirs("results", exist_ok=True)
    json.dump(dict(generated=datetime.datetime.now().isoformat(timespec="seconds"), records=recs),
              open("results/main_tables.json", "w"), indent=1)
    write_md(recs, n)
    print(f"{n_runs} manifest runs; {len(recs)} records; {n} anchors OK -> results/main_tables.json, RESULTS.md")


if __name__ == "__main__":
    main()
