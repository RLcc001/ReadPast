"""Router comparison (stdlib, host): which questions read the caption memory, for every question-only router,
combined offline with the recorded L0 (recent window) / L1 (cap16 timeline + recent window) answers.

Routers (decisions from kfs/route.py and kfs/route_baselines.py; all see the question text only):
  ours (calibrated)   p_past >= p_cf, p_cf = mean p_past of content-free questions (no labels, no tuning)
  ours (fixed 0.5)    p_past >= 0.5
  ShallowStream gate  logit(A) - logit(B) >= tau, zero-shot and 10-shot prompts; tau = 0 (released config default),
                      tau = Wilson (lowest tau whose one-sided 95% Wilson lower bound on retrieval precision >= 0.9 on
                      the synthetic set kfs/data/route_synth_v1.json, the paper's criterion), tau = 11.0 (paper, Qwen3)
  FlashBack           structured temporal-evidence router, fixed state -> Native / Recall mapping
  ... tau=cf / FlashBack calibrated (round 8, experiment A): our label-free content-free calibration applied to the
                      prior gates, from the routers_cf_* runs (scores of "N/A", "", "[MASK]"). ShallowStream: read iff
                      sigmoid(gap) >= mean sigmoid(gap_cf). FlashBack: per stage q = softmax(state scores), calibrated
                      state = argmax(log q - log mean q_cf) (Zhao et al. 2021), then the fixed mapping.
Reference rows: never (r4), always L1, label route (OVO: BT -> L1).

Cells: {Qwen2.5, Qwen3} x {OVO (avg of BT/RT task means), SB-RT (accuracy)}. A question without a timeline is
answered at L0 by every rule. CIs are paired video-bootstrap (1000) differences against ours (calibrated).

usage: python3 kfs/analyze_routers.py
"""

import glob
import json
import math
import sys

sys.path.insert(0, "kfs")
from analyze_pilot import BACKWARD, REALTIME, load, scores  # noqa: E402
from analyze_route import RUNS, boot, no_hld  # noqa: E402
from analyze_sb import acc  # noqa: E402
from analyze_sb import boot as boot_sb  # noqa: E402

BACKBONES = {
    "Qwen2.5-VL-7B": dict(ovo=("r4_ss", "cap16"), sb="261025-04464*-sb_casc_s[0-3]", route_ovo="route_cal_p0",
                          route_sb="route_cal_sb_p0", routers="routers_q25", routers_cf="routers_cf_q25", paper_tau=None),
    "Qwen3-VL-8B": dict(ovo=("r4_q3", "cap16_q3"), sb="261101-0240*-sb_casc_q3_s[0-3]", route_ovo="route_q3_cal_p0",
                        route_sb="route_q3_cal_sb_p0", routers="routers_q3", routers_cf="routers_cf_q3", paper_tau=11.0),
    "Qwen3-VL-4B": dict(ovo=("r4_q4", "cap16_q4"), sb="2611*-sb_casc_q4_s[0-3]", route_ovo="route_q4_cal_p0",
                        route_sb="route_q4_cal_sb_p0", routers="routers_q4", routers_cf="routers_q4", paper_tau=None),
    "LLaVA-OV-7B": dict(ovo=("r4_lov", "cap16_lov"), sb="2611*-sb_casc_lov_s[0-3]", route_ovo="route_lov_cal_p0",
                        route_sb="route_lov_cal_sb_p0", routers="routers_lov", routers_cf="routers_lov", paper_tau=0.875),
}
SYNTH = "kfs/data/route_synth_v1.json"


def newest(pattern):
    paths = sorted(glob.glob(f"{RUNS}/{pattern}"))
    if not paths:
        raise FileNotFoundError(pattern)
    return paths[-1]


def wilson_lb(k, n, z=1.645):
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def wilson_tau(scores_synth, field):
    """Lowest threshold (max recall) with one-sided 95% Wilson LB on retrieval precision >= 0.9."""
    labels = [it["label"] == "A" for it in json.load(open(SYNTH))["items"]]
    xs = [(scores_synth[f"synth:{i}"][field], y) for i, y in enumerate(labels)]
    for t in sorted({x for x, _ in xs}):
        sel = [y for x, y in xs if x >= t]
        if wilson_lb(sum(sel), len(sel)) >= 0.9:
            return t
    return math.inf


FB1_RECALL = {"HISTORICAL_STATE": True, "TEMPORAL_INTERACTION": True, "GLOBAL_AGGREGATION": True,
              "PREDICTIVE_CONSEQUENCE": False}
FB2_RECALL = {"SIMULTANEOUS_BOUNDARY_CONFIGURATION": False, "ONGOING_PROCESS": False,
              "PERSISTENT_CURRENT_STATE": True, "REALIZED_BOUNDED_OUTCOME": True}


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def softmax(d):
    m = max(d.values())
    e = {k: math.exp(v - m) for k, v in d.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def mean_dist(ds):
    return {k: sum(d[k] for d in ds) / len(ds) for k in ds[0]}


def cf_routers(cfg, bench, sc_old):
    """Round 8 experiment A: content-free calibration of the prior gates (empty if the runs are missing)."""
    try:
        cdir = newest(f"*-{cfg['routers_cf']}/out")
        cf, sc = json.load(open(f"{cdir}/scores_cf.json")), json.load(open(f"{cdir}/scores_{bench}.json"))
    except (FileNotFoundError, KeyError):
        return {}, {}
    out, info = {}, {}
    for f, lab in (("ss_zs", "zero-shot"), ("ss_fs10", "10-shot")):
        p_cf = sum(sigmoid(v[f]) for v in cf.values()) / len(cf)
        out[f"ShallowStream {lab} tau=cf"] = {k: sigmoid(v[f]) >= p_cf for k, v in sc.items()}
        info[f"{f} tau=cf (gap)"] = round(math.log(p_cf / (1 - p_cf)), 3) if 0 < p_cf < 1 else p_cf
    q1 = mean_dist([softmax(v["fb_s1"]) for v in cf.values()])
    q2 = mean_dist([softmax(v["fb_s2"]) for v in cf.values()])

    def cal(rec):
        s1 = softmax(rec["fb_s1"])
        top1 = max(s1, key=lambda k: math.log(s1[k]) - math.log(q1[k]))
        if top1 != "CURRENT_PERCEPTION":
            return FB1_RECALL[top1]
        s2 = softmax(rec["fb_s2"])
        return FB2_RECALL[max(s2, key=lambda k: math.log(s2[k]) - math.log(q2[k]))]

    out["FlashBack calibrated"] = {k: cal(v) for k, v in sc.items()}
    out["FlashBack (rescored, uncal.)"] = {k: v["fb_recall"] for k, v in sc.items()}
    info["ss score mismatches vs routers run"] = sum(
        abs(sc[k][f] - sc_old[k][f]) > 1e-2 for k in sc if k in sc_old for f in ("ss_zs", "ss_fs10"))
    return out, info


def routers(cfg, bench):
    """name -> (key -> read memory?), keys as in route.json / scores_*.json."""
    rdir = newest(f"*-{cfg['route_' + bench]}/out")
    route, p_cf = json.load(open(f"{rdir}/route.json")), json.load(open(f"{rdir}/calib.json"))["p_cf"]
    sdir = newest(f"*-{cfg['routers']}/out")
    sc, synth = json.load(open(f"{sdir}/scores_{bench}.json")), json.load(open(f"{sdir}/scores_synth.json"))
    out = {"ours (calibrated)": {k: v["p_past"] >= p_cf for k, v in route.items()},
           "ours (fixed 0.5)": {k: v["p_past"] >= 0.5 for k, v in route.items()}}
    taus = {}
    for f, lab in (("ss_zs", "zero-shot"), ("ss_fs10", "10-shot")):
        ts = {"0": 0.0, "Wilson": wilson_tau(synth, f)}
        if f == "ss_fs10" and cfg["paper_tau"] is not None:
            ts["paper"] = cfg["paper_tau"]
        for tn, t in ts.items():
            out[f"ShallowStream {lab} tau={tn}"] = {k: v[f] >= t for k, v in sc.items()}
            taus[f"{f} tau={tn}"] = t
    out["FlashBack structured"] = {k: v["fb_recall"] for k, v in sc.items()}
    mismatch = sum(abs(sc[k]["p_past"] - route[k]["p_past"]) > 1e-3 for k in route if k in sc)
    cfo, cfi = cf_routers(cfg, bench, sc)
    out.update(cfo)
    return out, {"p_cf": p_cf, **taus, "p_past mismatches vs route run": mismatch, **cfi}, sc, route


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def ovo_cell(name, cfg):
    base, mem = load(cfg["ovo"][0], RUNS), load(cfg["ovo"][1], RUNS)
    keys = sorted(set(base) & set(mem))
    tasks = {k: base[k]["task"] for k in keys}
    video = {k: base[k]["video"] for k in keys}
    b = {k: base[k]["correct"] for k in keys}
    m = {k: mem[k]["correct"] for k in keys}
    rs, info, sc, route = routers(cfg, "ovo")
    bt = [k for k in keys if tasks[k] in BACKWARD]
    rt = [k for k in keys if tasks[k] in REALTIME]
    aucs = {"ours p_past": auc([route[k]["p_past"] for k in bt], [route[k]["p_past"] for k in rt])}
    for f in ("ss_zs", "ss_fs10"):
        aucs[f] = auc([sc[k][f] for k in bt], [sc[k][f] for k in rt])
    print(f"\n=== {name} OVO  ({info})\nAUC BT-vs-RT: " + ", ".join(f"{k} {v:.3f}" for k, v in aucs.items()))
    rules = {"never (r4)": {k: False for k in keys}, "always L1": {k: True for k in keys},
             "label route (BT->L1)": {k: tasks[k] in BACKWARD for k in keys}, **rs}
    ref = {k: (m if rules["ours (calibrated)"][k] else b)[k] for k in keys}
    s0 = scores(b, tasks)
    print(f"{'rule':38s}   avg   d_r4    RT  noHLD  read BT/RT   d vs ours (95% CI)")
    res = {}
    for rn, dec in rules.items():
        c = {k: (m if dec[k] else b)[k] for k in keys}
        s = scores(c, tasks)
        rb = 100 * sum(dec[k] for k in bt) / len(bt)
        rr = 100 * sum(dec[k] for k in rt) / len(rt)
        lo, hi, _ = boot(keys, tasks, video, ref, c, lambda x: x["avg"], reps=1000)
        d = s["avg"] - scores(ref, tasks)["avg"]
        print(f"{rn:38s} {s['avg']:5.2f} {s['avg'] - s0['avg']:+5.2f} {s['rt']:5.2f} {no_hld(s):6.2f}  {rb:4.0f}/{rr:3.0f}"
              f"   {d:+5.2f} [{lo:+.2f}, {hi:+.2f}]")
        res[rn] = dict(score=s["avg"], d_r4=s["avg"] - s0["avg"], d_rt=s["rt"] - s0["rt"],
                       ok=s["avg"] >= s0["avg"] + 1.0 and s["rt"] >= s0["rt"] - 0.3)
    return res


def sb_cell(name, cfg):
    rows = {}
    for p in sorted(glob.glob(f"{RUNS}/{cfg['sb']}/out/results_incremental.jsonl")):
        for line in open(p):
            r = json.loads(line)
            rows[r["_key"]] = r
    rows = list(rows.values())
    qid = {r["_key"]: r["_key"].split("_", 1)[1] for r in rows}  # SB keys are "video.mp4_<question_id>"
    rs, info, _, _ = routers(cfg, "sb")
    print(f"\n=== {name} SB-RT  (n={len(rows)}; {info})")
    has_tl = lambda r: r["timeline_tokens"] > 0  # noqa: E731
    rules = {"never (r4)": None, "always L1": "all", **rs}

    def pick(dec):
        if dec is None:
            return lambda r: r["correct_l0"]
        if dec == "all":
            return lambda r: r["correct_l1"]
        return lambda r: r["correct_l1"] if has_tl(r) and dec[qid[r["_key"]]] else r["correct_l0"]

    ref, base = pick(rs["ours (calibrated)"]), pick(None)
    print(f"{'rule':38s}   acc   d_r4  read   d vs ours (95% CI)")
    res = {}
    for rn, dec in rules.items():
        f = pick(dec)
        read = 100 * sum(has_tl(r) and (dec == "all" or (dec is not None and dec[qid[r["_key"]]])) for r in rows) / len(rows)
        lo, hi, _ = boot_sb(rows, ref, f)
        print(f"{rn:38s} {acc(rows, f):5.2f} {acc(rows, f) - acc(rows, base):+5.2f}  {read:4.0f}"
              f"   {acc(rows, f) - acc(rows, ref):+5.2f} [{lo:+.2f}, {hi:+.2f}]")
        res[rn] = dict(score=acc(rows, f), d_r4=acc(rows, f) - acc(rows, base), ok=acc(rows, f) >= acc(rows, base))
    return res


def main():
    cells = {}
    for name, cfg in BACKBONES.items():
        cells[(name, "OVO")] = ovo_cell(name, cfg)
        cells[(name, "SB")] = sb_cell(name, cfg)
    names = [n for n in next(iter(cells.values())) if all(n in c for c in cells.values())]
    print("\n=== summary: gain over r4 per cell (pass = OVO avg >= r4+1.0 and RT >= r4-0.3; SB >= r4)")
    print(f"{'rule':38s} " + " ".join(f"{b.replace('Qwen', 'Q').replace('-VL-', '-'):>7s}-{x:3s}" for b, x in cells) + "   mean  passes")
    for n in names:
        ds = [cells[c][n]["d_r4"] for c in cells]
        ok = sum(cells[c][n]["ok"] for c in cells)
        print(f"{n:38s} " + " ".join(f"{d:+11.2f}" for d in ds) + f"  {sum(ds) / len(ds):+5.2f}  {ok}/{len(ds)}")


if __name__ == "__main__":
    main()
