"""Round 22 (EXPERIMENTS.md): write-lag correction -- which questions must be re-answered.

Online, the caption of clip [t0, t1] is usable only after the writer has finished it: t1 + w <= query time,
w = p90 per-clip write time of that backbone (kfs/bench_latency.py --stages write, one exclusive H100;
summary.total_s.p90 over the non-warm-up clips). Offline (canonical runs) it was usable at t1.

Affected question = gate (a) reads the timeline (p_past >= p_cf, canonical route run) AND the informative
captions with t1 + w <= query time differ from those with t1 <= query time. Every other question keeps its
canonical answer, so the corrected score is an offline composition (kfs/analyze_round22.py).
SimpleStream never reads the timeline and is unchanged.

Check (fails loudly): the offline caption count recomputed here equals the canonical records
(OVO n_captions; SB timeline non-empty <=> count > 0).

Writes kfs/data/r22/: w.json, keys_ovo_{bb}.json (all affected OVO keys, plain prompt),
keys_ovo_abst_{bb}.json (affected keys among the 187 abstention-option MCQs, shared prompt),
keys_sb_{bb}.json (affected SB _keys).

usage (repo root, host python, stdlib only):
  python3 kfs/make_keys_r22.py --bench q25=RUN_ID q3=RUN_ID q4=RUN_ID lov=RUN_ID
"""
from __future__ import annotations

import argparse
import json
import os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = json.load(open("kfs/data/canonical_runs.json"))
RUNS = MANIFEST["runs_root"]
ABST_KEYS = set(json.load(open("kfs/data/keys_ovo_abstopt187.json")))
OUT = "kfs/data/r22"


def odir(run_id: str) -> str:
    return f"{RUNS}/{run_id}/out"


def raw(run_id: str) -> dict:
    return {(r := json.loads(line))["_key"]: r for line in open(f"{odir(run_id)}/results_incremental.jsonl")}


def gate_a(route_run: str) -> dict:
    """Same rule as kfs/make_tables.py gates()['a']."""
    route = json.load(open(f"{odir(route_run)}/route.json"))
    p_cf = json.load(open(f"{odir(route_run)}/calib.json"))["p_cf"]
    return {k: v["p_past"] >= p_cf for k, v in route.items()}


_caps: dict = {}


def n_caps(cap_dir: str, video: str, t: float) -> int:
    """Same filter as kfs/eval_ovo_kfs.py timeline_captions (informative captions with t1 <= t)."""
    if (cap_dir, video) not in _caps:
        p = os.path.join(cap_dir, video.replace("/", "__") + ".json")
        _caps[cap_dir, video] = json.load(open(p))["captions"] if os.path.exists(p) else []
    return sum(c["t1"] <= t + 1e-6 and c["text"].strip().lower().rstrip(".") != "no change" for c in _caps[cap_dir, video])


def hms(s: str) -> float:
    """SimpleStream eval_streamingbench.timestamp_to_seconds (MM:SS or HH:MM:SS)."""
    return float(sum(int(x) * 60 ** i for i, x in enumerate(reversed(s.split(":")))))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", nargs="+", required=True, help="bb=RUN_ID of kfs/bench_latency.py (write stage)")
    args = ap.parse_args()
    bench = dict(x.split("=", 1) for x in args.bench)
    os.makedirs(OUT, exist_ok=True)
    w_all, stats = {}, {}
    for bb, m in MANIFEST["ours"].items():
        wsum = json.load(open(f"{odir(bench[bb])}/bench.json"))["write"]["summary"]
        w = wsum["total_s"]["p90"]
        w_all[bb] = {"w": w, "median": wsum["total_s"]["median"], "n": wsum["total_s"]["n"], "bench_run": bench[bb],
                     "same_as_cache": wsum["same_as_cache"], "n_rows": wsum["n_rows"]}

        # OVO: query time and caption count are in the canonical timeline run's records
        recs, g = raw(m["ovo_timeline"]), gate_a(m["route_ovo"])
        cap = f"cache/captions_{bb}_s16"
        aff = []
        n_edge = 0
        for k, r in recs.items():
            n0 = n_caps(cap, r["video"], r["query_ts"])
            if n0 != r["n_captions"]:
                # recorded query_ts is rounded to 0.01 s; the eval passed full[-1] / fps, which can sit just below a
                # clip end (e.g. 63.999 vs t1 = 64). Only that boundary case is tolerated; trust the record there.
                assert n0 == r["n_captions"] + 1 and n_caps(cap, r["video"], r["query_ts"] - 0.011) == r["n_captions"], \
                    (bb, k, n0, r["n_captions"])
                n0, n_edge = r["n_captions"], n_edge + 1
            if g[k] and n_caps(cap, r["video"], r["query_ts"] - w) != n0:
                aff.append(k)
        bt = [k for k in aff if k.split(":")[0] in ("EPM", "ASI", "HLD")]
        abst = [k for k in aff if k in ABST_KEYS]
        json.dump(sorted(aff), open(f"{OUT}/keys_ovo_{bb}.json", "w"))
        json.dump(sorted(abst), open(f"{OUT}/keys_ovo_abst_{bb}.json", "w"))

        # SB
        rows = {}
        for run_id in m["sb"]:
            rows.update(raw(run_id))
        gs = gate_a(m["route_sb"])
        cap = f"cache/captions_{bb}_s16_sb"
        aff_sb = []
        for k, r in rows.items():
            if r["response_l0"] is None:  # undecodable recent window: scored wrong for every method, nothing to redo
                continue
            ts = hms(r["time_stamp"])
            n0 = n_caps(cap, r["video"], ts)
            assert (n0 > 0) == (r["timeline_tokens"] > 0), (bb, k, n0, r["timeline_tokens"])
            if gs[k.split("_", 1)[1]] and n0 > 0 and n_caps(cap, r["video"], ts - w) != n0:
                aff_sb.append(k)
        json.dump(sorted(aff_sb), open(f"{OUT}/keys_sb_{bb}.json", "w"))
        stats[bb] = {"w": round(w, 3), "ovo_affected": len(aff), "ovo_bt": len(bt), "ovo_rt": len(aff) - len(bt),
                     "ovo_abst": len(abst), "ovo_n": len(recs), "ovo_edge": n_edge, "sb_affected": len(aff_sb), "sb_n": len(rows)}
    json.dump(w_all, open(f"{OUT}/w.json", "w"), indent=1)
    for bb, s in stats.items():
        print(bb, json.dumps(s))


if __name__ == "__main__":
    main()
