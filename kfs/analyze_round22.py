"""Round 22 (EXPERIMENTS.md): write-lag corrected ReadPast scores.

Composition: the canonical runs of kfs/data/canonical_runs.json, with the records of the affected questions
(kfs/make_keys_r22.py: gate (a) reads and the timeline loses its newest caption(s) under lag w) replaced by
the round-22 reruns (same commands + --write_lag w). Scoring = kfs/make_tables.py (imported, unchanged), so
"offline" below reproduces RESULTS.md exactly (checked against its anchors).

Checks: every affected key has a rerun record; each rerun record's n_captions equals the lagged count
(OVO); unaffected keys are untouched.

Input: kfs/data/r22/runs.json  {bb: {"ovo": RUN, "ovo_instr": RUN, "sb": [RUN, ...]}}
usage (repo root, host python): python3 kfs/analyze_round22.py [--md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "kfs")
import make_tables as mt  # noqa: E402
import make_keys_r22 as mk  # noqa: E402

R22 = "kfs/data/r22"
_orig_raw = mt.raw
OVERLAY: dict[str, dict] = {}  # canonical run id -> {key: rerun record}


def raw_overlay(run_id):
    rows = _orig_raw(run_id)
    for k, r in OVERLAY.get(run_id, {}).items():
        if k in rows:
            rows[k] = r
    return rows


def load_reruns(run_ids):
    rows = {}
    for run_id in run_ids if isinstance(run_ids, list) else [run_ids]:
        rows.update(_orig_raw(run_id))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--md", action="store_true", help="print markdown tables")
    args = ap.parse_args()
    runs = json.load(open(f"{R22}/runs.json"))
    w_all = json.load(open(f"{R22}/w.json"))
    mt.raw = raw_overlay
    out = {}
    for bb, m in mt.MANIFEST["ours"].items():
        if bb not in runs:
            continue
        w = w_all[bb]["w"]
        OVERLAY.clear()
        off_ovo, _ = mt.ours_ovo(m)
        off_sb = mt.ours_sb(m)
        for (b, meth, prompt, f), v in mt.ANCHORS.items():  # offline must reproduce RESULTS.md
            if b == bb and meth == "ReadPast":
                x = off_sb["ReadPast"]["overall"] if f == "SB" else off_ovo[("ReadPast", prompt)][f]
                assert round(x, 2) == v, (bb, prompt, f, x, v)

        keys_ovo = json.load(open(f"{R22}/keys_ovo_{bb}.json"))
        keys_abst = json.load(open(f"{R22}/keys_ovo_abst_{bb}.json"))
        keys_sb = json.load(open(f"{R22}/keys_sb_{bb}.json"))
        new_ovo, new_abst, new_sb = (load_reruns(runs[bb][x]) for x in ("ovo", "ovo_instr", "sb"))
        assert set(new_ovo) == set(keys_ovo), (bb, "ovo keys", len(new_ovo), len(keys_ovo))
        assert set(new_abst) == set(keys_abst), (bb, "abst keys", len(new_abst), len(keys_abst))
        assert set(new_sb) == set(keys_sb), (bb, "sb keys", len(new_sb), len(keys_sb))
        base = _orig_raw(m["ovo_timeline"])
        for k, r in new_ovo.items():
            exp = mk.n_caps(f"cache/captions_{bb}_s16", r["video"], r["query_ts"] - w)
            assert r["n_captions"] == exp and r["n_captions"] < base[k]["n_captions"], (bb, k, r["n_captions"], exp)
        for k, r in new_abst.items():
            assert r["n_captions"] == new_ovo[k]["n_captions"], (bb, k)

        OVERLAY[m["ovo_timeline"]] = new_ovo
        OVERLAY[m["ovo_timeline_instr"]] = new_abst
        for sid in m["sb"]:
            OVERLAY[sid] = new_sb
        lag_ovo, _ = mt.ours_ovo(m)
        lag_sb = mt.ours_sb(m)
        OVERLAY.clear()

        # answer changes among the affected questions (generated letter; shared prompt uses argmax on the 187)
        flips_ovo = sum(mt.mcq(new_ovo[k]["response"]) != mt.mcq(base[k]["response"]) for k in keys_ovo)
        sb_base = load_reruns(m["sb"])
        flips_sb = sum(new_sb[k]["response_l1"] != sb_base[k]["response_l1"] for k in keys_sb)
        l0_diff = sum(new_sb[k]["p_opts_l0"] != sb_base[k]["p_opts_l0"] for k in keys_sb)  # must be 0 (same recent view)
        assert l0_diff == 0, (bb, "SB L0 differs from canonical on", l0_diff)
        res = {"w": w, "n_ovo": len(keys_ovo), "n_abst": len(keys_abst), "n_sb": len(keys_sb),
               "ovo_answer_changed": flips_ovo, "sb_answer_changed": flips_sb}
        for prompt in ("shared", "official"):
            ss, off, lag = (d[(rule, prompt)] for d, rule in
                            ((off_ovo, "SimpleStream"), (off_ovo, "ReadPast"), (lag_ovo, "ReadPast")))
            res[prompt] = {f: {"SimpleStream": ss[f], "offline": off[f], "lag": lag[f]}
                           for f in ("RT", "BT", "BT_noHLD", "Avg", "Avg_noHLD", "EPM", "ASI", "HLD")}
        res["SB"] = {"SimpleStream": off_sb["SimpleStream"]["overall"], "offline": off_sb["ReadPast"]["overall"],
                     "lag": lag_sb["ReadPast"]["overall"]}
        out[bb] = res
    json.dump(out, open("results/round22_writelag.json", "w"), indent=1)

    print("| backbone | w (s) | redo OVO / SB | metric | SimpleStream | ReadPast offline | ReadPast online (lag w) | gain offline -> online |")
    print("|---|---|---|---|---|---|---|---|")
    for bb, r in out.items():
        rows = [(f"OVO {f}", r["shared"][f]) for f in ("Avg", "Avg_noHLD", "RT", "BT")] + [("SB", r["SB"])]
        for i, (name, d) in enumerate(rows):
            head = f"| {mt.BB[bb][2]} | {r['w']:.2f} | {r['n_ovo']} / {r['n_sb']} |" if i == 0 else "| | | |"
            print(f"{head} {name} | {d['SimpleStream']:.2f} | {d['offline']:.2f} | {d['lag']:.2f} | "
                  f"{d['offline'] - d['SimpleStream']:+.2f} -> {d['lag'] - d['SimpleStream']:+.2f} |")
    print("\nofficial prompt, OVO Avg (SimpleStream / offline / online):",
          {bb: tuple(round(r['official']['Avg'][x], 2) for x in ("SimpleStream", "offline", "lag")) for bb, r in out.items()})
    print("answers changed among redone questions (OVO / SB):",
          {bb: (r["ovo_answer_changed"], r["sb_answer_changed"]) for bb, r in out.items()})


if __name__ == "__main__":
    main()
