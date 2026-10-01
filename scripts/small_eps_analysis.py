"""Analysis for docs/prereg_small_eps.md (CPU only).

    python scripts/small_eps_analysis.py
"""
from __future__ import annotations

import json
import os

import numpy as np

LARGE = ["llama3.1-8b", "qwen2.5-7b", "olmo2-7b", "granite3.1-8b", "mistral-7b"]
DIRS = {"triviaqa": "results/small_eps/triviaqa", "nq_open": "results/small_eps/nq_open"}
EPS = [0.01, 0.02, 0.03]


def main():
    items = json.load(open("results/judge_audit/items.json"))
    V = [json.load(open(f"results/judge_audit/verdicts_{k}.json")) for k in ("qwen", "phi")]
    out = {}
    for ds, d in DIRS.items():
        flagged = {}
        for n, it in enumerate(items):
            if it["audit"] == "B" and it["ds"] == ds and any(v[n] == 1 for v in V):
                flagged.setdefault(it["model"], set()).add(it["j"])
        arr = lambda t, key: np.array([np.nan if v is None else v for v in t[key]], float)
        R = {}
        for m in LARGE:
            p = os.path.join(d, m, "type_control.json")
            if not os.path.exists(p):
                continue
            t = json.load(open(p))
            R[m] = {}
            for e in EPS:
                g, w, r = arr(t, f"gold:{e}"), arr(t, f"sampled_wrong:{e}"), arr(t, f"random:{e}")
                keep = ~np.isnan(g) & ~np.isnan(w) & np.array([j not in flagged.get(m, set()) for j in range(len(g))])
                diff = g[keep] - w[keep]
                bs = np.random.default_rng(0).integers(0, len(diff), size=(2000, len(diff)))
                ci = [float(np.percentile(diff[bs].mean(1), q)) for q in (2.5, 97.5)]
                R[m][e] = {"gold": float(g[keep].mean()), "wrong": float(w[keep].mean()),
                           "random": float(np.nanmean(r)), "n": int(keep.sum()), "ci": ci}
        ok_eps = [e for e in EPS if R and all(R[m][e]["random"] <= 0.15 for m in R)]
        e_star = max(ok_eps) if ok_eps else None
        res = {"eps_star": e_star, "per_model": R}
        if e_star is not None:
            n_ok = sum(not (R[m][e_star]["ci"][0] > 0) for m in R)
            res.update({"gold_not_easier": n_ok, "of": len(R), "holds": n_ok >= 4})
        out[ds] = res
        print(f"== {ds}: eps* = {e_star}")
        for m, x in R.items():
            for e in EPS:
                y = x[e]
                print(f"  {m:14s} eps {e}: gold {y['gold']:.2f} wrong {y['wrong']:.2f} random {y['random']:.2f} "
                      f"n {y['n']} ci {np.round(y['ci'], 2)}")
        if e_star is not None:
            print(f"  headline at eps*: gold not easier in {res['gold_not_easier']}/{res['of']} -> "
                  f"{'holds' if res['holds'] else 'FAILS'}")
    json.dump(out, open("results/small_eps/analysis.json", "w"), indent=1)


if __name__ == "__main__":
    main()
