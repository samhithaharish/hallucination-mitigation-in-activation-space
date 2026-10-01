"""Detector-gated generate-then-select with calibrated abstention (see docs/prereg_generate_then_select.md).

CPU only; reads results/mixed_vs_separated/<key>/scores.npz.

    python scripts/path_b.py                   # all models with scores, plus the held-out aggregate
"""
from __future__ import annotations

import json
import os
import sys
from itertools import product

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from mixed_vs_separated import MODELS  # noqa: E402

ROOT = os.environ.get("MVS_ROOT", "results/mixed_vs_separated")
DEV = "llama3.2-3b"
LAMBDAS = [1, 2]


def per_question(z):
    y, p_h, cq, cl = z["y"], z["p_h"], z["cq"], z["cl"]
    pr = np.clip(z["cprobe"], 1e-4, 1 - 1e-4)
    F = np.c_[np.log(pr / (1 - pr)), z["cfreq"], z["clogp"], z["cgreedy"]]
    trq = np.zeros(len(y), bool)
    trq[z["tr"]] = True
    ctr = trq[cq]
    meta = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(F[ctr], cl[ctr])
    s = meta.predict_proba(F)[:, 1]
    Q = {k: np.zeros(len(y)) for k in ("g_ok", "g_s", "g_f", "b_ok", "b_s", "m_ok", "m_f")}
    Q["g_ok"] = 1.0 - y
    by = {}
    for j, q in enumerate(cq):
        by.setdefault(q, []).append(j)
    for q, js in by.items():
        js = np.array(js)
        g = js[z["cgreedy"][js] == 1]
        if len(g):
            Q["g_s"][q], Q["g_f"][q] = s[g[0]], z["cfreq"][g[0]]
        b = js[np.argmax(s[js])]
        Q["b_ok"][q], Q["b_s"][q] = cl[b], s[b]
        m = js[np.lexsort((s[js], z["cgreedy"][js], z["cfreq"][js]))[-1]]
        Q["m_ok"][q], Q["m_f"][q] = cl[m], z["cfreq"][m]
    Q["p_h"] = p_h
    return Q


def reliability(ok, ans, lam):
    return (np.sum(ok * ans) - lam * np.sum((1 - ok) * ans)) / len(ok)


def switch(Q, tau, m):
    sw = (Q["p_h"] > tau) & (Q["b_s"] - Q["g_s"] > m)
    return np.where(sw, Q["b_ok"], Q["g_ok"]), np.where(sw, Q["b_s"], Q["g_s"])


def analyse(key):
    z = np.load(os.path.join(ROOT, key, "scores.npz"))
    Q = per_question(z)
    tr, te = z["tr"], z["te"]
    sub = lambda d, idx: {k: v[idx] for k, v in d.items()}
    Qtr, Qte = sub(Q, tr), sub(Q, te)
    B = np.random.default_rng(0).integers(0, len(te), size=(2000, len(te)))
    ci = lambda d: [float(np.percentile(d[B].mean(1), 2.5)), float(np.percentile(d[B].mean(1), 97.5))]

    # switch rule, tuned on train for accuracy
    taus = np.quantile(Qtr["p_h"], [0.5, 0.6, 0.7, 0.8, 0.9])
    tau, m = max(product(taus, [0, 0.05, 0.1, 0.2, 0.3]), key=lambda tm: switch(Qtr, *tm)[0].mean())
    sw_ok, sw_s = switch(Qte, tau, m)
    out = {"key": key, "size": MODELS[key][1], "dev": key == DEV, "tau": float(tau), "m": m,
           "acc": {"greedy": float(Qte["g_ok"].mean()), "majority": float(Qte["m_ok"].mean()),
                   "switch": float(sw_ok.mean()), "oracle_best_candidate": None},
           "switch_minus_greedy_ci": ci(sw_ok - Qte["g_ok"]),
           "switch_minus_majority_ci": ci(sw_ok - Qte["m_ok"])}

    def tuned(score_tr, ok_tr, score_te, ok_te, lam, higher_is_confident=True):
        """Pick the answer/abstain threshold on train (answering everything is always a candidate)."""
        def answered(sc, t):
            if t is None:
                return np.ones(len(sc))
            return ((sc >= t) if higher_is_confident else (sc <= t)).astype(float)
        grid = [None] + list(np.unique(np.quantile(score_tr, np.linspace(0, 1, 21))))
        t = max(grid, key=lambda t: reliability(ok_tr, answered(score_tr, t), lam))
        return answered(score_tr, t), answered(score_te, t)

    out["reliability"] = {}
    for lam in LAMBDAS:
        base = {}
        one_tr, one_te = np.ones(len(tr)), np.ones(len(te))
        base["a_greedy"] = (Qtr["g_ok"], one_tr, Qte["g_ok"], one_te)
        a_tr, a_te = tuned(Qtr["p_h"], Qtr["g_ok"], Qte["p_h"], Qte["g_ok"], lam, higher_is_confident=False)
        base["b_detector"] = (Qtr["g_ok"], a_tr, Qte["g_ok"], a_te)
        a_tr, a_te = tuned(Qtr["g_f"], Qtr["g_ok"], Qte["g_f"], Qte["g_ok"], lam)
        base["c_selfconsistency"] = (Qtr["g_ok"], a_tr, Qte["g_ok"], a_te)
        thr = lam / (1 + lam)
        base["d_calibrated"] = (Qtr["g_ok"], (Qtr["g_s"] > thr) * 1.0, Qte["g_ok"], (Qte["g_s"] > thr) * 1.0)
        a_tr, a_te = tuned(Qtr["m_f"], Qtr["m_ok"], Qte["m_f"], Qte["m_ok"], lam)
        base["e_majority_sc"] = (Qtr["m_ok"], a_tr, Qte["m_ok"], a_te)
        ours_tr_ok, ours_tr_s = switch(Qtr, tau, m)
        ours_te_ans = (sw_s > thr) * 1.0
        R = {k: {"train": float(reliability(v[0], v[1], lam)), "test": float(reliability(v[2], v[3], lam)),
                 "coverage": float(v[3].mean())} for k, v in base.items()}
        R["ours"] = {"train": float(reliability(ours_tr_ok, (ours_tr_s > thr) * 1.0, lam)),
                     "test": float(reliability(sw_ok, ours_te_ans, lam)), "coverage": float(ours_te_ans.mean())}
        best = max(base, key=lambda k: R[k]["train"])
        ok_b, ans_b = base[best][2], base[best][3]
        per_ours = sw_ok * ours_te_ans - lam * (1 - sw_ok) * ours_te_ans
        per_best = ok_b * ans_b - lam * (1 - ok_b) * ans_b
        # B3: matched coverage vs self-consistency ranking of greedy answers
        k = int(ours_te_ans.sum())
        top = np.argsort(-Qte["g_f"], kind="stable")[:k]
        sel_acc_sc = float(Qte["g_ok"][top].mean()) if k else float("nan")
        sel_acc_ours = float(sw_ok[ours_te_ans == 1].mean()) if k else float("nan")
        out["reliability"][f"lambda{lam}"] = {"R": R, "best_baseline": best, "ours_minus_best_ci": ci(per_ours - per_best),
                                              "matched_coverage": {"coverage": k / len(te), "ours": sel_acc_ours,
                                                                   "selfconsistency": sel_acc_sc}}
    return out


def main():
    keys = [k for k in MODELS if os.path.exists(os.path.join(ROOT, k, "scores.npz"))]
    res = {k: analyse(k) for k in keys}
    held = [r for r in res.values() if not r["dev"]]
    agg = {"n_heldout": len(held),
           "B1_switch_gt_greedy": sum(r["switch_minus_greedy_ci"][0] > 0 for r in held),
           "B1_switch_ge_majority": sum(r["acc"]["switch"] >= r["acc"]["majority"] for r in held)}
    for lam in LAMBDAS:
        L = [r["reliability"][f"lambda{lam}"] for r in held]
        agg[f"B2_lambda{lam}"] = sum(x["ours_minus_best_ci"][0] > 0 for x in L)
        agg[f"B3_lambda{lam}"] = sum(x["matched_coverage"]["ours"] > x["matched_coverage"]["selfconsistency"] for x in L)
    agg["B1"] = agg["B1_switch_gt_greedy"] >= 6 and agg["B1_switch_ge_majority"] >= 6
    agg["B2"] = all(agg[f"B2_lambda{l}"] >= 6 for l in LAMBDAS)
    agg["B3"] = all(agg[f"B3_lambda{l}"] >= 6 for l in LAMBDAS)
    json.dump({"per_model": res, "aggregate": agg}, open(os.path.join(ROOT, "path_b.json"), "w"), indent=1)
    for k, r in res.items():
        l1 = r["reliability"]["lambda1"]
        print(f"{k:14s} {'DEV ' if r['dev'] else ''}acc greedy {r['acc']['greedy']:.3f} maj {r['acc']['majority']:.3f} "
              f"switch {r['acc']['switch']:.3f} ci {np.round(r['switch_minus_greedy_ci'], 3)} | R1 ours "
              f"{l1['R']['ours']['test']:.3f} best({l1['best_baseline']}) {l1['R'][l1['best_baseline']]['test']:.3f} "
              f"ci {np.round(l1['ours_minus_best_ci'], 3)}")
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    main()
