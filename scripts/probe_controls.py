"""What does the error-risk probe measure? Two controls on the saved mixed-vs-separated sweep (CPU only).

Exp 1  Lexical control: TF-IDF of the prompt (question; the instruction and chat template are constant) ->
       greedy answer wrong, same train/test split and labels as the activation probe. Also the model's own
       confidence (-mean log-prob of its greedy answer) and an incremental test: TF-IDF + probe score stacked.
Exp 2  Knowledge control: correlation of the probe's risk score with K(q) = share of 10 samples that are correct.

    python scripts/probe_controls.py                                    # TriviaQA
    python scripts/probe_controls.py --root results/nq/mixed_vs_separated
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

STALE_LOGPROB = {"mistral-7b"}     # scored before the answer_sep fix; rerun before using its log-prob
CS = [0.01, 0.1, 1.0, 10.0]


def auc_ci(y, s, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            b.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def diff_ci(y, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            d.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    return {"mean": float(roc_auc_score(y, a) - roc_auc_score(y, b)),
            "low": float(np.percentile(d, 2.5)), "high": float(np.percentile(d, 97.5))}


def tfidf_scores(texts, y, tr, te):
    """C chosen by 5-fold CV on train; returns out-of-fold train scores and test scores."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform([texts[i] for i in tr]), vec.transform([texts[i] for i in te])
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    best = max(CS, key=lambda C: roc_auc_score(y[tr], cross_val_predict(
        LogisticRegression(C=C, max_iter=3000), Xtr, y[tr], cv=cv, method="decision_function")))
    oof = cross_val_predict(LogisticRegression(C=best, max_iter=3000), Xtr, y[tr], cv=cv, method="decision_function")
    m = LogisticRegression(C=best, max_iter=3000).fit(Xtr, y[tr])
    return oof, m.decision_function(Xte), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/mixed_vs_separated")
    a = ap.parse_args()
    out = []
    for d in sorted(glob.glob(os.path.join(a.root, "*", "scores.npz"))):
        key = d.split(os.sep)[-2]
        z = np.load(d)
        recs = json.load(open(os.path.join(os.path.dirname(d), "records.json")))
        y, tr, te, p = z["y"], z["tr"], z["te"], z["p_h"]          # y = 1: greedy answer wrong
        texts = [r["prompt"] for r in recs]
        oof, s_tfidf, C = tfidf_scores(texts, y, tr, te)
        logit = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
        stack = LogisticRegression(max_iter=1000).fit(np.c_[oof, logit(p[tr])], y[tr])
        s_both = stack.decision_function(np.c_[s_tfidf, logit(p[te])])
        r = {"model": key, "n_test": int(len(te)), "error_rate": float(y.mean()), "tfidf_C": C,
             "probe_auroc": float(roc_auc_score(y[te], p[te])), "probe_ci": auc_ci(y[te], p[te]),
             "tfidf_auroc": float(roc_auc_score(y[te], s_tfidf)), "tfidf_ci": auc_ci(y[te], s_tfidf),
             "probe_minus_tfidf": diff_ci(y[te], p[te], s_tfidf),
             "stacked_auroc": float(roc_auc_score(y[te], s_both)),
             "stacked_minus_tfidf": diff_ci(y[te], s_both, s_tfidf)}
        lp = np.full(len(y), np.nan)
        for j in np.where(z["cgreedy"] == 1)[0]:
            if np.isnan(lp[z["cq"][j]]):
                lp[z["cq"][j]] = z["clogp"][j]
        ok = te[~np.isnan(lp[te])]
        r["confidence_auroc"] = None if key in STALE_LOGPROB else float(roc_auc_score(y[ok], -lp[ok]))
        k = np.array([rr["n_ok"] for rr in recs]) / 10.0               # Exp 2: K(q)
        rho = spearmanr(p[te], k[te])
        r["spearman_probe_vs_K"] = float(rho.correlation)
        r["probe_risk_by_K"] = {f"{int(v * 10)}/10": float(p[te][k[te] == v].mean())
                                for v in np.unique(k[te]) if (k[te] == v).sum() >= 10}
        r["K_auroc"] = float(roc_auc_score(y[te], -k[te]))
        out.append(r)
        print(f"{key:14s} probe {r['probe_auroc']:.3f}  tfidf {r['tfidf_auroc']:.3f}  "
              f"probe-tfidf {r['probe_minus_tfidf']['mean']:+.3f} [{r['probe_minus_tfidf']['low']:+.3f},"
              f"{r['probe_minus_tfidf']['high']:+.3f}]  stacked-tfidf {r['stacked_minus_tfidf']['mean']:+.3f}  "
              f"conf {r['confidence_auroc'] if r['confidence_auroc'] is None else round(r['confidence_auroc'], 3)}  "
              f"rho(probe,K) {r['spearman_probe_vs_K']:+.3f}  K-auroc {r['K_auroc']:.3f}", flush=True)
    json.dump(out, open(os.path.join(a.root, "probe_controls.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
