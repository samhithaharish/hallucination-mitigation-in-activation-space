"""Detection baselines vs the question-level probe, on the held-out split of the mixed-vs-separated sweep.

CPU only; reads results/<root>/<key>/scores.npz (written by mixed_vs_separated.py).
Signals scored for "the greedy answer is wrong" (AUROC):
  probe(question)   - logistic probe on question-token states at the best layer, read before generation
  -logprob(greedy)  - minus the mean token log-prob of the model's own greedy answer
  1-agreement       - one minus the share of 10 samples that match the greedy answer

    python scripts/detection_baselines.py                                   # TriviaQA
    python scripts/detection_baselines.py --root results/nq/mixed_vs_separated
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

# scored before the answer_sep fix (wrong tokenization of the answer); rerun before using its log-prob
STALE_LOGPROB = {"mistral-7b"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/mixed_vs_separated")
    args = ap.parse_args()
    rows = []
    print(f"{'model':15s} {'probe(question)':>15s} {'-logprob(greedy)':>17s} {'1-agreement':>12s}  n_test")
    for path in sorted(glob.glob(os.path.join(args.root, "*", "scores.npz"))):
        key = path.split(os.sep)[-2]
        z = np.load(path)
        y, te = z["y"], z["te"]
        lp, ag = np.full(len(y), np.nan), np.full(len(y), np.nan)
        for j in np.where(z["cgreedy"] == 1)[0]:
            q = z["cq"][j]
            if np.isnan(lp[q]):
                lp[q], ag[q] = z["clogp"][j], z["cfreq"][j]
        ok = te[~np.isnan(lp[te])]
        r = {"key": key, "n_test": int(len(ok)),
             "probe_question": roc_auc_score(y[ok], z["p_h"][ok]),
             "neg_logprob_greedy": None if key in STALE_LOGPROB else roc_auc_score(y[ok], -lp[ok]),
             "one_minus_agreement": roc_auc_score(y[ok], -ag[ok])}
        rows.append(r)
        lp_s = "  stale (rerun)" if r["neg_logprob_greedy"] is None else f"{r['neg_logprob_greedy']:17.3f}"
        print(f"{key:15s} {r['probe_question']:15.3f} {lp_s:>17s} {r['one_minus_agreement']:12.3f}  {r['n_test']}")
    json.dump(rows, open(os.path.join(args.root, "detection_baselines.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
