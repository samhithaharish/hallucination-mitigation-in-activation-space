"""Analysis for docs/prereg_prior_work_baselines.md (CPU).

    python -m experiments.layer_scaling.baselines_analyze --model llama32_3b [--judge laya]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from experiments.layer_scaling.analyze import fit_logit, load_split
from experiments.layer_scaling.baselines import AXIS, OUT, TEST_Q
from experiments.layer_scaling.context_select_analyze import boot_diff, load as load_ctx

RESULTS = Path(__file__).resolve().parents[2] / 'results' / 'prior_work'


def baseline_probe(model, layer, kind):
    """Probe on the 6,000 baseline train states (prompt end, block-`layer` output); label keep F1 >= 0.5."""
    tr, va = load_split(model, 'train'), load_split(model, 'validation')
    Xtr, Xva = tr['states'][:, layer - 1, 0].astype(np.float32), va['states'][:, layer - 1, 0].astype(np.float32)
    ytr, yva = (tr['f1'][:, 0] >= .5).astype(int), (va['f1'][:, 0] >= .5).astype(int)
    if kind == 'logit':
        sc, m, _ = fit_logit(Xtr, ytr, Xva, yva)
        return lambda X: m.decision_function(sc.transform(X.astype(np.float32)))
    sc = StandardScaler().fit(Xtr)        # Jin-style: small MLP with early stopping
    m = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, max_iter=300,
                      random_state=0).fit(sc.transform(Xtr), ytr)
    return lambda X: m.predict_proba(sc.transform(X.astype(np.float32)))[:, 1]


def top2_pair(scores):
    return tuple(sorted(np.argsort(-np.asarray(scores))[:2].tolist()))


def analyze(model, judge):
    rows, S, conf, y, layers = load_ctx(model, judge)
    acc = {(r['qi'], *sorted(r['pair'])): int(v) for r, v in zip(rows, y)}
    by_q = defaultdict(list)
    for k, r in enumerate(rows):
        by_q[r['qi']].append(k)
    sg, pr = np.load(OUT / model / 'singles.npz'), np.load(OUT / model / 'pairs.npz')
    mid, axis = [int(x) for x in sg['layers']]
    assert axis == AXIS[model]

    ours = baseline_probe(model, axis, 'logit')
    jin = baseline_probe(model, mid, 'mlp')
    ours_mid = baseline_probe(model, mid, 'logit')
    s_ours_pair = ours(S[:, layers.index(axis), 0])           # as in the context-selection pilot
    s_ours_single, s_jin = ours(sg['states'][:, 1]), jin(sg['states'][:, 0])
    s_mid_pair = ours_mid(pr['states'][:, 0])
    pair_key_idx = {tuple(k): i for i, k in enumerate(pr['keys'].tolist())}

    singles_by_q = defaultdict(list)
    for i, (qi, para) in enumerate(sg['keys'].tolist()):
        singles_by_q[qi].append(i)
    bge = json.loads((OUT / 'bge_scores.json').read_text())['scores']
    seakr = None
    if (OUT / model / 'seakr.json').exists():
        sk = json.loads((OUT / model / 'seakr.json').read_text())
        seakr = defaultdict(list)
        for key, sc in zip(sk['keys'], sk['score']):
            seakr[key[0]].append((sc, tuple(sorted(key[1:]))))

    picks = defaultdict(list)
    for qi in TEST_Q:
        ks = by_q[qi]
        if len(singles_by_q[qi]) < 2:
            continue
        get = lambda pair: acc[(qi, *pair)]
        picks['ours_pair'].append(int(y[ks[int(np.argmax(s_ours_pair[ks]))]]))
        si = singles_by_q[qi]
        picks['ours_single'].append(get(top2_pair(s_ours_single[si])))
        picks['jin_single'].append(get(top2_pair(s_jin[si])))
        cand = [(s_mid_pair[pair_key_idx[(qi, *sorted(rows[k]['pair']))]], k) for k in ks]
        picks['mid_layer_pair'].append(int(y[max(cand)[1]]))
        picks['bge'].append(get(top2_pair(bge[str(qi)])))
        if seakr is not None:
            picks['seakr'].append(get(min(seakr[qi])[1]))
    out = {'model': model, 'judge': judge, 'n_questions': len(picks['ours_pair']), 'mid_layer': mid,
           'axis_layer': axis, 'accuracy': {k: float(np.mean(v)) for k, v in picks.items()}}
    for k in picks:
        if k != 'ours_pair' and len(picks[k]) == len(picks['ours_pair']):
            out[f'ours_pair_minus_{k}'] = boot_diff(picks['ours_pair'], picks[k])
    cost = {'ours_pair': '45 forward passes', 'ours_single': '10 forward passes', 'jin_single': '10 forward passes',
            'mid_layer_pair': '45 forward passes', 'bge': '10 reranker passes (no LLM)',
            'seakr': '45 x 8 sampled generations + 360 forward passes'}
    out['cost_per_question'] = {k: cost[k] for k in picks}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='llama32_3b')
    ap.add_argument('--judge', choices=['qwen', 'laya'], default='qwen')
    a = ap.parse_args()
    res = analyze(a.model, a.judge)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{a.model}{'' if a.judge == 'qwen' else '_laya'}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
