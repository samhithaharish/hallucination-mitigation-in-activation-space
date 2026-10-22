"""Analysis for docs/prereg_context_select.md (CPU; the cross-encoder runs on CPU too).

    python -m experiments.layer_scaling.context_select_analyze --model llama32_3b
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from experiments.layer_scaling.analyze import CS, fit_logit, load_split
from experiments.layer_scaling.context_select import OUT
from experiments.layer_scaling.drift_analyze import probe_axis
from experiments.paths import SCALE_RUN
from experiments.recovery_scale.actions import public_example, ranked

RESULTS = Path(__file__).resolve().parents[2] / 'results' / 'context_select'
CROSS_ENCODER = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


def load(model):
    rows, states, conf = [], [], []
    layers = None
    for jp in sorted((OUT / model).glob('batch_*.json')):
        b = json.loads(jp.read_text())
        layers = b['layers']
        rows += b['rows']
        with np.load(jp.with_suffix('.npz')) as z:
            states.append(z['states'])
            conf.append(z['confidence'])
    j = json.loads((OUT / model / 'judged.json').read_text())
    if j['keys'] != [[r['id'], *r['pair']] for r in rows]:
        raise ValueError('judge verdicts misaligned')
    return rows, np.concatenate(states), np.concatenate(conf), np.array(j['correct']), layers


def boot_diff(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(a, float) - np.asarray(b, float)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {'mean': float(d.mean()), 'low': float(np.percentile(bs, 2.5)), 'high': float(np.percentile(bs, 97.5))}


def analyze(model):
    rows, S, conf, y, layers = load(model)
    by_q = defaultdict(list)
    for k, r in enumerate(rows):
        by_q[r['qi']].append(k)
    qs = sorted(by_q)
    raw = json.loads((SCALE_RUN / 'data' / 'test.json').read_text())

    # 1. zero-shot probe: the drift-study axis (baseline train states, layer chosen on validation)
    ax = probe_axis(model, 0)
    if ax['layer'] not in layers:
        raise ValueError(f"axis layer {ax['layer']} not captured ({layers})")
    tr, va = load_split(model, 'train'), load_split(model, 'validation')
    l = ax['layer'] - 1
    sc, m, _ = fit_logit(tr['states'][:, l, 0].astype(np.float32), (tr['f1'][:, 0] >= .5).astype(int),
                         va['states'][:, l, 0].astype(np.float32), (va['f1'][:, 0] >= .5).astype(int))
    li = layers.index(ax['layer'])
    zero = m.decision_function(sc.transform(S[:, li, 0].astype(np.float32)))

    # 4. cross-encoder over (question, paragraph)
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(CROSS_ENCODER, device='cpu')
    ce_pick = {}
    for q in qs:
        e = public_example(raw[q])
        s = ce.predict([(e['question'], p['title'] + '. ' + p['text']) for p in e['paragraphs']])
        ce_pick[q] = set(np.argsort(-s)[:2].tolist())

    # 6. pair-trained probe on questions 1-500, prompt-end states, layer and C by grouped 5-fold CV
    train_q = set(qs[:500])
    tr_idx = np.array([k for q in qs[:500] for k in by_q[q]])
    groups = np.array([rows[k]['qi'] for k in tr_idx])
    best = None
    for lj in range(len(layers)):
        X = S[tr_idx, lj, 0].astype(np.float32)
        for C in CS:
            aucs = []
            for a, b in GroupKFold(5).split(X, y[tr_idx], groups):
                s2 = StandardScaler().fit(X[a])
                mm = LogisticRegression(C=C, max_iter=2000).fit(s2.transform(X[a]), y[tr_idx][a])
                aucs.append(roc_auc_score(y[tr_idx][b], mm.decision_function(s2.transform(X[b]))))
            if best is None or np.mean(aucs) > best[0]:
                best = (float(np.mean(aucs)), lj, C)
    _, lj, C = best
    s2 = StandardScaler().fit(S[tr_idx, lj, 0].astype(np.float32))
    pm = LogisticRegression(C=C, max_iter=2000).fit(s2.transform(S[tr_idx, lj, 0].astype(np.float32)), y[tr_idx])
    trained = pm.decision_function(s2.transform(S[:, lj, 0].astype(np.float32)))

    rng = np.random.default_rng(20261022)
    pick = {name: [] for name in ('probe_zero_shot', 'bm25_top2', 'random', 'cross_encoder', 'confidence',
                                  'probe_pair_trained', 'gold_pair', 'oracle')}
    test_q = [q for q in qs if q not in train_q]
    for q in test_q:
        ks = by_q[q]
        e = public_example(raw[q])
        top2 = set(ranked(e)[:2])
        support = set(rows[ks[0]]['support'])
        acc = lambda k: int(y[k])
        pick['probe_zero_shot'].append(acc(ks[int(np.argmax(zero[ks]))]))
        pick['probe_pair_trained'].append(acc(ks[int(np.argmax(trained[ks]))]))
        pick['confidence'].append(acc(ks[int(np.argmax(conf[ks, 0]))]))
        pick['random'].append(acc(ks[rng.integers(len(ks))]))
        bm = [k for k in ks if set(rows[k]['pair']) == top2]
        pick['bm25_top2'].append(acc(bm[0]) if bm else acc(ks[0]))
        cx = [k for k in ks if set(rows[k]['pair']) == ce_pick[q]]
        pick['cross_encoder'].append(acc(cx[0]) if cx else acc(ks[0]))
        gp = [k for k in ks if set(rows[k]['titles']) == support]
        pick['gold_pair'].append(acc(gp[0]) if gp else acc(ks[0]))
        pick['oracle'].append(int(y[ks].max()))
    out = {'model': model, 'n_test_questions': len(test_q), 'axis_layer': ax['layer'],
           'pair_trained': {'layer': layers[lj], 'C': C, 'cv_auroc': best[0]},
           'accuracy': {k: float(np.mean(v)) for k, v in pick.items()},
           'C1_zero_shot_minus_bm25': boot_diff(pick['probe_zero_shot'], pick['bm25_top2']),
           'C2_zero_shot_minus_cross_encoder': boot_diff(pick['probe_zero_shot'], pick['cross_encoder']),
           'C3_trained_minus_zero_shot': boot_diff(pick['probe_pair_trained'], pick['probe_zero_shot']),
           'zero_shot_minus_confidence': boot_diff(pick['probe_zero_shot'], pick['confidence']),
           'trained_minus_cross_encoder': boot_diff(pick['probe_pair_trained'], pick['cross_encoder']),
           'candidate_auroc_zero_shot_test': float(roc_auc_score(
               y[[k for q in test_q for k in by_q[q]]], zero[[k for q in test_q for k in by_q[q]]]))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='llama32_3b')
    a = ap.parse_args()
    res = analyze(a.model)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f'{a.model}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
