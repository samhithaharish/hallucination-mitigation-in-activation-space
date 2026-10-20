"""Analysis for docs/prereg_input_drift.md (CPU only).

    python -m experiments.layer_scaling.drift_analyze --model llama32_3b
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from experiments.layer_scaling.analyze import fit_logit, load_split
from experiments.layer_scaling.drift import CONDITIONS, OUT

INFORMATIVE = ('more_evidence', 'oracle_evidence', 'metadata', 'warning')
RESULTS = Path(__file__).resolve().parents[2] / 'results' / 'input_drift'


def probe_axis(model, pos):
    """Correct/wrong probe on baseline train states: layer and C chosen on validation (keep F1 >= 0.5)."""
    tr, va = load_split(model, 'train'), load_split(model, 'validation')
    ytr, yva = (tr['f1'][:, 0] >= 0.5).astype(int), (va['f1'][:, 0] >= 0.5).astype(int)
    best = None
    for l in range(tr['states'].shape[1]):
        sc, m, auc = fit_logit(tr['states'][:, l, pos].astype(np.float32), ytr,
                               va['states'][:, l, pos].astype(np.float32), yva)
        if best is None or auc > best[0]:
            best = (auc, l, sc, m)
    auc, l, sc, m = best
    return {'layer': l + 1, 'val_auroc': float(auc),
            'logit': lambda S: m.decision_function(sc.transform(S[:, l, pos].astype(np.float32)))}


def load_condition(model, c):
    d = OUT / model / c
    rows, states = [], []
    for jp in sorted(d.glob('batch_*.json')):
        rows += json.loads(jp.read_text())['rows']
        with np.load(jp.with_suffix('.npz')) as z:
            states.append(z['states'])
    j = json.loads((d / 'judged.json').read_text())
    if j['ids'] != [r['id'] for r in rows]:
        raise ValueError('judge verdicts misaligned: ' + c)
    return {'ids': [r['id'] for r in rows], 'states': np.concatenate(states),
            'judged': np.array(j['correct']), 'f1ok': np.array([r['f1'] >= 0.5 for r in rows]).astype(int)}


def boot(values, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return {'mean': float(v.mean()), 'low': float(np.percentile(b, 2.5)), 'high': float(np.percentile(b, 97.5))}


def pooled_auc_ci(drift, fixed, qids, n=2000, seed=0):
    """AUROC pooled over (question, condition) pairs; bootstrap resamples questions."""
    rng = np.random.default_rng(seed)
    groups = {}
    for k, q in enumerate(qids):
        groups.setdefault(q, []).append(k)
    keys = list(groups)
    base = roc_auc_score(fixed, drift)
    bs = []
    for _ in range(n):
        idx = np.concatenate([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        if fixed[idx].min() != fixed[idx].max():
            bs.append(roc_auc_score(fixed[idx], drift[idx]))
    return {'auroc': float(base), 'low': float(np.percentile(bs, 2.5)), 'high': float(np.percentile(bs, 97.5)),
            'n_pairs': int(len(fixed)), 'n_fixed': int(fixed.sum())}


def analyze(model):
    C = {c: load_condition(model, c) for c in CONDITIONS}
    out = {'model': model, 'n': len(C['base']['ids'])}
    for pos, name in ((0, 'prompt_end'), (1, 'answer_end')):
        ax = probe_axis(model, pos)
        L = {c: ax['logit'](C[c]['states']) for c in CONDITIONS}
        base_wrong = C['base']['judged'] == 0
        res = {'axis_layer': ax['layer'], 'axis_val_auroc': ax['val_auroc'], 'conditions': {}}
        for c in CONDITIONS:
            d = L[c] - L['base']
            r = {'judged_acc': float(C[c]['judged'].mean()), 'f1_acc': float(C[c]['f1ok'].mean()),
                 'acc_change_vs_base': boot(C[c]['judged'] - C['base']['judged']), 'drift': boot(d)}
            if c != 'base':
                bw = base_wrong
                crossed = bw & (L['base'] < 0) & (L[c] > 0)
                r['D3_crossed_n'] = int(crossed.sum())
                r['D3_crossed_still_wrong'] = float((C[c]['judged'][crossed] == 0).mean()) if crossed.any() else None
                if C[c]['judged'][bw].min() != C[c]['judged'][bw].max():
                    r['fix_auroc_of_drift'] = float(roc_auc_score(C[c]['judged'][bw], d[bw]))
            res['conditions'][c] = r
        bw = np.where(base_wrong)[0]
        dr = np.concatenate([(L[c] - L['base'])[bw] for c in INFORMATIVE])
        fx = np.concatenate([C[c]['judged'][bw] for c in INFORMATIVE])
        q = np.concatenate([bw for _ in INFORMATIVE])
        res['D2_pooled'] = pooled_auc_ci(dr, fx, q)
        fd, ff = (L['filler'] - L['base'])[bw], C['filler']['judged'][bw]
        res['D2_filler_control'] = (pooled_auc_ci(fd, ff, bw) if ff.min() != ff.max() else None)
        res['D1_pass'] = bool(res['conditions']['oracle_evidence']['drift']['low'] > 0 and
                              res['conditions']['misleading']['drift']['high'] < 0)
        res['D2_pass'] = bool(res['D2_pooled']['low'] > 0.5)
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='llama32_3b')
    a = ap.parse_args()
    res = analyze(a.model)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f'{a.model}.json').write_text(json.dumps(res, indent=1))
    for name in ('prompt_end', 'answer_end'):
        r = res[name]
        print(f"== {name}: axis layer {r['axis_layer']} (val AUROC {r['axis_val_auroc']:.3f}); "
              f"D1 {r['D1_pass']}  D2 {r['D2_pass']} pooled AUROC {r['D2_pooled']['auroc']:.3f} "
              f"[{r['D2_pooled']['low']:.3f}, {r['D2_pooled']['high']:.3f}]")
        for c, v in r['conditions'].items():
            print(f"  {c:15s} acc {v['judged_acc']:.3f}  dAcc {v['acc_change_vs_base']['mean']:+.3f}  "
                  f"drift {v['drift']['mean']:+.3f} [{v['drift']['low']:+.3f},{v['drift']['high']:+.3f}]  "
                  f"fixAUROC {v.get('fix_auroc_of_drift', float('nan')):.3f}  "
                  f"crossed {v.get('D3_crossed_n', '-')} still-wrong {v.get('D3_crossed_still_wrong')}")


if __name__ == '__main__':
    main()
