"""Layer-wise separability of "which fix to apply", as pre-registered in docs/prereg_layer_separability.md.

CPU only; reads runs/layer_scaling/<model>/{train,validation,test}/batch_*.{json,npz}.

    python -m experiments.layer_scaling.analyze --model phi35            # full analysis -> results/layer_scaling/
    python -m experiments.layer_scaling.analyze --model phi35 --partial  # whatever train batches exist (smoke)
"""
import argparse
import json
from pathlib import Path

import os

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from experiments.layer_scaling.collect import OUT
from experiments.recovery_scale.actions import ACTIONS, CALLS

CS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_JOBS = int(os.environ.get('ANALYSIS_JOBS', '12'))
POSITIONS = ('prompt_end', 'answer_end')
CALLS_A = np.array(CALLS, dtype=float)
RESULTS = Path(__file__).resolve().parents[2] / 'results' / 'layer_scaling'


def load_split(model, split):
    rows, states, conf, meta = [], [], [], []
    for jp in sorted((OUT / model / split).glob('batch_*.json')):
        npz = jp.with_suffix('.npz')
        if not npz.exists():
            continue
        b = json.loads(jp.read_text())
        with np.load(npz) as z:
            states.append(z['states'])
            conf.append(z['confidence'])
        rows += b['rows']
        meta += [r['metadata'] for r in b['rows']]
    if not rows:
        raise FileNotFoundError(f'no batches for {model}/{split}')
    # keep-only collections (no recovery actions) leave the other actions as NaN
    f1 = np.array([[r['actions'][a]['score']['f1'] if a in r['actions'] else np.nan for a in ACTIONS] for r in rows])
    return {'states': np.concatenate(states), 'conf': np.concatenate(conf).astype(np.float64),
            'meta': np.array(meta, dtype=np.float64), 'f1': f1}


def targets(f1):
    gain = f1[:, 1:] - f1[:, :1]
    best = np.lexsort((CALLS_A[None].repeat(len(f1), 0).T, -f1.T), axis=0)[0]   # max F1, then fewest calls
    return {'T1': (gain.max(1) > 0).astype(int), 'T2': (gain > 0).astype(int), 'T3': best}


def fit_logit(Xtr, ytr, Xva, yva):
    """L2 strength picked on validation AUROC; returns (fitted scaler+model, val AUROC)."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xva)
    best = (-1, None)
    for C in CS:
        m = LogisticRegression(C=C, max_iter=2000).fit(A, ytr)
        auc = roc_auc_score(yva, m.predict_proba(B)[:, 1])
        if auc > best[0]:
            best = (auc, m)
    return sc, best[1], best[0]


def paired_ci(a, b, n_boot=4000, seed=0):
    """Mean of a-b with a 95% paired bootstrap interval (a, b: per-question values)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {'mean': float(d.mean()), 'low': float(np.percentile(boots, 2.5)),
            'high': float(np.percentile(boots, 97.5))}


def auc_diff_ci(y, s1, s2, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, s1) - roc_auc_score(y, s2)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        boots.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return {'mean': float(base), 'low': float(np.percentile(boots, 2.5)), 'high': float(np.percentile(boots, 97.5))}


def layer_curve(tr, va, ytr, yva):
    """Validation AUROC for every (layer, position); fits run in parallel across CPU cores."""
    L = tr['states'].shape[1]
    # pass per-layer slices as arguments (joblib memory-maps them) rather than closing over the full array
    aucs = Parallel(n_jobs=N_JOBS)(
        delayed(_val_auc)(tr['states'][:, l, p].astype(np.float32), ytr,
                          va['states'][:, l, p].astype(np.float32), yva)
        for l in range(L) for p in range(2))
    return np.array(aucs).reshape(L, 2)


def _val_auc(Xtr, ytr, Xva, yva):
    return fit_logit(Xtr, ytr, Xva, yva)[2]


def selector_f1(gain_pred, f1, lam):
    choice = np.argmax(np.c_[np.zeros(len(f1)), gain_pred - lam * CALLS_A[None, 1:]], axis=1)
    return f1[np.arange(len(f1)), choice], CALLS_A[choice]


def fit_selector(Xtr, f1tr, Xva, f1va, Xte, f1te, budget=1.0):
    sc = StandardScaler().fit(Xtr)
    A, B, Cx = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    gtr = f1tr[:, 1:] - f1tr[:, :1]
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        r = Ridge(alpha=alpha).fit(A, gtr)
        pva = r.predict(B)
        for lam in np.linspace(0, 1.0, 101):   # lambda = 1 forces keep, so the budget is always reachable
            s, calls = selector_f1(pva, f1va, lam)
            if calls.mean() <= budget and (best is None or s.mean() > best[0]):
                best = (s.mean(), alpha, lam)
    _, alpha, lam = best
    r = Ridge(alpha=alpha).fit(A, gtr)
    s, calls = selector_f1(r.predict(Cx), f1te, lam)
    return s, calls, {'alpha': alpha, 'lambda': float(lam)}


def analyze(model, partial=False):
    if partial:
        d = load_split(model, 'train')
        n = len(d['f1'])
        idx = np.random.default_rng(0).permutation(n)
        cut = [idx[:int(.6 * n)], idx[int(.6 * n):int(.8 * n)], idx[int(.8 * n):]]
        tr, va, te = ({k: v[c] for k, v in d.items()} for c in cut)
    else:
        tr, va, te = (load_split(model, s) for s in ('train', 'validation', 'test'))
    ttr, tva, tte = targets(tr['f1']), targets(va['f1']), targets(te['f1'])
    out = {'model': model, 'partial': partial, 'n': {'train': len(tr['f1']), 'validation': len(va['f1']),
           'test': len(te['f1'])}, 'n_layers': int(tr['states'].shape[1]),
           'base_rates': {'T1_fixable': float(tte['T1'].mean()),
                          'T2_helps': dict(zip(ACTIONS[1:], tte['T2'].mean(0).round(4).tolist()))},
           'mean_f1': dict(zip(ACTIONS, te['f1'].mean(0).round(4).tolist()))}

    # ---- T1: fixability, full depth curve on validation, test only the primary layer ----
    curve = layer_curve(tr, va, ttr['T1'], tva['T1'])
    l, p = np.unravel_index(np.argmax(curve), curve.shape)
    out['T1_val_curve'] = {pos: curve[:, j].round(4).tolist() for j, pos in enumerate(POSITIONS)}
    out['T1_primary'] = {'layer': int(l) + 1, 'position': POSITIONS[p], 'depth': float((l + 1) / curve.shape[0])}
    X = lambda d: d['states'][:, l, p].astype(np.float32)
    sc, m, _ = fit_logit(X(tr), ttr['T1'], X(va), tva['T1'])
    s_act = m.predict_proba(sc.transform(X(te)))[:, 1]
    scc, mc, _ = fit_logit(tr['conf'], ttr['T1'], va['conf'], tva['T1'])
    s_conf = mc.predict_proba(scc.transform(te['conf']))[:, 1]
    cm = lambda d: np.c_[d['conf'], d['meta']]
    scm, mm, _ = fit_logit(cm(tr), ttr['T1'], cm(va), tva['T1'])
    s_cm = mm.predict_proba(scm.transform(cm(te)))[:, 1]
    shuf = np.random.default_rng(1).permutation(ttr['T1'])
    scs, ms, _ = fit_logit(X(tr), shuf, X(va), tva['T1'])
    s_shuf = ms.predict_proba(scs.transform(X(te)))[:, 1]
    scx = StandardScaler().fit(X(tr))
    mlp = MLPClassifier(hidden_layer_sizes=(256,), alpha=1e-2, early_stopping=True, max_iter=200,
                        random_state=0).fit(scx.transform(X(tr)), ttr['T1'])
    s_mlp = mlp.predict_proba(scx.transform(X(te)))[:, 1]
    y = tte['T1']
    out['T1_test'] = {'activation_auroc': roc_auc_score(y, s_act), 'confidence_auroc': roc_auc_score(y, s_conf),
                      'confidence_meta_auroc': roc_auc_score(y, s_cm), 'shuffled_auroc': roc_auc_score(y, s_shuf),
                      'mlp_auroc': roc_auc_score(y, s_mlp),
                      'H1_activation_minus_confidence': auc_diff_ci(y, s_act, s_conf),
                      'activation_minus_confidence_meta': auc_diff_ci(y, s_act, s_cm),
                      'mlp_minus_linear': auc_diff_ci(y, s_mlp, s_act)}

    # ---- T2: per action, primary layer chosen per action on validation (prompt or answer position) ----
    t2 = {}
    for j, a in enumerate(ACTIONS[1:]):
        if any(t['T2'][:, j].min() == t['T2'][:, j].max() for t in (ttr, tva, tte)):
            continue   # this fix never (or always) helps in some split: AUROC undefined
        cv = layer_curve(tr, va, ttr['T2'][:, j], tva['T2'][:, j])
        la, pa = np.unravel_index(np.argmax(cv), cv.shape)
        Xa = lambda d: d['states'][:, la, pa].astype(np.float32)
        sa, ma, _ = fit_logit(Xa(tr), ttr['T2'][:, j], Xa(va), tva['T2'][:, j])
        sc2, mc2, _ = fit_logit(tr['conf'], ttr['T2'][:, j], va['conf'], tva['T2'][:, j])
        ya = tte['T2'][:, j]
        pa_s, pc_s = ma.predict_proba(sa.transform(Xa(te)))[:, 1], mc2.predict_proba(sc2.transform(te['conf']))[:, 1]
        t2[a] = {'layer': int(la) + 1, 'position': POSITIONS[pa], 'base_rate': float(ya.mean()),
                 'activation_auroc': roc_auc_score(ya, pa_s), 'confidence_auroc': roc_auc_score(ya, pc_s),
                 'activation_minus_confidence': auc_diff_ci(ya, pa_s, pc_s),
                 'val_curve': {pos: cv[:, k].round(4).tolist() for k, pos in enumerate(POSITIONS)}}
    out['T2_test'] = t2
    out['H2_actions_passing'] = [a for a, v in t2.items() if v['activation_minus_confidence']['low'] > 0]

    # ---- T3 / H3: selector at mean extra calls <= 1.0 ----
    s_a, c_a, h_a = fit_selector(X(tr), tr['f1'], X(va), va['f1'], X(te), te['f1'])
    s_c, c_c, h_c = fit_selector(tr['conf'], tr['f1'], va['conf'], va['f1'], te['conf'], te['f1'])
    fixed = {a: te['f1'][:, j] for j, a in enumerate(ACTIONS) if CALLS[j] <= 1}
    best_fixed = max(fixed, key=lambda a: fixed[a].mean())
    out['selector_test'] = {
        'keep_f1': float(te['f1'][:, 0].mean()), 'oracle_f1': float(te['f1'].max(1).mean()),
        'best_fixed_action_le1call': best_fixed, 'best_fixed_f1': float(fixed[best_fixed].mean()),
        'activation_f1': float(s_a.mean()), 'activation_calls': float(c_a.mean()), 'activation_hparams': h_a,
        'confidence_f1': float(s_c.mean()), 'confidence_calls': float(c_c.mean()), 'confidence_hparams': h_c,
        'H3_activation_minus_confidence': paired_ci(s_a, s_c),
        'activation_minus_best_fixed': paired_ci(s_a, fixed[best_fixed])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--partial', action='store_true')
    a = ap.parse_args()
    res = analyze(a.model, a.partial)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{a.model}{'_partial' if a.partial else ''}.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    t1 = res['T1_test']
    print(json.dumps({'n': res['n'], 'T1_primary': res['T1_primary'],
                      'T1': {k: round(v, 3) if isinstance(v, float) else v for k, v in t1.items()},
                      'H2_actions_passing': res['H2_actions_passing'],
                      'selector': res['selector_test']}, indent=1, default=float))
    print('saved', path)


if __name__ == '__main__':
    main()
