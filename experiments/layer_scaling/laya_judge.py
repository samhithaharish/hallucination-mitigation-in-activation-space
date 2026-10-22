"""Laya as a second, independent answer judge: agreement check, then full re-grading.

Runs in the isolated .venv-laya (pip install laya). Two question formats, since the Laya model card warns
that its yes/no ("noul") type can follow its option labels instead of the input:
  noul   : P(yes) to "Does the proposed answer refer to the same thing as the reference answer?"
  choice : two labelled options, read as the probability of "same"

    .venv-laya/bin/python -m experiments.layer_scaling.laya_judge agree --n 1000   # vs Qwen judge and string F1
    .venv-laya/bin/python -m experiments.layer_scaling.laya_judge grade --format choice   # all candidates
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
CTX = ROOT / 'runs' / 'context_select' / 'llama32_3b'
OUT = ROOT / 'results' / 'context_select'

QUESTIONS = {
    'noul': {'same': {'type': 'noul', 'instructions':
                      'Does the proposed answer refer to the same thing as the reference answer?'}},
    'choice': {'same': {'type': 'choice', 'instructions':
                        'Does the proposed answer refer to the same thing as the reference answer?',
                        'criteria': {'same': 'yes, the proposed answer means the same as the reference answer',
                                     'different': 'no, the proposed answer is different from the reference answer'}}},
}


def load_items():
    rows = [r for jp in sorted(CTX.glob('batch_*.json')) for r in json.loads(jp.read_text())['rows']]
    qwen = json.loads((CTX / 'judged.json').read_text())['correct']
    for r, v in zip(rows, qwen):
        r['qwen'] = int(v)
        r['f1ok'] = int(r['f1'] >= 0.5)
    return rows


def state(r):
    return f"Question: {r['question']}\nReference answer: {r['gold']}\nProposed answer: {r['answer'] or '(no answer)'}"


def p_same(router, r, fmt):
    a = router.predict(state(r), QUESTIONS[fmt])['answers']['same']
    if fmt == 'noul':
        return float(a['noul'])
    probs = a.get('probabilities')
    if isinstance(probs, dict) and 'same' in probs:
        return float(probs['same'])
    return float(a['choice'] == 'same')


def agree(n):
    from laya import Router
    rows = load_items()
    random.Random(0).shuffle(rows)
    sample = rows[:n]
    router = Router()
    res = {}
    for fmt in QUESTIONS:
        p = np.array([p_same(router, r, fmt) for r in sample])
        yhat = (p > 0.5).astype(int)
        q, f = np.array([r['qwen'] for r in sample]), np.array([r['f1ok'] for r in sample])
        res[fmt] = {'agree_qwen': float((yhat == q).mean()), 'kappa_qwen': float(cohen_kappa_score(yhat, q)),
                    'agree_string': float((yhat == f).mean()), 'kappa_string': float(cohen_kappa_score(yhat, f)),
                    'laya_yes_rate': float(yhat.mean()), 'p_mean': float(p.mean())}
    q, f = np.array([r['qwen'] for r in sample]), np.array([r['f1ok'] for r in sample])
    res['reference'] = {'n': n, 'qwen_vs_string_agree': float((q == f).mean()),
                        'qwen_vs_string_kappa': float(cohen_kappa_score(q, f)), 'qwen_yes_rate': float(q.mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'laya_agreement.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def grade(fmt):
    from laya import Router
    rows = load_items()
    router = Router()
    p = [p_same(router, r, fmt) for r in rows]
    (CTX / f'laya_{fmt}.json').write_text(json.dumps({'format': fmt, 'keys': [[r['id'], *r['pair']] for r in rows],
                                                      'p_same': p}))
    print(json.dumps({'graded': len(p), 'yes_rate': float(np.mean(np.array(p) > 0.5))}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['agree', 'grade'])
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--format', choices=list(QUESTIONS), default='choice')
    a = ap.parse_args()
    agree(a.n) if a.step == 'agree' else grade(a.format)
