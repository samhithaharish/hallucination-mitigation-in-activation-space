"""Pre-scoring review amendment: penalize unlabelled displacement prediction error."""
import argparse
import json
import time

import numpy as np
from experiments.activation_drift import pilot


def freeze(key):
    info=json.loads((pilot.ROOT/key/'geometry_fit.json').read_text())
    errors=info['selection']['operators']
    record={'time_unix':time.time(),'operator_penalty':[0.]+[v['validation_mse'] for v in errors],
        'mean_penalty':[0.]+[v['mean_validation_mse'] for v in errors],
        'source_sha256':pilot.sha(pilot.CODE/'supplement.py'),
        'amendment_sha256':pilot.sha(pilot.CODE/'README.md'),
        'geometry_sha256':pilot.sha(pilot.ROOT/key/'geometry.joblib'),
        'action_success_labels_used':False,'multiplier':1.}
    path=pilot.ROOT/key/'supplement_frozen.json'
    if path.exists():
        raise ValueError('Supplement already frozen')
    pilot.save(path,record)


def evaluate(key):
    from experiments.activation_drift import analyze
    record=json.loads((pilot.ROOT/key/'supplement_frozen.json').read_text())
    if record['source_sha256']!=pilot.sha(pilot.CODE/'supplement.py'):
        raise ValueError('Changed supplementary source')
    if record['geometry_sha256']!=pilot.sha(pilot.ROOT/key/'geometry.joblib'):
        raise ValueError('Changed geometry')
    if record['amendment_sha256']!=pilot.sha(pilot.CODE/'README.md'):
        raise ValueError('Changed supplementary design')
    scores={name:[] for name in ('mean','operator')}; risk=[]
    for path in sorted((pilot.ROOT/key/'test').glob('predictions_*.npz')):
        z=np.load(path); risk.append(z['risk'])
        for name in scores:
            scores[name].append(z[name]-np.array(record[name+'_penalty']))
    scores={k:np.concatenate(v) for k,v in scores.items()}; risk=np.concatenate(risk)
    # Save the full set of supplement choices before opening scored outcomes.
    predpath=pilot.ROOT/key/'supplement_predictions.npz'
    np.savez_compressed(predpath,**scores)
    raw=np.load(pilot.ROOT/key/'scored_test.npz'); f1,em=raw['f1'],raw['em']
    rows=[]
    for p in sorted((pilot.ROOT/key/'test').glob('batch_*.json')):
        rows.extend(json.loads(p.read_text())['rows'])
    tokens=np.array([[a['output_tokens'] for a in row['actions']] for row in rows])
    result={}; row=np.arange(len(f1)); best=int(f1.mean(0).argmax())
    for name,score in scores.items():
        choices=score.argmax(1); quality=f1[row,choices]
        stats=analyze.policy_stats(f1,em,choices,tokens)
        stats['vs_keep']=analyze.interval(quality-f1[:,0])
        stats['vs_best_fixed']=analyze.interval(quality-f1[:,best])
        stats['vs_risk_matched_shuffle']=analyze.risk_matched_interval(f1,choices,risk)
        result[name+'_error_penalty']=stats
    pilot.save(pilot.ROOT/key/'supplement_evaluation.json',{'policies':result,'freeze':record,
        'predictions_sha256':pilot.sha(predpath),'best_fixed':pilot.ACTIONS[best]})
    print(json.dumps({'model':key,'supplement':result},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,choices=('qwen7b','phi35'))
    p.add_argument('--phase',choices=('freeze','evaluate'),required=True); args=p.parse_args()
    (freeze if args.phase=='freeze' else evaluate)(args.model)
