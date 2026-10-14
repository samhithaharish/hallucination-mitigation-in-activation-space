"""Score new test outcomes only after the error-field models are frozen."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score,log_loss

from experiments.paths import REPO
from experiments.paired_evidence.data import OUT
from experiments.paired_evidence.task import ACTIONS,CONDITIONS
from experiments.paired_evidence.evaluate import (arrays,frozen_predictions,quality,stats,interval,
    matched_interval,shuffle_interval,gap_scores)
from experiments.recovery_help.data import dump,file_digest

POLICIES=('choices','class_rule','observable_rule','confidence_rule','condition_observable_rule',
          'condition_confidence_rule','condition_field_choices')


def evaluate(key,draws=4000):
    path=OUT/key/'fitted/best.joblib'
    metadata=json.loads(path.with_suffix('.json').read_text())
    for relative,expected in metadata['source_hashes'].items():
        if file_digest(REPO/relative)!=expected:raise ValueError('Changed frozen source: '+relative)
    if file_digest(OUT/'data_manifest.json')!=metadata['data_manifest_sha256']:
        raise ValueError('Changed frozen cohort manifest')
    data,rows,cases,f1,em,tokens=arrays(key,'test')
    prediction,audit=frozen_predictions(key,'test',data)
    best=int(f1.mean((0,1)).argmax());policies={}
    for name in POLICIES:
        choices=prediction[name];q=quality(f1,choices)
        policies[name]={**stats(f1,em,tokens,choices),
            'vs_keep':interval(q-f1[:,:,0],draws),
            'vs_hindsight_best_fixed':interval(q-f1[:,:,best],draws),
            'vs_matched_risk_fixed':matched_interval(f1,choices,prediction['risk'],draws),
            'vs_risk_matched_shuffle':shuffle_interval(f1[:,:,:4],choices,prediction['risk'],draws)}
    primary=quality(f1,prediction['choices'])
    for name in POLICIES[1:]:
        policies['choices']['vs_'+name]=interval(primary-quality(f1,prediction[name]),draws)
    controls=('vs_keep','vs_matched_risk_fixed','vs_risk_matched_shuffle')+tuple('vs_'+n for n in POLICIES[1:])
    criterion=all(policies['choices'][c]['ci95'][0]>0 for c in controls)
    model=joblib.load(path);transfer={}
    for action,name in enumerate(ACTIONS):
        x=model.coordinates.transform(data['states'][:,:,action])[0]
        logit=model.probe.decision_function(x.reshape(-1,x.shape[-1]));truth=1.-em[:,:,action].ravel()
        transfer[name]={'baseline_probe_applied_to_action_error_log_loss':float(log_loss(truth,expit(logit),labels=[0.,1.])),
            'baseline_probe_applied_to_action_error_auroc':float(roc_auc_score(truth,logit)) if len(np.unique(truth))==2 else None}
    gap=gap_scores(data['states']);badpairs=(em[:,0,0,None]==1)&(em[:,1:,0]==0)
    collapse=badpairs&(em[:,0,5,None]==0)&(em[:,1:,5]==0)&(gap[:,:,-1]>=gap.max(-1)-1e-10)
    result=dict(model=key,n_base_questions=len(cases),n_versions=3*len(cases),
        policies=policies,exploratory_all_controls_passed=criterion,controls=list(controls),
        prediction_audit=audit,hindsight_best_fixed_action=ACTIONS[best],
        fixed={c:{a:dict(f1=float(f1[:,ci,ai].mean()),em=float(em[:,ci,ai].mean()))
                  for ai,a in enumerate(ACTIONS)} for ci,c in enumerate(CONDITIONS)},
        oracle_f1=float(f1[:,:,:4].max(-1).mean()),
        learned_control_policy=stats(f1,em,tokens,prediction['control_choices']),
        baseline_probe_transfer=transfer,baseline_error_probe=metadata['probe'],
        condition_accuracy=float(np.mean(np.sum((prediction['coordinates'][...,None,:]-
            np.array([[0.,0.],[1.,0.],[0.,1.]]))**2,axis=-1).argmin(-1)==np.arange(3))),
        raw_gap_deletion_control=dict(empty_maximizer_rate=float(np.mean(gap[:,:,-1]>=gap.max(-1)-1e-10)),
            empty_f1=float(f1[:,1:,-1].mean()),original_bad_keep_f1=float(f1[:,1:,0].mean()),
            concrete_counterexamples=int(collapse.sum())),
        baseline_correctness_labels_used=True,mitigation_success_targets_used=False,
        note='Exploratory follow-up with fresh questions; not a new confirmatory ICML claim.')
    dump(OUT/key/'error_evaluation.json',result)
    np.savez_compressed(OUT/key/'error_scored.npz',f1=f1,em=em,tokens=tokens)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('qwen7b','phi35'),required=True)
    p.add_argument('--draws',type=int,default=4000);a=p.parse_args();evaluate(a.model,a.draws)
