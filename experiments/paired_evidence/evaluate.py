"""Answer scoring and paired uncertainty; never imported by predictor fitting."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import spearmanr
from experiments.recovery_help.data import dump,file_digest
from experiments.recovery_help.task import score_answer
from .data import OUT
from .task import CONDITIONS,ACTIONS
from .fit import load,SELECTABLE


def interval(values,draws=4000):
    values=np.asarray(values)
    if values.ndim>1:values=values.reshape(len(values),-1).mean(1)
    rng=np.random.default_rng(20261013);means=[]
    for start in range(0,draws,250):
        idx=rng.integers(len(values),size=(min(250,draws-start),len(values)))
        means.extend(values[idx].mean(1))
    return {'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}


def gap_scores(states):
    h=np.asarray(states,dtype=np.float32)
    scales=np.maximum(np.mean(np.sum(h[:,0,0]**2,axis=-1),axis=0),1e-8)
    difference=h[:,1:]-h[:,:1]
    distance=np.mean(np.sum(difference**2,axis=-1)/scales,axis=-1)
    return distance[:,:,:1]-distance


def quality(f1,choices):
    return np.take_along_axis(f1,np.asarray(choices)[...,None],axis=-1)[...,0]


def shuffle_expectation(f1,choices,risk,bins=10):
    shape=choices.shape;f=f1.reshape(-1,f1.shape[-1]);c=choices.ravel();r=risk.ravel()
    labels=np.empty(len(c),int);labels[np.argsort(r,kind='stable')]=np.arange(len(c))*bins//len(c)
    expected=np.empty(len(c))
    for label in np.unique(labels):
        mask=labels==label;freq=np.bincount(c[mask],minlength=f.shape[-1])/mask.sum()
        expected[mask]=f[mask]@freq
    return expected.reshape(shape)


def shuffle_interval(f1,choices,risk,draws=4000,bins=10):
    point=float(np.mean(quality(f1,choices)-shuffle_expectation(f1,choices,risk,bins)))
    rng=np.random.default_rng(20261013);values=[]
    for _ in range(draws):
        idx=rng.integers(len(f1),size=len(f1));ff,cc,rr=f1[idx],choices[idx],risk[idx]
        values.append(float(np.mean(quality(ff,cc)-shuffle_expectation(ff,cc,rr,bins))))
    return {'mean':point,'ci95':np.quantile(values,[.025,.975]).tolist()}


def matched_fixed(f1,choices,risk):
    n=choices.size;budget=2*int((choices>0).sum())
    priority=np.argsort(-risk.ravel(),kind='stable');costs=(0,2,2,2,1,1)
    candidates=[]
    for action in range(1,f1.shape[-1]):
        k=min(n,budget//costs[action])
        candidate=np.zeros(n,int);candidate[priority[:k]]=action
        value=quality(f1,candidate.reshape(choices.shape))
        candidates.append((float(value.mean()),action,value))
    return max(candidates,key=lambda x:x[0])


def matched_interval(f1,choices,risk,draws=4000):
    best,action,_=matched_fixed(f1,choices,risk);point=float(quality(f1,choices).mean()-best)
    rng=np.random.default_rng(20261013);values=[]
    for _ in range(draws):
        idx=rng.integers(len(f1),size=len(f1));ff,cc,rr=f1[idx],choices[idx],risk[idx]
        other,_,_=matched_fixed(ff,cc,rr)
        values.append(float(quality(ff,cc).mean()-other))
    return {'mean':point,'ci95':np.quantile(values,[.025,.975]).tolist(),'hindsight_action':ACTIONS[action]}


def arrays(key,split):
    collected=load(key,split);rows=[]
    for path in sorted((OUT/key/split).glob('batch_*.json')):
        rows.extend(json.loads(path.read_text())['rows'])
    cases=json.loads((OUT/'data'/f'{split}.json').read_text())
    if [r['id'] for r in rows]!=[r['id'] for r in cases]:raise ValueError('Misaligned scoring cases')
    f1=[];em=[];tokens=[]
    for row,case in zip(rows,cases):
        scores=[[score_answer(a['answer'],case['answer']) for a in v['actions']] for v in row['versions']]
        f1.append([[a['f1'] for a in s] for s in scores]);em.append([[a['em'] for a in s] for s in scores])
        tokens.append([[a['input_tokens']+a['output_tokens']+
            (a['aux']['input_tokens']+a['aux']['output_tokens'] if 'aux' in a else 0)
            for a in v['actions']] for v in row['versions']])
    return collected,rows,cases,np.array(f1),np.array(em),np.array(tokens)


def stats(f1,em,tokens,choices):
    current=quality(em,choices);baseline=em[:,:,0];active=choices>0
    calls=np.array([0,2,2,2,1,1])[choices]
    return {'f1':float(quality(f1,choices).mean()),'em':float(current.mean()),
            'extra_calls':float(calls.mean()),'extra_tokens':float((quality(tokens,choices)*active).mean()),
            'rescued':int(((baseline==0)&(current==1)).sum()),
            'harmed':int(((baseline==1)&(current==0)).sum()),
            'counts':{a:int((choices==i).sum()) for i,a in enumerate(ACTIONS)}}


def check_timing(meta,record):
    if meta['time_unix']>record['mitigation_start_time_unix']:
        raise ValueError('Decision was not frozen before mitigation started')


def frozen_predictions(key,split,collected):
    path=OUT/key/'fitted/best.joblib';model=joblib.load(path)
    metadata=json.loads(path.with_suffix('.json').read_text())
    if file_digest(path)!=metadata['sha256']:raise ValueError('Changed frozen predictor')
    result={};start=0;max_difference=0.
    for batch in sorted((OUT/key/split).glob('batch_*.json')):
        record=json.loads(batch.read_text());n=len(record['rows'])
        savedpath=batch.parent/f'predictions_{start:05d}.npz'
        meta=json.loads(savedpath.with_suffix('.json').read_text())
        if meta['sha256']!=file_digest(savedpath) or meta['model']!=model.fingerprint:raise ValueError('Changed saved decisions')
        check_timing(meta,record)
        if meta['ids']!=collected['ids'][start:start+n]:
            raise ValueError('Missing pre-action decision freeze')
        prediction=model.predict(collected['states'][start:start+n,:,0],
            collected['texts'][3*start:3*(start+n)],collected['confidence'][start:start+n],
            public=collected['public'][3*start:3*(start+n)])
        saved=np.load(savedpath)
        if set(saved.files)!=set(prediction):raise ValueError('Changed prediction fields')
        for name,value in prediction.items():
            max_difference=max(max_difference,float(np.max(np.abs(value-saved[name]))))
            np.testing.assert_allclose(value,saved[name],atol=1e-5,rtol=1e-6)
            if name.endswith('rule') or name.endswith('choices'):np.testing.assert_array_equal(value,saved[name])
            result.setdefault(name,[]).append(saved[name])
        start+=n
    return {k:np.concatenate(v) for k,v in result.items()},{'n_bases':start,'max_reload_difference':max_difference,'all_choices_precede_actions':True}


def acceptance(policies):
    controls=('vs_keep','vs_matched_risk_fixed','vs_risk_matched_shuffle',
              'vs_observable_rule','vs_confidence_rule')
    routing={name:all(policies[name][control]['ci95'][0]>0 for control in controls)
             for name in ('choices','class_rule')}
    return {'routing':routing,'strategy_fields_add_value':bool(routing['choices'] and
            policies['choices']['vs_class_rule']['ci95'][0]>0)}


def replicated_acceptance(results):
    if set(results)!={'qwen7b','phi35'}:
        raise ValueError('Both prespecified backbones are required')
    return {'routing':{name:all(r['acceptance']['routing'][name] for r in results.values())
                       for name in ('choices','class_rule')},
            'strategy_fields_add_value':all(r['acceptance']['strategy_fields_add_value']
                                           for r in results.values())}


def evaluate(key,split='pilot',draws=4000):
    data,rows,cases,f1,em,tokens=arrays(key,split)
    gap=gap_scores(data['states']);bad=f1[:,1:];bad_em=em[:,1:];bad_tokens=tokens[:,1:]
    all_gap_choices=gap.argmax(-1);gap_choices=gap[:,:,:SELECTABLE].argmax(-1)
    gains=f1[:,1:]-f1[:,1:,:1]
    rho=spearmanr(gap.ravel(),gains.ravel()).statistic
    clean_correct_bad_wrong=(em[:,0,0,None]==1)&(em[:,1:,0]==0)
    deletion_counterexample=(clean_correct_bad_wrong&(em[:,0,5,None]==0)&(em[:,1:,5]==0)
                            &(gap[:,:,-1]>=gap.max(-1)-1e-10))
    result={'model':key,'split':split,'n_base_questions':len(cases),'n_versions':3*len(cases),
        'fixed':{condition:{action: {'f1':float(f1[:,c,a].mean()),'em':float(em[:,c,a].mean())}
                  for a,action in enumerate(ACTIONS)} for c,condition in enumerate(CONDITIONS)},
        'oracle_same_action_gap':{'selectable_only':stats(bad,bad_em,bad_tokens,gap_choices),
            'including_empty_control':stats(bad,bad_em,bad_tokens,all_gap_choices),
            'empty_choice_rate':float((all_gap_choices==ACTIONS.index('empty')).mean()),
            'empty_maximizer_rate':float(np.mean(gap[:,:,-1]>=gap.max(-1)-1e-10)),
            'empty_f1':float(f1[:,1:,-1].mean()),'original_bad_keep_f1':float(f1[:,1:,0].mean()),
            'clean_correct_bad_wrong_pairs':int(clean_correct_bad_wrong.sum()),
            'empty_maximizer_both_answers_wrong_counterexamples':int(deletion_counterexample.sum()),
            'gap_vs_gain_spearman':None if not np.isfinite(rho) else float(rho)},
        'crossed_remedy_contrasts':{
            'retrieve_minus_keep_missing':interval(f1[:,1,1]-f1[:,1,0],draws),
            'focus_minus_keep_distractor':interval(f1[:,2,3]-f1[:,2,0],draws),
            'retrieve_minus_focus_missing':interval(f1[:,1,1]-f1[:,1,3],draws),
            'focus_minus_retrieve_distractor':interval(f1[:,2,3]-f1[:,2,1],draws)},
        'filter_valid_rate':float(np.mean([v['actions'][2]['selection_valid'] for r in rows for v in r['versions']])),
        'note':'Oracle gap uses clean counterparts and actual action states; it is not a deployable selector.'}
    if split=='test':
        predictions,audit=frozen_predictions(key,split,data)
        result['learned_control_policy']=stats(f1,em,tokens,predictions['control_choices'])
        selected_f1=f1[:,:,:SELECTABLE];selected_em=em[:,:,:SELECTABLE];selected_tokens=tokens[:,:,:SELECTABLE]
        best=int(f1.mean((0,1)).argmax())
        policies={}
        for name in ('choices','class_rule','observable_rule','confidence_rule'):
            choices=predictions[name];q=quality(selected_f1,choices)
            policies[name]={**stats(selected_f1,selected_em,selected_tokens,choices),
                'vs_keep':interval(q-selected_f1[:,:,0],draws),
                'vs_hindsight_best_fixed':interval(q-f1[:,:,best],draws),
                'vs_matched_risk_fixed':matched_interval(f1,choices,predictions['risk'],draws),
                'vs_risk_matched_shuffle':shuffle_interval(selected_f1,choices,predictions['risk'],draws)}
        for name in ('choices','class_rule'):
            primary=quality(selected_f1,predictions[name])
            for baseline in ('class_rule','observable_rule','confidence_rule'):
                if baseline!=name:
                    policies[name]['vs_'+baseline]=interval(primary-quality(selected_f1,predictions[baseline]),draws)
        result.update({'policies':policies,'prediction_audit':audit,'hindsight_fixed_action':ACTIONS[best],
                       'oracle_f1':float(selected_f1.max(-1).mean()),
                       'condition_accuracy':float(np.mean(np.sum((predictions['coordinates'][...,None,:]-
                           np.array([[0.,0.],[1.,0.],[0.,1.]]))**2,axis=-1).argmin(-1)==np.arange(3)))})
        result['acceptance']=acceptance(policies)
        result['primary_criterion_passed']=result['acceptance']['routing']['choices']
    dump(OUT/key/f'{split}_evaluation.json',result)
    np.savez_compressed(OUT/key/f'{split}_scored.npz',f1=f1,em=em,gap=gap,tokens=tokens)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('qwen7b','phi35'),required=True)
    p.add_argument('--split',default='pilot');p.add_argument('--draws',type=int,default=4000);a=p.parse_args()
    evaluate(a.model,a.split,a.draws)
