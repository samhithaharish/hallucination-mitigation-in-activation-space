"""Reconstruct scores, prompts, costs, full raw-state predictions, and grouped intervals."""
import argparse
import copy
import json
from pathlib import Path
import joblib
import numpy as np
from transformers import AutoTokenizer
from experiments.recovery_help.audit import official_scorer
from experiments.recovery_help.backend import chat
from experiments.recovery_help.policy import allocate,select,metrics,paired_interval
from .data import OUT, OLD_OUT, COUNTS, STRATA, MODEL_PATHS, SEED, prepare, dump, digest, file_digest, normalize
from .actions import ACTIONS,CALLS,public_example,prompt,bounded_prompt,costs,cap_for,metadata,extract_answer,majority
from .collect import load_split
from .train import verify,encode,semantic_for,baselines,mask_menu


def article_groups(rows):
    parent=list(range(len(rows)));owners={}
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,r in enumerate(rows):
        for title in r['supporting_facts']['title']:
            if title in owners:parent[root(i)]=root(owners[title])
            else:owners[title]=i
    labels={};groups=[]
    for i in range(len(rows)):
        j=root(i)
        if j not in labels:labels[j]=len(labels)
        groups.append(labels[j])
    return np.array(groups)


def cluster_interval(values,groups,repeats=4000):
    values=np.asarray(values);_,labels=np.unique(groups,return_inverse=True)
    count=np.bincount(labels);total=np.bincount(labels,weights=values);n=len(count)
    rng=np.random.default_rng(SEED);means=[]
    for _ in range(repeats):
        drawn=rng.integers(n,size=n);means.append(total[drawn].sum()/count[drawn].sum())
    return {'mean':float(values.mean()),'low':float(np.quantile(means,.025)),
        'high':float(np.quantile(means,.975)),'groups':n,'largest_group':int(count.max()),
        'unit':'connected supporting-article component','replicates':repeats}


def check_score(action,record,gold,scorer=None):
    f1,em=official_scorer() if scorer is None else scorer
    answer=majority([s['text'] for s in record['stages']]) if action=='vote' else extract_answer(
        record['text'],action in ('reason','both','query','verify'))
    assert answer==record['answer']
    assert abs(f1(answer,gold)[0]-record['score']['f1'])<1e-12
    assert float(em(answer,gold))==record['score']['em']


def audit_splits():
    manifest=prepare();splits={s:json.loads((OUT/'data'/f'{s}.json').read_text()) for s in COUNTS}
    prior=[r for p in (OLD_OUT/'data').glob('*.json') for r in json.loads(p.read_text())]
    ids={r['id'] for r in prior};questions={normalize(r['question']) for r in prior}
    for s,rows in splits.items():
        assert len(rows)==COUNTS[s]
        for r in rows:
            assert r['id'] not in ids and normalize(r['question']) not in questions
            ids.add(r['id']);questions.add(normalize(r['question']))
        if s in ('train','validation','test'):
            for (level,kind),fraction in STRATA.items():
                assert sum(r['level']==level and r['type']==kind for r in rows)==round(len(rows)*fraction)
    titles={t for s in ('validation','test','transfer') for r in splits[s] for t in r['supporting_facts']['title']}
    assert not titles.intersection(t for s in ('train','calibration') for r in splits[s] for t in r['supporting_facts']['title'])
    return splits


def run(key,sizes=(1200,3000,6000)):
    splits=audit_splits();scorer=official_scorer();result={'counts':{},'restoration':{},'cluster_intervals':{}}
    tok=AutoTokenizer.from_pretrained(MODEL_PATHS[key],local_files_only=True)
    protocol=json.loads((OUT/key/'main'/'protocol.json').read_text())
    for path,sha in protocol['source_hashes'].items():assert file_digest(Path(__file__).parent.parent/path)==sha,path
    for split in ('train','validation','test','transfer'):
        d=load_split(key,split);k=1 if split=='transfer' else 2
        assert d['states'].shape[:3]==(len(splits[split]),6,2)
        assert d['metadata'].shape==(len(splits[split]),27)
        assert np.isfinite(d['states']).all() and np.isfinite(d['metadata']).all()
        for raw,row in zip(splits[split],d['rows']):
            assert raw['id']==row['id'] and raw['answer']==row['gold']
            changed=copy.deepcopy(raw);changed['answer']='HIDDEN_LABEL';changed['supporting_facts']={'title':['HIDDEN_TITLE'],'sent_id':[999]}
            e=public_example(raw);assert public_example(changed)==e
            keep=row['actions']['keep'];answer=extract_answer(keep['text']);anchor=answer or '(no valid initial answer)'
            bp=chat(tok,prompt(e,initial_k=k));assert row['baseline_prompt_hash']==digest(bp)
            assert row['baseline_text_features']==bp+'\nInitial answer: '+answer
            assert keep['input_tokens']==len(tok.encode(bp,add_special_tokens=False))
            expected=metadata(e)+[len(e['question'].split()),len(answer.split()),keep['input_tokens'],
                                  float('<final>' in keep['text']),float(bool(answer)),k]
            np.testing.assert_allclose(row['metadata'],expected,rtol=0,atol=1e-10)
            reserved=costs(tok,e,anchor,k)
            for j,a in enumerate(ACTIONS):
                r=row['actions'][a];check_score(a,r,raw['answer'],scorer)
                assert r['calls']==CALLS[j] and r['reserved_tokens']==reserved[j]
                if a=='keep':continue
                assert len(r['stages'])==CALLS[j]
                for step,stage in enumerate(r['stages']):
                    p=bounded_prompt(tok,e,a,anchor,k,r['stages'][0]['text'],2) if a in ('query','verify') and step==1 else chat(tok,prompt(e,a,anchor,k))
                    assert digest(p)==stage['prompt_hash']
                    assert len(tok.encode(p,add_special_tokens=False))==stage['input_tokens']
                    assert stage['cap']==cap_for(a,step+1)
                    assert 0<=stage['output_tokens']<=stage['cap']
                    assert stage['at_cap']==(stage['output_tokens']==stage['cap'])
                for field in ('input_tokens','output_tokens'):assert r[field]==sum(s[field] for s in r['stages'])
                assert r['input_tokens']+r['output_tokens']<=r['reserved_tokens']
        result['counts'][split]=len(d['rows'])
        print(json.dumps({'audited_raw':key,'split':split,'n':len(d['rows'])}),flush=True)
    for size in sizes:
        root=OUT/key/f'prediction_{size}';selection=verify(root,key)
        evaluation=json.loads((root/'evaluation.json').read_text());assert evaluation['selection_hash']==file_digest(root/'selection.json')
        encoder=joblib.load(root/'encoder.joblib');models=joblib.load(root/'models.joblib')
        result['restoration'][str(size)]={}
        for split in ('validation','test','transfer'):
            d=load_split(key,split);features=encode(encoder,d,semantic_for(key,split,d))
            restored=baselines(selection['mean_gains'],d['rows'])
            restored.update({n:m['model'].predict(features[m['family']]) for n,m in models.items()})
            fname='validation_predictions.npz' if split=='validation' else f'predictions_{split}.npz'
            saved=dict(np.load(root/fname));maxerror=max(float(np.max(np.abs(restored[n]-saved[n]))) for n in saved)
            assert maxerror<1e-5,('Raw prediction restore',maxerror)
            result['restoration'][str(size)][split]={'raw_prediction_max_abs_difference':maxerror}
            if split=='validation':continue
            choices=dict(np.load(root/f'choices_{split}.npz'));y=d['f1'];idx=np.arange(len(y));er=evaluation['splits'][split]
            for prefix,record in er['budgets'].items():
                menu,cost_kind,fraction=prefix.split('_');menu=int(menu)
                budget=record['budget_per_question']*len(y)
                for name,numbers in record['methods'].items():
                    c,m=allocate(mask_menu(restored[name],menu),d[cost_kind],budget)
                    assert np.array_equal(c,choices[prefix+'_'+name]),('Choice mismatch',key,size,split,prefix,name)
                    assert d[cost_kind][idx,c].sum()<=budget+1e-6
                    actual=metrics(y,d['em'],d[cost_kind],c)
                    for field in ('f1','em','f1_gain','rescue_rate','harm_rate','action_counts'):
                        assert actual[field]==numbers[field]
                names=selection['choices'][str(menu)];a=choices[prefix+'_'+names['activation']];b=choices[prefix+'_'+names['behavioral']]
                assert paired_interval(y[idx,a]-y[idx,b])==record['activation_minus_behavioral']
            for menu in (5,10):
                for label in ('behavioral','activation','gate'):
                    name=selection['choices'][str(menu)][label]
                    multiplier=json.loads((root/'validation_candidates.json').read_text())[str(menu)][name]['multiplier']
                    c=select(mask_menu(restored[name],menu),d['calls'],multiplier)
                    assert np.array_equal(c,choices[f'{menu}_online_{label}'])
            if size==6000:
                s=selection['choices']['10'];a=choices['10_calls_1_'+s['activation']];b=choices['10_calls_1_'+s['behavioral']]
                g=choices['10_calls_1_'+s['gate']];groups=article_groups(splits[split])
                result['cluster_intervals'][split]={
                    'activation_minus_behavioral':cluster_interval(y[idx,a]-y[idx,b],groups),
                    'activation_minus_gate':cluster_interval(y[idx,a]-y[idx,g],groups)}
        print(json.dumps({'audited_fits':key,'size':size}),flush=True)
    result['checks']=['fresh split exclusion and title embargo','exact stratum matching','source and array hashes',
        'gold-invariant public inputs','official scorer','prompt/metadata/cost reconstruction','all stages counted',
        'raw-state encoder/model reload','all held-out budget choices and metrics','bootstrap reconstruction','online decisions']
    dump(OUT/key/'audit.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('qwen7b','phi35'),required=True)
    args=p.parse_args();run(args.model)
