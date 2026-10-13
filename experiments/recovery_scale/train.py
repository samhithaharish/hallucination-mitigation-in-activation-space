"""Predeclared supervised comparisons: train, freeze validation selection, evaluate."""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from experiments.recovery_help.policy import allocate, select, metrics as base_metrics, paired_interval, dual_upper_bound
from .data import OUT, SEED, dump, digest, file_digest
from .collect import load_split
from .actions import ACTIONS
from .models import Encoder, GainModel, CorrectnessGate, semantic_embeddings, SEMANTIC_PATH

MAIN_FAMILIES=('confidence','lexical','semantic','activation','full','random','full_random','full_pca16')
BEHAVIORAL=('confidence','lexical','semantic')
GRID=[('ridge',a,0) for a in (1.,10.,100.,1000.)]+[('mlp',1.,s) for s in (0,1,2)]
GATE_CS=(.001,.01,.1,1.)


def metrics(*args):
    result=base_metrics(*args)
    result['resource_units_per_question']=result.pop('reserved_tokens_per_question')
    return result


def mask_menu(predictions,count):
    p=predictions.copy();p[:,count:]=-1e6;return p


def baselines(mean,rows):
    n=len(rows);mean=np.asarray(mean)
    priority=np.array([int(hashlib.sha256(r['id'].encode()).hexdigest()[:12],16)/16**12 for r in rows])
    # Break otherwise inseparable fixed-policy ties without labels or question semantics.
    jitter=1+1e-7*priority[:,None]
    result={'keep':np.zeros((n,10)),'constant':np.tile(mean,(n,1))*jitter}
    for j,a in enumerate(ACTIONS[1:],1):
        p=np.zeros((n,10));p[:,j]=jitter[:,0];result['fixed_'+a]=p
    rng=np.random.default_rng(SEED)
    result['random_policy']=np.column_stack([np.zeros(n),rng.uniform(0,1,(n,9))])
    return result


def source_hashes():
    paths=[Path(__file__).parent/f for f in ('models.py','train.py')]
    paths += [Path(__file__).parent.parent/'recovery_help'/'policy.py']
    return {str(p.relative_to(Path(__file__).parent.parent)):file_digest(p) for p in paths}


def semantic_for(key,split,data):
    root=OUT/key/'semantic';root.mkdir(parents=True,exist_ok=True)
    stem=root/f'{split}_{len(data["rows"])}';fp=digest({'texts':data['texts'],'model':SEMANTIC_PATH,
                                                   'weights':file_digest(Path(SEMANTIC_PATH)/'model.safetensors')})
    if stem.with_suffix('.json').exists():
        saved=json.loads(stem.with_suffix('.json').read_text())
        if saved['fingerprint']!=fp or saved['sha']!=file_digest(stem.with_suffix('.npy')):
            raise ValueError('Changed frozen semantic cache')
        return np.load(stem.with_suffix('.npy'))
    x=semantic_embeddings(data['texts']);np.save(stem.with_suffix('.npy'),x)
    dump(stem.with_suffix('.json'),{'fingerprint':fp,'sha':file_digest(stem.with_suffix('.npy'))})
    return x


def encode(encoder,data,semantic):
    return encoder.transform(data['states'],data['metadata'],data['texts'],semantic=semantic)


def fit(key,size=6000):
    root=OUT/key/f'prediction_{size}';root.mkdir(parents=True,exist_ok=True)
    if (root/'selection.json').exists():
        verify(root,key);print(f'{key} size{size} already frozen',flush=True);return
    train=load_split(key,'train',limit=size);val=load_split(key,'validation')
    encoder=Encoder().fit(train['states'],train['metadata'],train['texts'])
    joblib.dump(encoder,root/'encoder.joblib')
    ft=encode(encoder,train,semantic_for(key,'train',train))
    fv=encode(encoder,val,semantic_for(key,'validation',val))
    np.savez_compressed(root/'features_validation.npz',**fv)
    target=(train['f1']-train['f1'][:,:1])[:,1:]
    mean=(train['f1']-train['f1'][:,:1]).mean(0)
    predictions=baselines(mean,val['rows']);models={}
    for family in ft:
        grid=GRID if family in MAIN_FAMILIES else [('ridge',a,0) for a in (1.,10.,100.,1000.)]
        for kind,alpha,seed in grid:
            name=f'{family}_{kind}_{alpha:g}_s{seed}'
            model=GainModel(kind,alpha,seed).fit(ft[family],target)
            predictions[name]=model.predict(fv[family]);models[name]={'family':family,'kind':'response','model':model}
        if family in MAIN_FAMILIES:
            for c in GATE_CS:
                name=f'{family}_gate_{c:g}'
                model=CorrectnessGate(c).fit(ft[family],train['f1'],train['em'])
                predictions[name]=model.predict(fv[family]);models[name]={'family':family,'kind':'gate','model':model}
        print(json.dumps({'fit':key,'size':size,'family':family,'done_candidates':len(models)}),flush=True)
    candidates={};selections={}
    for menu in (5,10):
        scores={}
        for name,pred in predictions.items():
            choice,multiplier=allocate(mask_menu(pred,menu),val['calls'],len(val['rows']))
            scores[name]={**metrics(val['f1'],val['em'],val['calls'],choice),'multiplier':multiplier}
        behavior=[n for n in predictions if n not in models or models[n]['family'] in BEHAVIORAL]
        activation=[n for n,m in models.items() if m['family'] in MAIN_FAMILIES and
                    m['family'] not in BEHAVIORAL and m['kind']=='response']
        gates=[n for n,m in models.items() if m['kind']=='gate']
        selections[str(menu)]={label:max(names,key=lambda n:scores[n]['f1']) for label,names in (
            ('behavioral',behavior),('activation',activation),('gate',gates))}
        selections[str(menu)]['family_best']={f:max([n for n,m in models.items() if m['family']==f and m['kind']=='response'],
                                                  key=lambda n:scores[n]['f1']) for f in ft}
        candidates[str(menu)]=scores
    joblib.dump(models,root/'models.joblib');np.savez_compressed(root/'validation_predictions.npz',**predictions)
    dump(root/'validation_candidates.json',candidates)
    selection={'model':key,'train_size':size,'seed':SEED,'grid':GRID,'gate_cs':GATE_CS,'choices':selections,
        'mean_gains':mean.tolist(),'primary_budget':1.,'token_unit':float(val['tokens'][:,4].mean()),
        'source_hashes':source_hashes(),'protocol_hash':file_digest(OUT/key/'main'/'protocol.json'),
        'data_manifest_hash':file_digest(OUT/'data_manifest.json'),
        'artifacts':{p.name:file_digest(p) for p in root.iterdir() if p.is_file()}}
    dump(root/'selection.json',selection)
    print(json.dumps({'frozen':key,'size':size,'selection':selections}),flush=True)


def verify(root,key):
    s=json.loads((root/'selection.json').read_text())
    if s['source_hashes']!=source_hashes():raise ValueError('Frozen predictor source changed')
    if s['protocol_hash']!=file_digest(OUT/key/'main'/'protocol.json'):raise ValueError('Frozen protocol changed')
    if s['data_manifest_hash']!=file_digest(OUT/'data_manifest.json'):raise ValueError('Frozen data changed')
    for name,sha in s['artifacts'].items():
        if file_digest(root/name)!=sha:raise ValueError('Frozen fitted artifact changed: '+name)
    return s


def evaluate(key,size=6000):
    root=OUT/key/f'prediction_{size}';selection=verify(root,key)
    encoder=joblib.load(root/'encoder.joblib');models=joblib.load(root/'models.joblib')
    val_candidates=json.loads((root/'validation_candidates.json').read_text())
    report={'selection_hash':file_digest(root/'selection.json'),'splits':{}}
    for split in ('test','transfer'):
        data=load_split(key,split);features=encode(encoder,data,semantic_for(key,split,data))
        predictions=baselines(selection['mean_gains'],data['rows'])
        predictions.update({n:m['model'].predict(features[m['family']]) for n,m in models.items()})
        np.savez_compressed(root/f'features_{split}.npz',**features)
        np.savez_compressed(root/f'predictions_{split}.npz',**predictions)
        y,em=data['f1'],data['em'];idx=np.arange(len(y));choices={};result={'n':len(y),'budgets':{},'online':{}}
        for menu in (5,10):
            selected=selection['choices'][str(menu)]
            names=list(dict.fromkeys([selected['behavioral'],selected['activation'],selected['gate'],
                *selected['family_best'].values(),'keep','constant',*['fixed_'+a for a in ACTIONS[1:menu]]]))
            for cost_kind,fraction in [('calls',.5),('calls',1.),('calls',2.),('tokens',.5),('tokens',1.),('tokens',2.)]:
                budget=fraction*(selection['token_unit'] if cost_kind=='tokens' else 1)*len(y)
                prefix=f'{menu}_{cost_kind}_{fraction:g}';methods={}
                for name in names:
                    choice,multiplier=allocate(mask_menu(predictions[name],menu),data[cost_kind],budget)
                    if data[cost_kind][idx,choice].sum()>budget+1e-6:raise ValueError('Budget violation')
                    methods[name]={**metrics(y,em,data[cost_kind],choice),'multiplier':multiplier,
                        'mean_calls':float(data['calls'][idx,choice].mean()),'mean_reserved_tokens':float(data['tokens'][idx,choice].mean())}
                    choices[prefix+'_'+name]=choice
                gains=mask_menu(y-y[:,:1],menu)
                oc,ol=allocate(gains,data[cost_kind],budget)
                oracle={**metrics(y,em,data[cost_kind],oc),
                    'dual_f1_upper':float(y[:,0].mean()+dual_upper_bound(gains,data[cost_kind],budget,ol)/len(y))}
                a=choices[prefix+'_'+selected['activation']];b=choices[prefix+'_'+selected['behavioral']]
                g=choices[prefix+'_'+selected['gate']]
                result['budgets'][prefix]={'budget_per_question':budget/len(y),'methods':methods,'oracle':oracle,
                    'activation_minus_behavioral':paired_interval(y[idx,a]-y[idx,b]),
                    'activation_minus_gate':paired_interval(y[idx,a]-y[idx,g]),
                    'activation_minus_behavioral_em':paired_interval(em[idx,a]-em[idx,b])}
            for label in ('behavioral','activation','gate'):
                name=selected[label];mult=val_candidates[str(menu)][name]['multiplier']
                c=select(mask_menu(predictions[name],menu),data['calls'],mult)
                choices[f'{menu}_online_{label}']=c
                result['online'][f'{menu}_{label}']={**metrics(y,em,data['calls'],c),'mean_calls':float(data['calls'][idx,c].mean())}
        result['actions']={a:metrics(y,em,data['calls'],np.full(len(y),j)) for j,a in enumerate(ACTIONS)}
        result['oracle_f1']={str(m):float(y[:,:m].max(1).mean()) for m in (5,10)}
        for label in ('behavioral','activation','gate'):
            s5=selection['choices']['5'][label];s10=selection['choices']['10'][label]
            c5=choices['5_calls_1_'+s5];c10=choices['10_calls_1_'+s10]
            result['expanded_menu_minus_old_'+label]=paired_interval(y[idx,c10]-y[idx,c5])
        np.savez_compressed(root/f'choices_{split}.npz',**choices)
        report['splits'][split]=result
    dump(root/'evaluation.json',report)
    print(json.dumps({'evaluated':key,'size':size,'primary':{s:v['budgets']['10_calls_1']['activation_minus_behavioral']
                                                         for s,v in report['splits'].items()}}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('qwen7b','phi35'),required=True)
    p.add_argument('--phase',choices=('fit','evaluate'),required=True);p.add_argument('--sizes',nargs='+',type=int,default=[1200,3000,6000])
    args=p.parse_args()
    for size in args.sizes:(fit if args.phase=='fit' else evaluate)(args.model,size)
