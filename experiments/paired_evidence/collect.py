"""Resumable paired collection; prediction checkpoint precedes candidate actions."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from experiments.recovery_help.data import dump,file_digest,digest
from experiments.recovery_help.task import extract_answer
from experiments.paths import REPO,MODEL_PATHS
from .data import OUT,prepare
from .task import (CONDITIONS,ACTIONS,answer_prompt,query_prompt,selection_prompt,
                   expand,select_passages,focus)


def execute(cases,generate,decide=None):
    public=[case['versions'][condition] for case in cases for condition in CONDITIONS]
    prompts=[answer_prompt(e,e['current']) for e in public]
    baseline=generate(prompts,capture=True)
    if decide is not None:
        decide(baseline['states'].reshape(len(cases),3,*baseline['states'].shape[1:]),
               prompts,baseline['confidence'])
    flat=[{'condition':condition,'actions':[]} for _ in cases for condition in CONDITIONS]
    allstates=[]; selection_result=None; selected=None
    for action in ACTIONS:
        aux=None;valid=[True]*len(public)
        if action=='keep':result=baseline; action_prompts=prompts
        else:
            if action=='retrieve':
                aux=generate([query_prompt(e) for e in public],capture=False)
                paragraphs=[expand(e,q) for e,q in zip(public,aux['texts'])]
            elif action in ('filter','focus'):
                if selection_result is None:
                    selection_result=generate([selection_prompt(e) for e in public],capture=False)
                    selected=[select_passages(e,text) for e,text in zip(public,selection_result['texts'])]
                aux=selection_result; valid=[s[1] for s in selected]
                paragraphs=([s[0] for s in selected] if action=='filter' else
                    [focus(e,s[0]) for e,s in zip(public,selected)])
            elif action=='empty':paragraphs=[[] for _ in public]
            else:paragraphs=[e['current'] for e in public]
            action_prompts=[answer_prompt(e,p,sham=(action=='sham')) for e,p in zip(public,paragraphs)]
            result=generate(action_prompts,capture=True)
        allstates.append(result['states'])
        for i,row in enumerate(flat):
            record={k:result[k][i] for k in ('texts','prompt_ids','output_ids','input_tokens','output_tokens','confidence')}
            record['name']=action;record['answer']=extract_answer(record.pop('texts'))
            record['text']=result['texts'][i];record['prompt']=action_prompts[i]
            record['calls']=0 if action=='keep' else (2 if action in ('retrieve','filter','focus') else 1)
            record['selection_valid']=valid[i]
            if aux is not None:
                record['aux']={k:aux[k][i] for k in ('texts','prompt_ids','output_ids','input_tokens','output_tokens')}
            row['actions'].append(record)
    states=np.stack(allstates,axis=1).reshape(len(cases),3,len(ACTIONS),*baseline['states'].shape[1:])
    rows=[{'id':case['id'],'versions':flat[3*i:3*i+3]} for i,case in enumerate(cases)]
    return states,rows


class Engine:
    def __init__(self,key):
        from experiments.recovery_scale.backend import load
        self.hybrid,self.tok=load(key)
        eos=self.hybrid.baseline.generation_config.eos_token_id
        self.stops=set(([eos] if isinstance(eos,int) else list(eos or []))+[self.tok.pad_token_id])

    def __call__(self,prompts,cap=64,capture=True):
        from vllm import SamplingParams
        from experiments.recovery_help.backend import chat
        from experiments.activation_drift.pilot import capture as measure
        encoded=[self.tok.encode(chat(self.tok,p),add_special_tokens=False) for p in prompts]
        if max(map(len,encoded))+cap>4096:
            raise ValueError('Paired prompt exceeds the frozen model context limit')
        outputs=self.hybrid.recovery.generate([{'prompt_token_ids':ids} for ids in encoded],
            SamplingParams(temperature=0.,top_p=1.,top_k=-1,max_tokens=cap,logprobs=1),use_tqdm=False)
        tokens=[list(o.outputs[0].token_ids) for o in outputs]
        lengths=[next((i for i,t in enumerate(ids) if t in self.stops),len(ids)) for ids in tokens]
        confidence=[]
        for o,n in zip(outputs,lengths):
            lp=[step[token].logprob for step,token in zip(o.outputs[0].logprobs,o.outputs[0].token_ids)][:max(n,1)]
            confidence.append([float(np.mean(lp)),float(min(lp)),float(lp[0]),float(n)])
        states=None
        if capture:
            # Measure the last prompt token, before ANY answer token is supplied.
            parts=[]
            for start in range(0,len(encoded),8):
                ids=encoded[start:start+8]
                parts.append(measure(self.hybrid.baseline,self.tok,ids,[[] for _ in ids],self.stops)[:,:,0])
            states=np.concatenate(parts)
        return {'texts':[o.outputs[0].text for o in outputs],'prompt_ids':encoded,'output_ids':tokens,
                'input_tokens':[len(ids) for ids in encoded],'output_tokens':lengths,
                'confidence':confidence,'states':states}


def collect(key,splits,batch_size=8,predictor=None):
    from experiments.recovery_scale.verified_collect import verify_identity
    if os.environ.get('PYTHONHASHSEED')!='0':raise ValueError('Set PYTHONHASHSEED=0')
    manifest=prepare();directory=OUT/key;directory.mkdir(parents=True,exist_ok=True)
    identity=REPO/'results/recovery_scale'/f'{key}_model.json'
    dump(directory/'identity_before.json',verify_identity(MODEL_PATHS[key],identity))
    sources={p.name:file_digest(p) for p in Path(__file__).parent.glob('*.py') if p.name in ('task.py','data.py','collect.py')}
    protocol={'model':key,'model_identity':file_digest(identity),'sources':sources,'batch_size':batch_size,
              'conditions':CONDITIONS,'actions':ACTIONS,'state_site':'last prompt token, before output',
              'generator':'vllm greedy','measurement':'transformers exact prompt token IDs',
              'cap':64,'seed':20261013}
    if (directory/'protocol.json').exists() and json.loads((directory/'protocol.json').read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('Changed paired protocol; use a new run directory')
    dump(directory/'protocol.json',protocol)
    engine=Engine(key)
    for split in splits:
        raw=json.loads((OUT/'data'/f'{split}.json').read_text());root=directory/split;root.mkdir(exist_ok=True)
        fingerprint=digest({'protocol':protocol,'data':manifest['hashes'][split]})
        started=time.monotonic()
        for start in range(0,len(raw),batch_size):
            path=root/f'batch_{start:05d}.json';npz=path.with_suffix('.npz')
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['fingerprint']!=fingerprint or saved['states_sha256']!=file_digest(npz):raise ValueError('Changed paired batch')
                continue
            chunk=raw[start:start+batch_size]
            def decide(states,texts,confidence):
                if predictor is None:return
                prediction=predictor.predict(states,texts,np.array(confidence).reshape(len(chunk),3,4))
                predpath=root/f'predictions_{start:05d}.npz'
                if predpath.exists():raise ValueError('Orphan predictions: preserve before restarting in a new run')
                np.savez_compressed(predpath,**prediction)
                dump(predpath.with_suffix('.json'),{'ids':[r['id'] for r in chunk],
                     'time_unix':time.time(),'sha256':file_digest(predpath),'model':predictor.fingerprint})
            states,rows=execute(chunk,engine,decide if predictor is not None else None)
            temporary=npz.with_suffix('.npz.tmp')
            with temporary.open('wb') as f:np.savez_compressed(f,states=states)
            temporary.replace(npz)
            dump(path,{'fingerprint':fingerprint,'states_sha256':file_digest(npz),'rows':rows,'completed_time_unix':time.time()})
            print(json.dumps({'model':key,'split':split,'done':start+len(chunk),'total':len(raw),
                              'seconds':round(time.monotonic()-started,1)}),flush=True)
        dump(root/'complete.json',{'n':len(raw),'seconds_this_run':time.monotonic()-started})
    dump(directory/'identity_after.json',verify_identity(MODEL_PATHS[key],identity))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=MODEL_PATHS,required=True)
    p.add_argument('--splits',nargs='+',default=['calibration']);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--predictor');a=p.parse_args()
    if a.predictor:
        import joblib
        model=joblib.load(a.predictor)
    else:model=None
    collect(a.model,a.splits,a.batch_size,model)
