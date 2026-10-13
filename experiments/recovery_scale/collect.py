"""Independent recovery branches, resumable immutable collection, aligned raw loading."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
import transformers
import importlib.metadata
from experiments.recovery_help.backend import chat, CONFIDENCE_NAMES
from .backend import generate as default_generate, load
from .data import OUT, SEED, MODEL_PATHS, dump, digest, file_digest, prepare
from .actions import (ACTIONS, CALLS, public_example, metadata, prompt, bounded_prompt,
                      costs, cap_for, majority, extract_answer, score_answer, DIRECT_CAP)


def validate_batch(payload, fingerprint, data_hash, ids):
    if (payload['fingerprint'] != fingerprint or payload['data_hash'] != data_hash or
            [r['id'] for r in payload['rows']] != ids):
        raise ValueError('Changed or misaligned cached batch')


def collect_batch(model, tok, raw, initial_k, seed, generate=default_generate):
    public = [public_example(r) for r in raw]
    baseline_prompts = [chat(tok,prompt(e,initial_k=initial_k)) for e in public]
    baseline = generate(model,tok,baseline_prompts,DIRECT_CAP,capture=True)
    answers = [extract_answer(t) for t in baseline['texts']]
    anchors = [a or '(no valid initial answer)' for a in answers]
    reserved = [costs(tok,e,a,initial_k) for e,a in zip(public,anchors)]
    rows = []
    for i,(r,e) in enumerate(zip(raw,public)):
        meta = metadata(e)+[len(e['question'].split()),len(answers[i].split()),baseline['input_lengths'][i],
                           float('<final>' in baseline['texts'][i]),float(bool(answers[i])),initial_k]
        rows.append({'id':r['id'],'gold':r['answer'],'level':r['level'] if 'level' in r else 'unknown',
            'type':r.get('type','unknown'), 'baseline_text_features':baseline_prompts[i]+'\nInitial answer: '+answers[i],
            'baseline_prompt_hash':digest(baseline_prompts[i]),'metadata':meta,'actions':{'keep':{
                'text':baseline['texts'][i],'answer':answers[i],'score':score_answer(answers[i],r['answer']),
                'calls':0,'reserved_tokens':0,'input_tokens':baseline['input_lengths'][i],
                'output_tokens':baseline['lengths'][i],'at_cap':baseline['at_cap'][i],'stages':[]}}})
    for j,action in enumerate(ACTIONS[1:],1):
        prompts = [chat(tok,prompt(e,action,a,initial_k)) for e,a in zip(public,anchors)]
        results, all_prompts, seeds = [], [], []
        for step in range(CALLS[j]):
            if action in ('query','verify') and step == 1:
                prompts = [bounded_prompt(tok,e,action,a,initial_k,aux,stage=2)
                           for e,a,aux in zip(public,anchors,results[0]['texts'])]
            sample_seed = seed+100*j+step if action=='vote' else None
            result = generate(model,tok,prompts,cap_for(action,step+1),seed=sample_seed)
            results.append(result);all_prompts.append(prompts);seeds.append(sample_seed)
        for i,r in enumerate(raw):
            reasoning = action in ('reason','both','query','verify','vote')
            answer = majority([b['texts'][i] for b in results]) if action=='vote' else extract_answer(results[-1]['texts'][i],reasoning)
            stages = [{'text':b['texts'][i],'prompt_hash':digest(ps[i]),'input_tokens':b['input_lengths'][i],
                       'output_tokens':b['lengths'][i],'cap':cap_for(action,s+1),'at_cap':b['at_cap'][i],
                       'seed':seeds[s]} for s,(b,ps) in enumerate(zip(results,all_prompts))]
            actual = sum(s['input_tokens']+s['output_tokens'] for s in stages)
            if actual>reserved[i][j]: raise RuntimeError('Pre-action token reservation was exceeded')
            rows[i]['actions'][action] = {'text':results[-1]['texts'][i],'answer':answer,
                'score':score_answer(answer,r['answer']),'calls':CALLS[j],'reserved_tokens':reserved[i][j],
                'input_tokens':sum(s['input_tokens'] for s in stages),'output_tokens':sum(s['output_tokens'] for s in stages),
                'at_cap':any(s['at_cap'] for s in stages),'stages':stages}
    return rows, {'states':baseline['states'],'confidence':baseline['confidence'], 'layers':baseline['layers']}


def protocol(key,batch_size,repeat):
    here = Path(__file__).parent
    sources = [here/f for f in ('actions.py','data.py','backend.py','collect.py')]
    sources += [here.parent/'recovery_help'/f for f in ('backend.py','task.py','data.py')]
    return {'seed':SEED,'model':key,'model_path':MODEL_PATHS[key],'batch_size':batch_size,'repeat':repeat,
        'baseline_engine':'transformers','recovery_engine':'vllm', 'vllm':importlib.metadata.version('vllm'),
        'flashinfer_sampler':os.environ.get('VLLM_USE_FLASHINFER_SAMPLER'),
        'deep_gemm':os.environ.get('VLLM_USE_DEEP_GEMM'),
        'recovery_kv_bytes':16<<30,'recovery_max_model_len':4096,'recovery_eager':True,
        'actions':list(ACTIONS),'calls':list(CALLS),'torch':torch.__version__,'transformers':transformers.__version__,
        'pythonhashseed':os.environ.get('PYTHONHASHSEED'),'direct_cap':64,'reason_cap':256,
        'query_cap':64,'critique_cap':192,'dynamic_input_cap':2048,'initial_k':2,'shift_initial_k':1,
        'vote_temperature':.7,'vote_top_p':.9,
        'source_hashes':{str(p.relative_to(here.parent)):file_digest(p) for p in sources},
        'model_config_hash':file_digest(Path(MODEL_PATHS[key])/'config.json')}


def run(key,splits,run_name='main',batch_size=64,repeat=0,limit=None):
    if os.environ.get('PYTHONHASHSEED') != '0': raise ValueError('Run with PYTHONHASHSEED=0 for stable BM25 accumulation')
    manifest=prepare(); base=OUT/key/run_name; p=protocol(key,batch_size,repeat)
    p['limit']=limit
    if (base/'protocol.json').exists() and json.loads((base/'protocol.json').read_text())!=p:
        raise ValueError('Changed frozen collection protocol; use a new versioned run')
    dump(base/'protocol.json',p); fingerprint=digest(p)
    model,tok=load(key)
    for split in splits:
        raw=json.loads((OUT/'data'/f'{split}.json').read_text())
        if limit is not None: raw=raw[:limit]
        directory=base/split;directory.mkdir(parents=True,exist_ok=True);start_time=time.monotonic()
        k=1 if split=='transfer' else 2
        for start in range(0,len(raw),batch_size):
            chunk=raw[start:start+batch_size];stem=directory/f'batch_{start:05d}'
            jp,npz=stem.with_suffix('.json'),stem.with_suffix('.npz')
            if jp.exists() and npz.exists():
                cached=json.loads(jp.read_text())
                validate_batch(cached,fingerprint,manifest['split_hashes'][split],[r['id'] for r in chunk])
                if file_digest(npz)!=cached['array_hash']: raise ValueError('Changed cached activation arrays')
                continue
            rows,arrays=collect_batch(model,tok,chunk,k,SEED+start+repeat*100000)
            layers=arrays.pop('layers')
            temp=npz.with_suffix('.npz.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,**arrays)
            temp.replace(npz)
            dump(jp,{'fingerprint':fingerprint,'data_hash':manifest['split_hashes'][split],
                'array_hash':file_digest(npz),'layers':layers,'rows':rows})
            print(json.dumps({'model':key,'run':run_name,'split':split,'done':start+len(chunk),
                              'total':len(raw),'seconds':round(time.monotonic()-start_time,1)}),flush=True)
        allrows=[r for jp in sorted(directory.glob('batch_*.json')) for r in json.loads(jp.read_text())['rows']]
        if [r['id'] for r in allrows]!=[r['id'] for r in raw]: raise ValueError('Collection differs from frozen cohort')
        summary={'n':len(raw),'seconds_this_run':time.monotonic()-start_time,'actions':{}}
        for action in ACTIONS:
            aa=[r['actions'][action] for r in allrows]
            summary['actions'][action]={m:float(np.mean([a['score'][m] for a in aa])) for m in ('f1','em')}
            summary['actions'][action].update({
                'empty':float(np.mean([not a['answer'] for a in aa])),
                **{m:float(np.mean([a[m] for a in aa])) for m in ('at_cap','input_tokens','output_tokens','reserved_tokens')}})
        dump(directory/'summary.json',summary)
        if split=='calibration':print(json.dumps(summary),flush=True)


def load_split(key,split,run_name='main',limit=None):
    directory=OUT/key/run_name/split
    if not (directory/'summary.json').exists():raise ValueError('Incomplete collection: '+str(directory))
    manifest=prepare(); p=json.loads((directory.parent/'protocol.json').read_text())
    rows,states,confidence=[],[],[]
    expected=manifest['ids'][split]
    if p['limit'] is not None:expected=expected[:p['limit']]
    for jp in sorted(directory.glob('batch_*.json')):
        b=json.loads(jp.read_text());start=len(rows)
        validate_batch(b,digest(p),manifest['split_hashes'][split],expected[start:start+len(b['rows'])])
        if file_digest(jp.with_suffix('.npz'))!=b['array_hash']:raise ValueError('Changed cached activation arrays')
        rows.extend(b['rows'])
        with np.load(jp.with_suffix('.npz')) as z:states.append(z['states']);confidence.append(z['confidence'])
    if [r['id'] for r in rows]!=expected:raise ValueError('Misaligned split')
    n=limit or len(rows);rows=rows[:n]
    return {'rows':rows,'states':np.concatenate(states)[:n],
        'metadata':np.concatenate([np.concatenate(confidence)[:n],np.array([r['metadata'] for r in rows])],axis=1),
        'texts':[r['baseline_text_features'] for r in rows],
        'f1':np.array([[r['actions'][a]['score']['f1'] for a in ACTIONS] for r in rows]),
        'em':np.array([[r['actions'][a]['score']['em'] for a in ACTIONS] for r in rows]),
        'calls':np.tile(CALLS,(len(rows),1)).astype(float),
        'tokens':np.array([[r['actions'][a]['reserved_tokens'] for a in ACTIONS] for r in rows],dtype=float)}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=MODEL_PATHS,required=True)
    parser.add_argument('--splits',nargs='+',default=['calibration']);parser.add_argument('--run-name',default='calibration')
    parser.add_argument('--batch-size',type=int,default=64);parser.add_argument('--repeat',type=int,default=0)
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    run(args.model,args.splits,args.run_name,args.batch_size,args.repeat,args.limit)
