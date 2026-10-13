"""Activation-based recovery pilot. See README.md for the recorded design."""
import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

from experiments.paths import DRIFT_RUN as ROOT
CODE = Path(__file__).resolve().parent
SEED = 20261010
ACTIONS = ('keep', 'retry', 'retrieve', 'reason', 'both', 'fresh', 'prune')


def helpers():
    from experiments.recovery_scale import data, actions
    return data, actions


def save(path, obj):
    from experiments.recovery_help.data import dump
    dump(path, obj)


def sha(path):
    from experiments.recovery_help.data import file_digest
    return file_digest(path)


def prepare():
    from experiments.recovery_help import data as old
    from experiments.recovery_scale.data import sample, STRATA
    import pyarrow.parquet as pq
    path = ROOT/'data_manifest.json'
    if path.exists():
        manifest = json.loads(path.read_text())
        for name, value in manifest['hashes'].items():
            if sha(ROOT/'data'/f'{name}.json') != value:
                raise ValueError('Changed data: '+name)
        return manifest
    data, _ = helpers()
    if any(not (root/'data_manifest.json').exists() for root in (old.OUT, data.OUT)):
        raise FileNotFoundError('Run python -m experiments.prepare before collecting data')
    previous = [row for root in (old.OUT, data.OUT) for p in (root/'data').glob('*.json')
                for row in json.loads(p.read_text())]
    ids = {r['id'] for r in previous}
    questions = {old.normalize(r['question']) for r in previous}
    files = sorted(old.CORPUS.glob('train-*.parquet'))
    rows = [r for p in files for r in pq.read_table(p).to_pylist()]
    splits = {}; embargo = set()
    for i,(name,n) in enumerate((('test',1000), ('validation',200), ('train',800), ('smoke',8))):
        selected = sample(rows, n, ids, questions, embargo, SEED+i,
                          STRATA if name != 'smoke' else None)
        splits[name] = selected
        ids.update(r['id'] for r in selected)
        questions.update(old.normalize(r['question']) for r in selected)
        if name in ('test','validation'):
            embargo.update(t for r in selected for t in r['supporting_facts']['title'])
        save(ROOT/'data'/f'{name}.json', selected)
    manifest = {'seed':SEED, 'excluded_previous':len(previous),
                'ids':{s:[r['id'] for r in rr] for s,rr in splits.items()},
                'hashes':{s:sha(ROOT/'data'/f'{s}.json') for s in splits},
                'source_hashes':{p.name:sha(p) for p in files}}
    save(path,manifest)
    return manifest


def replay_inputs(prompts, outputs, stop_ids):
    rows, points = [], []
    for p, raw in zip(prompts, outputs):
        if not p:
            raise ValueError('Empty prompt')
        n = next((i for i,t in enumerate(raw) if t in stop_ids), len(raw))
        row = list(p) + list(raw[:max(n-1,0)])
        rows.append(row); points.append([len(p)-1, len(row)-1])
    if len(rows) != len(prompts) or len(prompts) != len(outputs):
        raise ValueError('Misaligned prompt/output lengths')
    return rows, points


def capture(model, tok, prompts, outputs, stop_ids, layers=None):
    import torch
    from experiments.recovery_help.backend import layer_grid
    layers = layer_grid(model) if layers is None else layers
    rows, points = replay_inputs(prompts,outputs,stop_ids)
    device = next(model.parameters()).device
    ids = torch.full((len(rows),max(map(len,rows))),tok.pad_token_id,device=device,dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i,row in enumerate(rows):
        ids[i,:len(row)] = torch.tensor(row,device=device)
        mask[i,:len(row)] = 1
    idx = torch.tensor(points,device=device)
    captured = {}
    with torch.inference_mode(), contextlib.ExitStack() as stack:
        for layer in layers:
            def hook(module, inputs, output, layer=layer):
                h = output[0] if isinstance(output,tuple) else output
                captured[layer] = h[torch.arange(len(rows),device=device)[:,None],idx].float().cpu().numpy()
            handle = model.model.layers[layer-1].register_forward_hook(hook)
            stack.callback(handle.remove)
        model.model(input_ids=ids,attention_mask=mask,use_cache=False)
    return np.stack([captured[l] for l in layers],axis=1).astype(np.float16)


def drift_scores(residual, moves):
    return (np.sum(residual**2,axis=-1)[:,None] -
            np.sum((residual[:,None,:]+moves)**2,axis=-1))/residual.shape[-1]


class Geometry:
    """The only supervised input is correctness of the ordinary baseline."""
    def __init__(self, components=16):
        self.components = components

    def encode(self, states):
        return [np.concatenate([p.transform(states[:,i,t].astype(np.float32))/scale
                for i,(p,scale) in enumerate(self.encoders[t])],axis=1)
                for t in (0,1)]

    @staticmethod
    def regress(x,y,vx,vy,alphas):
        models = [Ridge(alpha=a).fit(x,y) for a in alphas]
        errors = [float(np.mean((m.predict(vx)-vy)**2)) for m in models]
        k = int(np.argmin(errors))
        return models[k], {'alpha':alphas[k], 'validation_mse':errors[k]}

    def fit(self, train_states, baseline_correct, val_states, val_baseline_correct):
        train_states, val_states = np.asarray(train_states), np.asarray(val_states)
        self.actions = train_states.shape[1]
        self.encoders = [[],[]]
        for t in (0,1):
            for l in range(train_states.shape[2]):
                x = train_states[:,0,l,t].astype(np.float32)
                p = PCA(min(self.components,*x.shape),svd_solver='randomized',random_state=SEED).fit(x)
                scale = np.maximum(np.sqrt(p.explained_variance_),.1*np.sqrt(p.explained_variance_[0]))
                scale = np.maximum(scale,1e-8)
                self.encoders[t].append((p,scale))
        p,e = self.encode(train_states[:,0]); vp,ve = self.encode(val_states[:,0])
        x,vx = np.concatenate([p,e],axis=1),np.concatenate([vp,ve],axis=1)
        ok,vok = np.asarray(baseline_correct,bool),np.asarray(val_baseline_correct,bool)
        if min(ok.sum(),(~ok).sum(),vok.sum(),(~vok).sum()) < 2:
            raise ValueError('Insufficient examples for correct/incorrect reference controls')
        self.references = {}; self.details = {'references':{},'operators':[]}
        rng = np.random.default_rng(SEED)
        masks = {'correct':(ok,vok),'incorrect':(~ok,~vok),
                 'permuted':(rng.permutation(ok),rng.permutation(vok))}
        for name,(m,vm) in masks.items():
            self.references[name],info = self.regress(p[m],e[m],vp[vm],ve[vm],[10,100,1000])
            self.details['references'][name] = {**info,'n_fit':int(m.sum()),'n_val':int(vm.sum())}
        self.means = [np.zeros(e.shape[1],dtype=np.float32)]; self.operators = [None]
        self.direction_diagnostics = []
        for a in range(1,self.actions):
            delta = self.encode(train_states[:,a])[1]-e
            vd = self.encode(val_states[:,a])[1]-ve
            self.means.append(delta.mean(0))
            op,info = self.regress(x,delta,vx,vd,[10,100,1000,10000])
            self.operators.append(op)
            self.details['operators'].append({**info,'action':ACTIONS[a] if a<len(ACTIONS) else str(a),
                'mean_validation_mse':float(np.mean((vd-self.means[-1])**2))})
            left,right = delta[:len(delta)//2].mean(0),delta[len(delta)//2:].mean(0)
            norm = np.linalg.norm(delta,axis=1)
            mean = self.means[-1]
            self.direction_diagnostics.append({'split_half_cosine':float(left@right/(np.linalg.norm(left)*np.linalg.norm(right)+1e-12)),
                'individual_alignment':float(np.mean(delta@mean/(norm*np.linalg.norm(mean)+1e-12))),
                'mean_energy_fraction':float(mean@mean/(np.mean(norm**2)+1e-12))})
        self.means = np.stack(self.means)
        return self

    def predict(self, baseline_states):
        p,e = self.encode(baseline_states)
        x = np.concatenate([p,e],axis=1)
        moves = np.stack([np.zeros_like(e)]+[op.predict(x) for op in self.operators[1:]],axis=1)
        mean = np.broadcast_to(self.means,moves.shape)
        r = e-self.references['correct'].predict(p)
        result = {'operator':drift_scores(r,moves), 'mean':drift_scores(r,mean),
                  'negative':drift_scores(r,-moves), 'risk':np.mean(r**2,axis=1)}
        for name in ('incorrect','permuted'):
            result[name+'_reference'] = drift_scores(e-self.references[name].predict(p),moves)
        rng = np.random.default_rng(SEED)
        permutation = np.r_[0,1+np.roll(np.arange(self.actions-1),1)]
        result['shuffled_actions'] = drift_scores(r,moves[:,permutation])
        # A common orthogonal signed-coordinate permutation preserves all move
        # norms and between-action geometry but destroys alignment with drift.
        order = rng.permutation(e.shape[1]); signs = rng.choice([-1,1],size=e.shape[1])
        result['random_orientation'] = drift_scores(r,moves[:,:,order]*signs)
        return result


def load_arrays(key,split):
    states, rows = [], []
    manifest = prepare()
    directory = ROOT/key/split
    for path in sorted(directory.glob('batch_*.json')):
        payload = json.loads(path.read_text())
        npz = path.with_suffix('.npz')
        if sha(npz) != payload['states_sha256']:
            raise ValueError('Changed state cache')
        states.append(np.load(npz)['states']); rows.extend(payload['rows'])
    if [r['id'] for r in rows] != manifest['ids'][split]:
        raise ValueError('Incomplete/misaligned split: '+key+'/'+split)
    return np.concatenate(states),rows


def fit(key):
    from experiments.recovery_help.task import score_answer
    states,rows = load_arrays(key,'train'); vs,vr = load_arrays(key,'validation')
    # Only baseline labels are constructed. Action answers are never scored here.
    def labels(split,rr):
        gold = {r['id']:r['answer'] for r in json.loads((ROOT/'data'/f'{split}.json').read_text())}
        return np.array([score_answer(r['actions'][0]['answer'],gold[r['id']])['em'] for r in rr],bool)
    model = Geometry().fit(states,labels('train',rows),vs,labels('validation',vr))
    path = ROOT/key/'geometry.joblib'; joblib.dump(model,path)
    save(ROOT/key/'geometry_fit.json',{'selection':model.details,'direction_diagnostics':model.direction_diagnostics,
        'model_sha256':sha(path),'source_sha256':sha(Path(__file__)),
        'protocol_sha256':sha(CODE/'README.md'),'action_success_labels_used':False})
    print(json.dumps({'model':key,'geometry_frozen':sha(path),'fit':model.details}),flush=True)
    return model


def generate_capture(hybrid,tok,prompts,cap):
    from vllm import SamplingParams
    ids = [tok.encode(p,add_special_tokens=False) for p in prompts]
    outputs = hybrid.recovery.generate([{'prompt_token_ids':p} for p in ids],
        SamplingParams(temperature=0.,top_p=1.,top_k=-1,max_tokens=cap),use_tqdm=False)
    tokens = [list(r.outputs[0].token_ids) for r in outputs]
    texts = [r.outputs[0].text for r in outputs]
    eos = hybrid.baseline.generation_config.eos_token_id
    stops = set(([eos] if isinstance(eos,int) else list(eos or []))+[tok.pad_token_id])
    lengths = [next((i for i,t in enumerate(row) if t in stops),len(row)) for row in tokens]
    states = capture(hybrid.baseline,tok,ids,tokens,stops)
    return {'texts':texts,'tokens':tokens,'prompt_ids':ids,'states':states,
            'input_tokens':[len(p) for p in ids],'output_tokens':lengths,
            'at_cap':[n==cap for n in lengths]}


def collect(key,split,hybrid,tok,geometry=None,batch_size=32):
    from experiments.recovery_help.backend import chat
    data, actions = helpers()
    manifest = prepare()
    raw = json.loads((ROOT/'data'/f'{split}.json').read_text())
    fingerprint = {'source':sha(Path(__file__)),'data':manifest['hashes'][split],
                   'model_identity':sha(data.OUT/key/'model_hashes.json'),'batch_size':batch_size}
    if geometry is not None:
        fingerprint['geometry'] = sha(ROOT/key/'geometry.joblib')
    directory = ROOT/key/split; directory.mkdir(parents=True,exist_ok=True)
    start_time = time.monotonic()
    for start in range(0,len(raw),batch_size):
        chunk = raw[start:start+batch_size]; path = directory/f'batch_{start:05d}.json'
        if path.exists():
            existing = json.loads(path.read_text())
            if existing['fingerprint'] != fingerprint or sha(path.with_suffix('.npz')) != existing['states_sha256']:
                raise ValueError('Changed collection protocol/cache')
            continue
        public = [actions.public_example(r) for r in chunk]
        prompts = [chat(tok,actions.prompt(r,'keep')) for r in public]
        result = generate_capture(hybrid,tok,prompts,64)
        baseline = [actions.extract_answer(t) for t in result['texts']]
        anchors = [a or '(no valid initial answer)' for a in baseline]
        # Freeze decisions before executing any mitigation on this test batch.
        if geometry is not None:
            predictions = geometry.predict(result['states'])
            predpath = directory/f'predictions_{start:05d}.npz'
            if predpath.exists():
                old = np.load(predpath)
                if set(old.files) != set(predictions) or any(not np.array_equal(old[k],v) for k,v in predictions.items()):
                    raise ValueError('Changed frozen test predictions')
            else:
                np.savez_compressed(predpath,**predictions)
            save(directory/f'predictions_{start:05d}.json',{'ids':[r['id'] for r in chunk],
                'sha256':sha(predpath),'geometry':fingerprint['geometry'],'time_unix':time.time()})
        rows = [{'id':r['id'],'actions':[]} for r in chunk]; allstates=[]
        for j,action in enumerate(ACTIONS):
            if j:
                prompts = [chat(tok,actions.prompt(r,action,a)) for r,a in zip(public,anchors)]
                result = generate_capture(hybrid,tok,prompts,actions.cap_for(action))
            allstates.append(result['states'])
            for i,row in enumerate(rows):
                row['actions'].append({'name':action,'text':result['texts'][i],
                    'answer':actions.extract_answer(result['texts'][i],action in ('reason','both')),
                    'prompt_ids':result['prompt_ids'][i],'output_ids':result['tokens'][i],
                    'input_tokens':result['input_tokens'][i],'output_tokens':result['output_tokens'][i],
                    'at_cap':result['at_cap'][i]})
        npz = path.with_suffix('.npz'); tmp = npz.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,states=np.stack(allstates,axis=1))
        tmp.replace(npz)
        save(path,{'fingerprint':fingerprint,'states_sha256':sha(npz),'rows':rows,
                   'completed_time_unix':time.time()})
        print(json.dumps({'model':key,'split':split,'done':start+len(chunk),'total':len(raw),
                          'seconds':round(time.monotonic()-start_time,1)}),flush=True)
    save(directory/'complete.json',{'n':len(raw),'seconds_this_run':time.monotonic()-start_time})


def technical_smoke(key,hybrid,tok):
    from experiments.recovery_help.backend import generate_batch,chat
    _,actions = helpers()
    raw = json.loads((ROOT/'data'/'smoke.json').read_text())[:2]
    prompts = [chat(tok,actions.prompt(actions.public_example(r),'keep')) for r in raw]
    native = generate_batch(hybrid.baseline,tok,prompts,64,capture=True)
    eos = hybrid.baseline.generation_config.eos_token_id
    stops = set(([eos] if isinstance(eos,int) else list(eos or []))+[tok.pad_token_id])
    replay = capture(hybrid.baseline,tok,[tok.encode(p,add_special_tokens=False) for p in prompts],native['ids'],stops)
    difference = replay.astype(float)-native['states'].astype(float)
    relative = np.linalg.norm(difference,axis=-1)/np.maximum(np.linalg.norm(native['states'].astype(float),axis=-1),1e-9)
    save(ROOT/key/'technical_smoke.json',{'relative_l2_max':float(relative.max()),
        'relative_l2_mean':float(relative.mean()),'n':2,'native_layers':native['layers']})
    if relative.max()>.1:
        raise ValueError('Replay/native measurement disagreement exceeds 10%')
    print(json.dumps({'model':key,'technical_capture_max_relative_l2':float(relative.max())}),flush=True)


def run(key,phase):
    from experiments.recovery_scale.backend import load
    from experiments.recovery_scale.verified_collect import verify_identity
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Use PYTHONHASHSEED=0')
    prepare(); data,_ = helpers()
    root = ROOT/key; root.mkdir(exist_ok=True)
    save(root/'identity_before.json',verify_identity(data.MODEL_PATHS[key],data.OUT/key/'model_hashes.json'))
    save(root/'execution.json',{'source_sha256':sha(Path(__file__)),'protocol_sha256':sha(CODE/'README.md'),
        'actions':ACTIONS,'batch_size':32,'seed':SEED,'phase':phase,'started_unix':time.time(),
        'generation_engine':'vllm','capture_engine':'transformers exact-token replay'})
    hybrid,tok = load(key)
    if phase in ('all','calibration'):
        technical_smoke(key,hybrid,tok)
        collect(key,'train',hybrid,tok)
        collect(key,'validation',hybrid,tok)
        model = fit(key)
    else:
        model = joblib.load(root/'geometry.joblib')
    if phase in ('all','test'):
        collect(key,'test',hybrid,tok,model)
    save(root/'identity_after.json',verify_identity(data.MODEL_PATHS[key],data.OUT/key/'model_hashes.json'))


if __name__=='__main__':
    # Use a stable module path in joblib artifacts even when run as a script.
    from experiments.activation_drift import pilot as implementation
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=('qwen7b','phi35'))
    parser.add_argument('--phase',choices=('prepare','all','calibration','test','fit'),default='all')
    args = parser.parse_args()
    if args.phase == 'prepare':
        print(json.dumps({k:v for k,v in implementation.prepare().items() if k!='ids'},indent=2))
    elif args.phase == 'fit':
        implementation.fit(args.model)
    else:
        implementation.run(args.model,args.phase)
