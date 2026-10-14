"""Fit paired condition coordinates and strategy changes, never action rewards."""
import argparse
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from experiments.recovery_help.data import dump,digest,file_digest
from .data import OUT,prepare
from .task import CONDITIONS,ACTIONS

CODES=np.array([[0.,0.],[1.,0.],[0.,1.]],np.float32)
SELECTABLE=4


def load(key,split):
    manifest=prepare();root=OUT/key/split
    if not (root/'complete.json').exists():raise ValueError('Incomplete paired split: '+split)
    states=[];ids=[];texts=[];confidence=[];hashes={}
    for path in sorted(root.glob('batch_*.json')):
        record=json.loads(path.read_text());npz=path.with_suffix('.npz')
        if file_digest(npz)!=record['states_sha256']:raise ValueError('Changed state array')
        states.append(np.load(npz)['states']);hashes[path.name]=file_digest(path)
        for row in record['rows']:
            ids.append(row['id'])
            if [v['condition'] for v in row['versions']]!=list(CONDITIONS):raise ValueError('Condition order changed')
            for version in row['versions']:
                if [a['name'] for a in version['actions']]!=list(ACTIONS):raise ValueError('Action order changed')
                baseline=version['actions'][0]
                texts.append(baseline['prompt']);confidence.append(baseline['confidence'])
    if ids!=manifest['ids'][split]:raise ValueError('Incomplete or reordered paired IDs')
    cases=json.loads((OUT/'data'/f'{split}.json').read_text())
    public=[case['versions'][c] for case in cases for c in CONDITIONS]
    return {'states':np.concatenate(states),'texts':texts,'public':public,'confidence':np.array(confidence).reshape(len(ids),3,4),
            'ids':ids,'fingerprint':digest(hashes)}


class Coordinates:
    def __init__(self,components=16):self.components=components

    def project(self,states):
        shape=states.shape[:-2];x=states.reshape(-1,*states.shape[-2:]).astype(np.float32)
        value=np.concatenate([p.transform(x[:,layer])[:,keep] for layer,p,keep in self.pcas],axis=1)
        return value.reshape(*shape,-1)

    def transform(self,states):
        raw=self.project(states);x=(raw-self.center)/self.scale
        z=self.probe.predict(x.reshape(-1,x.shape[-1])).reshape(*x.shape[:-1],2)
        return x,z

    def fit(self,train,validation):
        difference=np.concatenate([np.zeros_like(train[:,:1],dtype=np.float32),
            train[:,1:].astype(np.float32)-train[:,:1].astype(np.float32)],axis=1).reshape(-1,*train.shape[-2:])
        self.pcas=[]
        for layer in range(train.shape[-2]):
            d=difference[:,layer]
            if np.max(np.abs(d))<1e-8:continue
            p=PCA(min(self.components,len(d)-1,d.shape[1]),svd_solver='randomized',random_state=20261013).fit(d)
            keep=p.explained_variance_>max(1e-10,float(p.explained_variance_.max())*1e-5)
            if keep.any():self.pcas.append((layer,p,keep))
        if not self.pcas:raise ValueError('No observable paired activation changes')
        x=self.project(train);vx=self.project(validation)
        self.center=x[:,0].mean(0)
        self.scale=np.maximum(x.reshape(-1,x.shape[-1]).std(0),1e-5)
        x=((x-self.center)/self.scale).reshape(-1,x.shape[-1])
        vx=((vx-self.center)/self.scale).reshape(-1,vx.shape[-1])
        target=np.tile(CODES,(len(train),1));vtarget=np.tile(CODES,(len(validation),1))
        candidates=[]
        for alpha in (.01,.1,1.,10.,100.,1000.):
            model=Ridge(alpha=alpha,fit_intercept=False).fit(x,target)
            candidates.append((float(np.mean((model.predict(vx)-vtarget)**2)),alpha,model))
        error,alpha,self.probe=min(candidates,key=lambda row:row[0])
        prediction=self.probe.predict(vx)
        labels=np.sum((prediction[:,None]-CODES[None])**2,axis=-1).argmin(1)
        self.details={'validation_condition_mse':error,'alpha':alpha,'features':x.shape[-1],
                      'validation_condition_accuracy':float(np.mean(labels==np.tile(np.arange(3),len(validation))))}
        return self


def public_metadata(example):
    from experiments.recovery_help.task import retrieval_scores
    current,pool=example['current'],example['candidates']
    scores=np.array(retrieval_scores(example['question'],pool))
    rank=np.argsort(-scores,kind='stable');present={p['title'] for p in current}
    current_scores=np.sort(retrieval_scores(example['question'],current))[::-1]
    coverage=[sum(pool[int(i)]['title'] in present for i in rank[:k])/min(k,len(pool)) for k in (1,2,4,8)]
    return ([len(current),len(pool),sum(p['title'] in present for p in pool)]
        +current_scores[:5].tolist()+np.sort(scores)[::-1][:5].tolist()+coverage)


class Observable:
    def __init__(self,text=True,pool=False):self.text=text;self.pool=pool

    def features(self,texts,confidence,fitting=False,public=None):
        conf=np.asarray(confidence).reshape(-1,4)
        parts=[conf,np.array([[len(t.split())] for t in texts])]
        if self.pool:
            if public is None:raise ValueError('Pool-aware comparator requires public baseline inputs')
            parts.append(np.array([public_metadata(e) for e in public]))
        if self.text:
            sparse=self.vectorizer.fit_transform(texts) if fitting else self.vectorizer.transform(texts)
            if fitting:
                self.svd=TruncatedSVD(min(64,sparse.shape[0]-1,sparse.shape[1]-1),random_state=20261013)
                parts.append(self.svd.fit_transform(sparse))
            else:parts.append(self.svd.transform(sparse))
        x=np.concatenate(parts,axis=1)
        return self.scale.fit_transform(x) if fitting else self.scale.transform(x)

    def fit(self,texts,confidence,public=None):
        self.scale=StandardScaler()
        if self.text:self.vectorizer=TfidfVectorizer(max_features=20000,min_df=2,ngram_range=(1,2),sublinear_tf=True)
        x=self.features(texts,confidence,True,public)
        self.model=Ridge(alpha=10.).fit(x,np.tile(CODES,(len(texts)//3,1)))
        return self

    def predict(self,texts,confidence,public=None):
        z=self.model.predict(self.features(texts,confidence,public=public))
        labels=np.sum((z[:,None]-CODES[None])**2,axis=-1).argmin(1)
        return np.array([0,1,3])[labels]


class Directions:
    def __init__(self,coordinates,kind='ridge',alpha=10.,features='state',observable=None,confidence=None):
        self.coordinates=coordinates;self.kind=kind;self.alpha=alpha;self.features=features
        self.observable=observable;self.confidence=confidence
        self.fingerprint='unfrozen'

    def inputs(self,x,z):
        return (np.concatenate([x,z],axis=-1) if self.features=='state' else z).reshape(-1,(x.shape[-1]+2) if self.features=='state' else 2)

    def fit(self,train,validation):
        x,z=self.coordinates.transform(train[:,:,0]);vx,vz=self.coordinates.transform(validation[:,:,0])
        inputs,vinputs=self.inputs(x,z),self.inputs(vx,vz)
        self.operators=[None];self.means=[np.zeros(2)];errors=[0.];nmse=[]
        for action in range(1,SELECTABLE):
            delta=(self.coordinates.transform(train[:,:,action])[1]-z).reshape(-1,2)
            vd=(self.coordinates.transform(validation[:,:,action])[1]-vz).reshape(-1,2)
            mean=delta.mean(0);self.means.append(mean)
            operator=None if self.kind=='mean' else Ridge(alpha=self.alpha).fit(inputs,delta)
            prediction=np.broadcast_to(mean,vd.shape) if operator is None else operator.predict(vinputs)
            squared=(prediction-vd)**2
            errors.append(float(squared.sum(1).mean()))
            nmse.append(float(np.mean(squared/np.maximum(delta.var(0),.01))))
            self.operators.append(operator)
        self.penalty=np.array(errors)
        self.details={'kind':self.kind,'alpha':self.alpha,'features':self.features,
                      'validation_activation_nmse':float(np.mean(nmse)),'penalty':errors}
        return self

    def predict(self,states,texts=None,confidence=None,public=None):
        x,z=self.coordinates.transform(states);shape=z.shape[:-1]
        inputs=self.inputs(x,z)
        moves=[np.zeros_like(z)]
        for action in range(1,SELECTABLE):
            op=self.operators[action]
            movement=np.broadcast_to(self.means[action],z.shape) if op is None else op.predict(inputs).reshape(*shape,2)
            moves.append(movement)
        moves=np.stack(moves,axis=-2)
        score=np.sum(z*z,axis=-1)[...,None]-np.sum((z[...,None,:]+moves)**2,axis=-1)-self.penalty
        labels=np.sum((z[...,None,:]-CODES)**2,axis=-1).argmin(-1)
        result={'scores':score,'choices':score.argmax(-1),'risk':np.sum(z*z,axis=-1),
                'coordinates':z,'class_rule':np.array([0,1,3])[labels]}
        if texts is not None and confidence is not None:
            result['observable_rule']=self.observable.predict(texts,confidence,public).reshape(shape)
            result['confidence_rule']=self.confidence.predict(texts,confidence).reshape(shape)
        return result


def coordinate_bundle(key):
    root=OUT/key/'fitted';root.mkdir(exist_ok=True)
    path=root/'coordinates.joblib'
    train=load(key,'train');val=load(key,'validation')
    fingerprint=digest({'train':train['fingerprint'],'validation':val['fingerprint'],
                        'fit_source':file_digest(Path(__file__))})
    if path.exists():
        meta=json.loads(path.with_suffix('.json').read_text())
        if meta['fingerprint']!=fingerprint or meta['sha256']!=file_digest(path):raise ValueError('Changed frozen coordinate bundle')
        return joblib.load(path),train,val
    coordinates=Coordinates().fit(train['states'][:,:,0],val['states'][:,:,0])
    observable=Observable(True,pool=True).fit(train['texts'],train['confidence'],train['public'])
    confidence=Observable(False).fit(train['texts'],train['confidence'])
    bundle=(coordinates,observable,confidence)
    joblib.dump(bundle,path)
    dump(path.with_suffix('.json'),{'fingerprint':fingerprint,'sha256':file_digest(path),
         'coordinates':coordinates.details,'action_success_labels_used':False})
    return bundle,train,val


def run(key,trial,metric_only=False):
    bundle,train,val=coordinate_bundle(key)
    config=json.loads((Path(__file__).parent/'predictor.json').read_text())
    model=Directions(bundle[0],observable=bundle[1],confidence=bundle[2],**config).fit(train['states'],val['states'])
    model.fingerprint=digest({'config':config,'coordinates':file_digest(OUT/key/'fitted/coordinates.joblib'),
                             'fit_source':file_digest(Path(__file__))})
    path=OUT/key/'fitted'/f'{trial}.joblib';joblib.dump(model,path)
    dump(path.with_suffix('.json'),{'config':config,'metric':model.details,'fingerprint':model.fingerprint,
         'sha256':file_digest(path),'time_unix':time.time(),'action_success_labels_used':False})
    print(model.details['validation_activation_nmse'] if metric_only else json.dumps(model.details,indent=2))


if __name__=='__main__':
    from experiments.paired_evidence import fit as implementation
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=('qwen7b','phi35'))
    p.add_argument('--trial',default='baseline');p.add_argument('--metric-only',action='store_true');a=p.parse_args()
    implementation.run(a.model,a.trial,a.metric_only)
