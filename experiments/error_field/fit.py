"""Fit baseline truth supervision and unscored activation-response models."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import joblib
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import log_loss,roc_auc_score

from experiments.paths import REPO
from experiments.paired_evidence.data import OUT
from experiments.paired_evidence.fit import load,Directions,CODES,CONDITION_ACTIONS,SELECTABLE
from experiments.recovery_help.data import dump,digest,file_digest
from experiments.recovery_help.task import score_answer

MODELS=('qwen7b','phi35')


def baseline_targets(rows,cases):
    if [r['id'] for r in rows]!=[c['id'] for c in cases]:raise ValueError('Misaligned baseline labels')
    result=[]
    for row,case in zip(rows,cases):
        targets=[]
        for version in row['versions']:
            original=version['actions'][0]
            if original['name']!='keep':raise ValueError('Truth target must come from original answer')
            targets.append(1.-score_answer(original['answer'],case['answer'])['em'])
        result.append(targets)
    return np.array(result)


def baseline_labels(key,split):
    if split not in ('train','validation'):raise ValueError('No heldout truth labels in fitting')
    rows=[r for p in sorted((OUT/key/split).glob('batch_*.json'))
            for r in json.loads(p.read_text())['rows']]
    cases=json.loads((OUT/'data'/f'{split}.json').read_text())
    return baseline_targets(rows,cases).ravel()


class DeltaModel:
    def __init__(self,kind='ridge',alpha=10.,features='state'):
        self.kind=kind;self.alpha=alpha;self.features=features

    def fit(self,x,target,vx,vtarget):
        self.means=target.mean(0);self.operators=[None];errors=[0.];nmse=[]
        for action in range(1,target.shape[1]):
            operator=None if self.kind=='mean' else Ridge(alpha=self.alpha).fit(x,target[:,action])
            predicted=np.full(len(vx),self.means[action]) if operator is None else operator.predict(vx)
            error=float(np.mean((predicted-vtarget[:,action])**2));errors.append(error)
            if action<SELECTABLE:nmse.append(error/max(float(target[:,action].var()),.01))
            self.operators.append(operator)
        self.sigma=np.sqrt(errors);self.metric=float(np.mean(nmse))
        return self

    def moves(self,x):
        columns=[np.zeros(len(x))]
        for action in range(1,len(self.operators)):
            op=self.operators[action]
            columns.append(np.full(len(x),self.means[action]) if op is None else op.predict(x))
        return np.stack(columns,axis=1)

    def scores(self,x):return -self.moves(x)-self.sigma


def design(coordinates,x,z,q,features):
    return np.concatenate([Directions(coordinates,features=features).inputs(x,z),q.reshape(-1,1)],axis=1)


class RiskRouter:
    def predict(self,states,texts=None,confidence=None,public=None):
        x,z=self.coordinates.transform(states);shape=z.shape[:-1]
        q=self.probe.decision_function(x.reshape(-1,x.shape[-1]))
        xx=design(self.coordinates,x,z,q,self.field.features)
        scores=self.field.scores(xx).reshape(*shape,-1)
        labels=np.sum((z[...,None,:]-CODES)**2,axis=-1).argmin(-1)
        result={'scores':scores[...,:SELECTABLE],'choices':scores[...,:SELECTABLE].argmax(-1),
                'control_scores':scores,'control_choices':scores.argmax(-1),
                'coordinates':z,'risk':expit(q).reshape(shape),'baseline_error_logit':q.reshape(shape),
                'forecast_error_logits':(q[:,None]+self.field.moves(xx)).reshape(*shape,-1),
                'class_rule':CONDITION_ACTIONS[labels],
                'condition_field_choices':self.previous.predict(states)['choices']}
        if texts is not None and confidence is not None:
            ox=self.observable.features(texts,confidence,public=public)
            cx=self.confidence.features(texts,confidence)
            result['observable_rule']=self.public_field.scores(ox)[:,:SELECTABLE].argmax(1).reshape(shape)
            result['confidence_rule']=self.confidence_field.scores(cx)[:,:SELECTABLE].argmax(1).reshape(shape)
            result['condition_observable_rule']=self.observable.predict(texts,confidence,public).reshape(shape)
            result['condition_confidence_rule']=self.confidence.predict(texts,confidence).reshape(shape)
        return result


def bundle(key):
    manifest=json.loads((OUT/'data_manifest.json').read_text())
    source=Path(manifest['source_development'])
    train,val=load(key,'train'),load(key,'validation')
    coordinates,observable,confidence=joblib.load(source/key/'fitted/coordinates.joblib')
    previous=joblib.load(source/key/'fitted/best.joblib')
    x,z=coordinates.transform(train['states'][:,:,0]);vx,vz=coordinates.transform(val['states'][:,:,0])
    xx=x.reshape(-1,x.shape[-1]);vxx=vx.reshape(-1,vx.shape[-1])
    y,vy=baseline_labels(key,'train'),baseline_labels(key,'validation')
    candidates=[]
    for c in (.01,.1,1.,10.):
        probe=LogisticRegression(C=c,max_iter=1000).fit(xx,y)
        candidates.append((float(log_loss(vy,probe.predict_proba(vxx),labels=[0.,1.])),c,probe))
    loss,c,probe=min(candidates,key=lambda r:r[0]);q=probe.decision_function(xx);vq=probe.decision_function(vxx)
    target=[np.zeros(len(q))];vtarget=[np.zeros(len(vq))]
    for action in range(1,6):
        after=coordinates.transform(train['states'][:,:,action])[0].reshape(-1,x.shape[-1])
        vafter=coordinates.transform(val['states'][:,:,action])[0].reshape(-1,x.shape[-1])
        target.append(probe.decision_function(after)-q);vtarget.append(probe.decision_function(vafter)-vq)
    router=RiskRouter();router.coordinates=coordinates;router.probe=probe
    router.observable=observable;router.confidence=confidence;router.previous=previous
    router.probe_details={'C':c,'validation_baseline_error_log_loss':loss,
                          'validation_baseline_error_auroc':float(roc_auc_score(vy,expit(vq))),
                          'baseline_correctness_labels_used':True,'mitigation_success_targets_used':False}
    return dict(router=router,x=x,z=z,q=q,vx=vx,vz=vz,vq=vq,
         target=np.stack(target,axis=1),vtarget=np.stack(vtarget,axis=1),
         ox=observable.features(train['texts'],train['confidence'],public=train['public']),
         vox=observable.features(val['texts'],val['confidence'],public=val['public']),
         cx=confidence.features(train['texts'],train['confidence']),
         vcx=confidence.features(val['texts'],val['confidence']),
         fingerprints={'train':train['fingerprint'],'validation':val['fingerprint'],
                       'coordinate_sha256':file_digest(source/key/'fitted/coordinates.joblib'),
                       'previous_model_sha256':file_digest(source/key/'fitted/best.joblib')})


def fit_all():
    checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Commit method sources before fitting')
    root=OUT.parent/'error_field';root.mkdir(exist_ok=True)
    if any((OUT/k/'fitted/best.joblib').exists() for k in MODELS):raise ValueError('Predictors already frozen')
    bundles={key:bundle(key) for key in MODELS}
    configs=[dict(kind='mean',alpha=0.,features='state')]+[
        dict(kind='ridge',alpha=a,features=f) for f in ('z','state','bilinear') for a in (1.,10.,100.,1000.)]
    all_models={};trials=[]
    for index,config in enumerate(configs):
        models={};metrics={}
        for key,b in bundles.items():
            x=design(b['router'].coordinates,b['x'],b['z'],b['q'],config['features'])
            vx=design(b['router'].coordinates,b['vx'],b['vz'],b['vq'],config['features'])
            model=DeltaModel(**config).fit(x,b['target'],vx,b['vtarget'])
            models[key]=model;metrics[key]=model.metric
        metric=float(np.mean(list(metrics.values())))
        trials.append(dict(index=index,config=config,metric=metric,per_model=metrics))
        all_models[index]=models
        print(json.dumps(trials[-1]),flush=True)
    best=min(trials,key=lambda t:t['metric']);frozen={}
    for key,b in bundles.items():
        router=b['router'];router.field=all_models[best['index']][key]
        baseline_configs=[dict(kind='mean',alpha=0.)]+[dict(kind='ridge',alpha=a) for a in (1.,10.,100.,1000.)]
        comparator_details={}
        for label,prefix in (('public','o'),('confidence','c')):
            candidates=[DeltaModel(**cfg).fit(b[prefix+'x'],b['target'],b['v'+prefix+'x'],b['vtarget'])
                        for cfg in baseline_configs]
            chosen=min(candidates,key=lambda m:m.metric)
            setattr(router,label+'_field',chosen)
            comparator_details[label]=dict(kind=chosen.kind,alpha=chosen.alpha,validation_activation_nmse=chosen.metric)
        sources={str(p.relative_to(REPO)):file_digest(p) for folder in ('error_field','paired_evidence')
                 for p in (REPO/'experiments'/folder).glob('*.py')}
        details=dict(config=best['config'],shared_validation_activation_nmse=best['metric'],
            probe=router.probe_details,comparators=comparator_details,data=b['fingerprints'],
            sources=sources,baseline_correctness_labels_used=True,mitigation_success_targets_used=False,
            pilot_informed_action_design=True,source_checkpoint=checkpoint,
            data_manifest_sha256=file_digest(OUT/'data_manifest.json'))
        router.fingerprint=digest(details)
        path=OUT/key/'fitted/best.joblib';path.parent.mkdir(exist_ok=True)
        joblib.dump(router,path)
        metadata=dict(details,sha256=file_digest(path),fingerprint=router.fingerprint,
                      frozen_time_unix=time.time(),source_hashes=sources)
        dump(path.with_suffix('.json'),metadata);frozen[key]=metadata
    dump(root/'selection.json',dict(trials=trials,best=best,frozen=frozen,
        baseline_correctness_labels_used=True,mitigation_success_targets_used=False,
        heldout_outcomes_used_for_design=False,time_unix=time.time()))
    print(json.dumps({'best':best,'probes':{k:b['router'].probe_details for k,b in bundles.items()}},indent=2))


if __name__=='__main__':
    from experiments.error_field import fit as implementation
    argparse.ArgumentParser(description=__doc__).parse_args()
    implementation.fit_all()
