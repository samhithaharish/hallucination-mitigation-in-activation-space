"""Paired clean-state reference; no correctness targets for this policy."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from experiments.paths import REPO
from experiments.paired_evidence.data import OUT
from experiments.paired_evidence.fit import load,Directions
from experiments.recovery_help.data import dump,digest,file_digest
from .fit import RiskRouter,MODELS


class Reference:
    def fit(self,x,target,vx,vtarget):
        baseline=float(np.mean((vx-vtarget)**2));candidates=[(baseline,'identity',None)]
        for alpha in (.01,.1,1.,10.,100.,1000.):
            model=Ridge(alpha=alpha).fit(x,target)
            candidates.append((float(np.mean((model.predict(vx)-vtarget)**2)),alpha,model))
        error,alpha,self.model=min(candidates,key=lambda r:r[0])
        self.details=dict(validation_clean_state_mse=error,identity_mse=baseline,alpha=alpha)
        return self

    def predict(self,x):return x if self.model is None else self.model.predict(x)


class StateDelta:
    def __init__(self,kind='ridge',alpha=10.,features='state'):
        self.kind=kind;self.alpha=alpha;self.features=features

    def fit(self,x,target,vx,vtarget):
        self.means=target.mean(0);self.operators=[None];errors=[0.];nmse=[]
        for action in range(1,target.shape[1]):
            op=None if self.kind=='mean' else Ridge(alpha=self.alpha).fit(x,target[:,action])
            prediction=np.broadcast_to(self.means[action],vtarget[:,action].shape) if op is None else op.predict(vx)
            squared=(prediction-vtarget[:,action])**2
            errors.append(float(squared.sum(-1).mean()))
            if action<4:nmse.append(float(np.mean(squared/np.maximum(target[:,action].var(0),.01))))
            self.operators.append(op)
        self.penalty=np.array(errors);self.metric=float(np.mean(nmse))
        return self

    def moves(self,x):
        parts=[np.zeros((len(x),self.means.shape[-1]))]
        for a,op in enumerate(self.operators[1:],1):
            parts.append(np.broadcast_to(self.means[a],parts[0].shape) if op is None else op.predict(x))
        return np.stack(parts,axis=1)

    def scores(self,x,residual):
        moves=self.moves(x)
        return (residual**2).sum(-1)[:,None]-((residual[:,None]+moves)**2).sum(-1)-self.penalty


class Residual:
    def fit(self,x,target,vx,vtarget):
        self.mean=target.mean(0);candidates=[(float(np.mean((vtarget-self.mean)**2)),None,None)]
        for alpha in (1.,10.,100.,1000.):
            model=Ridge(alpha=alpha).fit(x,target)
            candidates.append((float(np.mean((model.predict(vx)-vtarget)**2)),alpha,model))
        error,alpha,self.model=min(candidates,key=lambda r:r[0]);self.details=dict(mse=error,alpha=alpha)
        return self

    def predict(self,x):
        return np.broadcast_to(self.mean,(len(x),len(self.mean))) if self.model is None else self.model.predict(x)


class ReferenceRouter(RiskRouter):
    def predict(self,states,texts=None,confidence=None,public=None):
        result=super().predict(states,texts,confidence,public)
        x,z=self.coordinates.transform(states);shape=z.shape[:-1];flat=x.reshape(-1,x.shape[-1])
        residual=flat-self.reference.predict(flat)
        design=Directions(self.coordinates,features=self.reference_field.features).inputs(x,z)
        scores=self.reference_field.scores(design,residual).reshape(*shape,-1)
        result.update(reference_scores=scores[...,:4],reference_choices=scores[...,:4].argmax(-1),
            reference_control_choices=scores.argmax(-1),reference_risk=(residual**2).sum(-1).reshape(shape))
        if texts is not None and confidence is not None:
            for prefix,extractor in (('public',self.observable),('confidence',self.confidence)):
                inputs=extractor.features(texts,confidence,public=public if prefix=='public' else None)
                r=getattr(self,'reference_'+prefix+'_residual').predict(inputs)
                score=getattr(self,'reference_'+prefix+'_field').scores(inputs,r)
                name='reference_observable_rule' if prefix=='public' else 'reference_confidence_rule'
                result[name]=score[:,:4].argmax(1).reshape(shape)
        return result


def fit_all():
    checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Commit sources first')
    bundles={}
    for key in MODELS:
        path=OUT/key/'fitted/best.joblib'
        if path.with_name('error_only.joblib').exists():raise ValueError('A composite router already exists')
        base=joblib.load(path);tr,va=load(key,'train'),load(key,'validation')
        states=base.coordinates.transform(tr['states'][:,:,0])[0];vstates=base.coordinates.transform(va['states'][:,:,0])[0]
        x=states.reshape(-1,states.shape[-1]);vx=vstates.reshape(-1,vstates.shape[-1])
        z=base.coordinates.transform(tr['states'][:,:,0])[1];vz=base.coordinates.transform(va['states'][:,:,0])[1]
        clean=np.repeat(states[:,:1],3,axis=1).reshape(x.shape)
        vclean=np.repeat(vstates[:,:1],3,axis=1).reshape(vx.shape)
        reference=Reference().fit(x,clean,vx,vclean)
        target=[np.zeros_like(x)];vtarget=[np.zeros_like(vx)]
        for a in range(1,6):
            target.append(base.coordinates.transform(tr['states'][:,:,a])[0].reshape(x.shape)-x)
            vtarget.append(base.coordinates.transform(va['states'][:,:,a])[0].reshape(vx.shape)-vx)
        bundles[key]=dict(base=base,states=states,vstates=vstates,z=z,vz=vz,x=x,vx=vx,
            target=np.stack(target,axis=1),vtarget=np.stack(vtarget,axis=1),reference=reference,
            residual=x-reference.predict(x),vresidual=vx-reference.predict(vx),
            train=tr,validation=va)
    configs=[dict(kind='mean',alpha=0.,features='state')]+[dict(kind='ridge',alpha=a,features=f)
        for f in ('z','state','bilinear') for a in (1.,10.,100.,1000.)]
    trials=[];models={}
    for i,cfg in enumerate(configs):
        models[i]={};metrics={}
        for key,b in bundles.items():
            x=Directions(b['base'].coordinates,features=cfg['features']).inputs(b['states'],b['z'])
            vx=Directions(b['base'].coordinates,features=cfg['features']).inputs(b['vstates'],b['vz'])
            field=StateDelta(**cfg).fit(x,b['target'],vx,b['vtarget'])
            models[i][key]=field;metrics[key]=field.metric
        trials.append(dict(index=i,config=cfg,metric=float(np.mean(list(metrics.values()))),per_model=metrics))
        print(json.dumps(trials[-1]),flush=True)
    best=min(trials,key=lambda t:t['metric']);frozen={}
    for key,b in bundles.items():
        model=ReferenceRouter();model.__dict__.update(b['base'].__dict__)
        model.reference=b['reference'];model.reference_field=models[best['index']][key]
        comparator_details={}
        for prefix,extractor in (('public',model.observable),('confidence',model.confidence)):
            tr,va=b['train'],b['validation']
            x=extractor.features(tr['texts'],tr['confidence'],public=tr['public'] if prefix=='public' else None)
            vx=extractor.features(va['texts'],va['confidence'],public=va['public'] if prefix=='public' else None)
            residual=Residual().fit(x,b['residual'],vx,b['vresidual'])
            candidates=[StateDelta(kind='mean').fit(x,b['target'],vx,b['vtarget'])]+[
                StateDelta(alpha=a).fit(x,b['target'],vx,b['vtarget']) for a in (1.,10.,100.,1000.)]
            field=min(candidates,key=lambda m:m.metric)
            setattr(model,'reference_'+prefix+'_residual',residual);setattr(model,'reference_'+prefix+'_field',field)
            comparator_details[prefix]=dict(residual=residual.details,field_nmse=field.metric,kind=field.kind,alpha=field.alpha)
        va=b['validation'];old=b['base'].predict(va['states'][:,:,0],va['texts'],va['confidence'],public=va['public'])
        new=model.predict(va['states'][:,:,0],va['texts'],va['confidence'],public=va['public'])
        for name in old:np.testing.assert_array_equal(old[name],new[name])
        path=OUT/key/'fitted/best.joblib';parent=json.loads(path.with_suffix('.json').read_text())
        shutil.copy2(path,path.with_name('error_only.joblib'))
        shutil.copy2(path.with_suffix('.json'),path.with_name('error_only.json'))
        sources={str(p.relative_to(REPO)):file_digest(p) for folder in ('error_field','paired_evidence')
                 for p in (REPO/'experiments'/folder).glob('*.py')}
        details=dict(parent,source_checkpoint=checkpoint,source_hashes=sources,sources=sources,
            parent_fit_sources=parent['sources'],
            parent_error_router_sha256=parent['sha256'],error_policy_parameters_unchanged=True,
            reference=dict(config=best['config'],shared_validation_activation_nmse=best['metric'],
                denoiser=b['reference'].details,comparators=comparator_details,
                baseline_correctness_labels_used=False,mitigation_success_targets_used=False))
        model.fingerprint=digest({'parent':b['base'].fingerprint,'reference':details['reference'],'sources':sources})
        temporary=path.with_suffix('.joblib.tmp');joblib.dump(model,temporary);temporary.replace(path)
        details.update(sha256=file_digest(path),fingerprint=model.fingerprint,frozen_time_unix=time.time())
        dump(path.with_suffix('.json'),details);frozen[key]=details
    dump(OUT.parent/'error_field/reference_selection.json',dict(best=best,trials=trials,frozen=frozen,
        baseline_correctness_labels_used=False,mitigation_success_targets_used=False,time_unix=time.time()))


if __name__=='__main__':
    from experiments.error_field import reference as implementation
    argparse.ArgumentParser(description=__doc__).parse_args();implementation.fit_all()
