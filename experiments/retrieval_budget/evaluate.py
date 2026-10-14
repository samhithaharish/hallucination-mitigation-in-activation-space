"""Secondary cost-matched check of the frozen missing-evidence rule."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import joblib
import numpy as np
from experiments.paths import REPO
from experiments.paired_evidence.data import OUT
from experiments.paired_evidence.fit import CODES
from experiments.paired_evidence.evaluate import (arrays,frozen_predictions,quality,stats,
    interval,matched_interval,shuffle_interval)
from experiments.recovery_help.data import dump,file_digest

MODELS=('qwen7b','phi35')
ROOT=REPO/'results/retrieval_budget'
CONTROLS=('vs_keep','vs_fixed_activation_gate','vs_fixed_public_gate',
          'vs_fixed_confidence_gate','vs_risk_shuffle')
BOOTSTRAP=dict(draws=4000,seed=20261013,bins=10,unit='base_question',
               interval='percentile',coverage=.95)


def analysis_sources():
    paths=[REPO/'experiments/paths.py']
    for name in ('retrieval_budget','paired_evidence','error_field','recovery_help'):
        paths.extend(sorted((REPO/'experiments'/name).glob('*.py')))
    return {str(p.relative_to(REPO)):file_digest(p) for p in paths}


def check_sources(expected):
    for relative,sha in expected.items():
        if file_digest(REPO/relative)!=sha:raise ValueError('Changed registered source: '+relative)


def check_model_sources(key):
    path=OUT/key/'fitted/best.joblib'
    meta=json.loads(path.with_suffix('.json').read_text())
    if file_digest(path)!=meta['sha256']:raise ValueError('Changed frozen policy model')
    sources={('experiments/paired_evidence/'+p if '/' not in p else p):sha
             for p,sha in meta['source_hashes'].items()}
    check_sources(sources)


def missing_margin(z):
    d=((z[...,None,:]-CODES)**2).sum(-1)
    return np.minimum(d[...,0],d[...,2])-d[...,1]


def compare(f1,choices,risk,public_margin,confidence_margin,draws=4000):
    return dict(vs_keep=interval(quality(f1,choices)-f1[:,:,0],draws),
        vs_fixed_activation_gate=matched_interval(f1,choices,risk,draws),
        vs_fixed_public_gate=matched_interval(f1,choices,public_margin,draws),
        vs_fixed_confidence_gate=matched_interval(f1,choices,confidence_margin,draws),
        vs_risk_shuffle=shuffle_interval(f1[:,:,:4],choices,risk,draws))


def freeze():
    path=ROOT/'registration.json'
    if path.exists():raise ValueError('Preserve the existing secondary registration')
    if any((OUT/k/'test/complete.json').exists() for k in MODELS):
        raise ValueError('Register before either fresh test collection completes')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Commit analysis sources before registering')
    for key in MODELS:check_model_sources(key)
    record=dict(time_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_sha256=file_digest(Path(__file__)),source_hashes=analysis_sources(),
        cohort_manifest_sha256=file_digest(OUT/'data_manifest.json'),
        frozen_model_sha256={k:file_digest(OUT/k/'fitted/best.joblib') for k in MODELS},
        policy='unchanged class_rule: detect missing evidence, retrieve; otherwise keep',
        controls=CONTROLS,bootstrap=BOOTSTRAP,
        criterion='positive lower 95% base-question bootstrap bound for every control on both models',
        origin='Exploratory budget analysis of the preceding paired test; registered during fresh collection before either model completed and before fresh outcomes were inspected',
        multiplicity='Secondary exploratory claim; nominal intervals do not control the broader iterative project family',
        fitting_or_action_changes=False,mitigation_success_training=False)
    dump(path,record);print(json.dumps(record,indent=2))


def evaluate(key,label):
    registration=json.loads((ROOT/'registration.json').read_text())
    check_sources(registration['source_hashes'])
    check_model_sources(key)
    if registration['bootstrap']!=BOOTSTRAP:raise ValueError('Changed registered bootstrap settings')
    if label=='replication':
        if file_digest(OUT/'data_manifest.json')!=registration['cohort_manifest_sha256']:
            raise ValueError('Different registered replication cohort')
        if file_digest(OUT/key/'fitted/best.joblib')!=registration['frozen_model_sha256'][key]:
            raise ValueError('Different registered policy model')
    data,rows,cases,f1,em,tokens=arrays(key,'test')
    prediction,audit=frozen_predictions(key,'test',data)
    model=joblib.load(OUT/key/'fitted/best.joblib');choices=prediction['class_rule']
    if not np.isin(choices,[0,1]).all():raise ValueError('Only the frozen keep/retrieve rule is eligible')
    margins={}
    for name,extractor in (('public',model.observable),('confidence',model.confidence)):
        x=extractor.features(data['texts'],data['confidence'],public=data['public'] if name=='public' else None)
        margins[name]=missing_margin(extractor.model.predict(x)).reshape(choices.shape)
    risk=(prediction['coordinates']**2).sum(-1)
    comparison=compare(f1,choices,risk,margins['public'],margins['confidence'],BOOTSTRAP['draws'])
    passed=all(comparison[c]['ci95'][0]>0 for c in CONTROLS)
    result=dict(model=key,label=label,n_base_questions=len(cases),policy=stats(f1,em,tokens,choices),
        baseline_keep_f1=float(f1[:,:,0].mean()),comparisons=comparison,all_controls_passed=passed,
        default_public=stats(f1,em,tokens,prediction['condition_observable_rule'] if 'condition_observable_rule' in prediction else prediction['observable_rule']),
        default_confidence=stats(f1,em,tokens,prediction['condition_confidence_rule'] if 'condition_confidence_rule' in prediction else prediction['confidence_rule']),
        prediction_audit=audit,bootstrap=BOOTSTRAP,registration_sha256=file_digest(ROOT/'registration.json'),
        cohort_manifest_sha256=file_digest(OUT/'data_manifest.json'),
        evaluated_time_utc=datetime.now(timezone.utc).isoformat(),
        interpretation='Secondary cost-budget analysis; original primary acceptance decisions remain unchanged')
    dump(ROOT/f'{label}_{key}.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze',action='store_true');p.add_argument('--model',choices=MODELS)
    p.add_argument('--label',choices=('exploration','replication'))
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.model and a.label:evaluate(a.model,a.label)
    else:p.error('Use --freeze or both --model and --label')
