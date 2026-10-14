"""Offline objective checks using post-action states, never deployable policies.

These descriptive comparisons isolate the objective/reference from forecasting.
They cannot change any frozen router or its acceptance decision.
"""
import argparse
from datetime import datetime,timezone
import json
import joblib
import numpy as np
from scipy.stats import spearmanr
from experiments.paired_evidence.data import OUT
from experiments.paired_evidence.evaluate import arrays,frozen_predictions,quality,stats,interval,matched_interval
from experiments.retrieval_budget.evaluate import missing_margin,analysis_sources
from experiments.recovery_help.data import dump,file_digest
from experiments.paths import REPO


def diagnose(key):
    data,rows,cases,f1,em,tokens=arrays(key,'test')
    prediction,audit=frozen_predictions(key,'test',data)
    model=joblib.load(OUT/key/'fitted/best.joblib')
    initial=model.coordinates.transform(data['states'][:,:,0])[0]
    shape=initial.shape[:-1];x=initial.reshape(-1,initial.shape[-1])
    denoised=model.reference.predict(x).reshape(initial.shape)
    clean=np.broadcast_to(initial[:,:1],initial.shape)
    projected=np.stack([model.coordinates.transform(data['states'][:,:,a])[0] for a in range(6)],axis=-2)
    q=model.probe.decision_function(projected.reshape(-1,projected.shape[-1])).reshape(*shape,6)
    q_gain=q[:,:,:1]-q
    reference_distance=((projected-denoised[:,:,None])**2).sum(-1)
    clean_distance=((projected-clean[:,:,None])**2).sum(-1)
    policies={
        'actual_post_action_error_readout':q[:,:,:4].argmin(-1),
        'actual_post_action_error_readout_plus_fixed_penalty':(q+model.field.sigma)[:,:,:4].argmin(-1),
        'actual_post_action_distance_to_predicted_clean':reference_distance[:,:,:4].argmin(-1),
        'actual_post_action_distance_plus_fixed_penalty':(reference_distance+model.reference_field.penalty)[:,:,:4].argmin(-1),
        'actual_post_action_distance_to_actual_clean':clean_distance[:,:,:4].argmin(-1)}
    results={}
    for name,choices in policies.items():
        results[name]=dict(stats(f1,em,tokens,choices),
            vs_keep=interval(quality(f1,choices)-f1[:,:,0]),
            agreement_with_frozen_error_policy=float(np.mean(choices==prediction['choices'])),
            agreement_with_frozen_reference_policy=float(np.mean(choices==prediction['reference_choices'])))
    actual_gain=f1-f1[:,:,:1]
    correlation=spearmanr(q_gain[:,:,1:4].ravel(),actual_gain[:,:,1:4].ravel())
    strong_controls={}
    for name,extractor in (('public',model.observable),('confidence',model.confidence)):
        inputs=extractor.features(data['texts'],data['confidence'],
            public=data['public'] if name=='public' else None)
        margin=missing_margin(extractor.model.predict(inputs)).reshape(shape)
        strong_controls[name]=matched_interval(f1,prediction['choices'],margin,4000)
    result=dict(model=key,post_action_information_used=True,actual_clean_used_only_in_named_oracle=True,
        usable_for_deployment=False,acceptance_decision_changed=False,
        n_base_questions=len(cases),offline_objectives=results,
        actual_error_logit_reduction_vs_actual_f1_gain_spearman=float(correlation.statistic),
        forecast_active_error_logit_transition_mse=float(np.mean((prediction['forecast_error_logits'][:,:,1:4]-q[:,:,1:4])**2)),
        strong_budget_controls=strong_controls,
        strong_budget_controls_posthoc_descriptive=True,
        oracle_action_f1=float(f1[:,:,:4].max(-1).mean()),prediction_audit=audit,
        source_sha256=file_digest(__file__),source_hashes=analysis_sources(),
        model_sha256=file_digest(OUT/key/'fitted/best.joblib'),
        time_utc=datetime.now(timezone.utc).isoformat(),
        note='Descriptive offline oracle decomposition, not a fresh policy claim or mitigation-success fitting')
    dump(REPO/'results/error_field'/f'{key}_offline_diagnostics.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def report():
    root=REPO/'results/error_field'
    results={k:json.loads((root/f'{k}_offline_diagnostics.json').read_text()) for k in ('qwen7b','phi35')}
    names={'actual_post_action_error_readout':'Actual error readout',
        'actual_post_action_error_readout_plus_fixed_penalty':'Actual error readout + penalty',
        'actual_post_action_distance_to_predicted_clean':'Actual distance to predicted clean',
        'actual_post_action_distance_plus_fixed_penalty':'Actual distance to predicted clean + penalty',
        'actual_post_action_distance_to_actual_clean':'Actual distance to actual clean (oracle)'}
    lines=['# Offline objective diagnostics','',
        '**These are not deployable policies.** Every row below uses actual post-action activations. '
        'The named actual-clean oracle additionally receives the matched clean baseline state. '
        'These descriptive comparisons do not change any fitted policy or acceptance decision.','',
        '| Objective using actual action states | Qwen F1 | Phi F1 |','|---|---:|---:|']
    keep={k:r['offline_objectives']['actual_post_action_error_readout']['f1']-r['offline_objectives']['actual_post_action_error_readout']['vs_keep']['mean'] for k,r in results.items()}
    lines.append('| Keep | '+' | '.join(f'{100*v:.2f}' for v in keep.values())+' |')
    for name,label in names.items():
        lines.append('| '+label+' | '+' | '.join(f"{100*r['offline_objectives'][name]['f1']:.2f}" for r in results.values())+' |')
    lines += ['', 'Different objective policies may use different actions and call counts; their '
        'F1 differences alone do not isolate forecasting error at equal cost. These comparisons '
        'test whether a frozen objective is useful even when its action-state forecast is replaced '
        'by the actual post-action state. They do not establish a causal error mechanism.','',
        'With actual action states and the fixed reference-distance penalty, keep is selected '
        'in '+', '.join(f"{r['offline_objectives']['actual_post_action_distance_plus_fixed_penalty']['counts']['keep']}/{3*r['n_base_questions']} {k}" for k,r in results.items())+
        ' inputs. This shows the estimated-reference objective itself can discourage nearly all '
        'interventions, even when action forecasting is removed.','',
        '## Additional fixed-budget controls','',
        'These **posthoc descriptive** controls evaluate the unchanged error-risk policy against '
        'the best single action and affordable threshold in hindsight, ranked by the frozen '
        'public/confidence missing-condition margin. Cheaper sham/empty actions and unused '
        'calls are allowed. Every bootstrap retains the three versions of a base question.','',
        '| Gate for fixed-action control | Qwen difference (pp, 95% interval) | Phi difference |',
        '|---|---:|---:|']
    for name in ('public','confidence'):
        values=[]
        for r in results.values():
            v=r['strong_budget_controls'][name]
            values.append(f"{100*v['mean']:+.2f} [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]")
        lines.append('| '+name+' | '+' | '.join(values)+' |')
    confidence_pass=all(r['strong_budget_controls']['confidence']['ci95'][0]>0 for r in results.values())
    lines += ['', ('Both confidence-budget differences have positive lower interval bounds.' if confidence_pass else
        'An advantage over the confidence-budget control is not established on both backbones.')+
        ' These posthoc controls do not revise the main policy acceptance result. Extra-call '
        'matching does not match tokens, latency or activation-extraction overhead.','',
        'Raw details, selected action counts, source/model hashes and timing audits: '
        '[Qwen](qwen7b_offline_diagnostics.json), [Phi](phi35_offline_diagnostics.json). '
        'See the [main results](README.md) and [method](../../experiments/error_field/README.md).','']
    (root/'OFFLINE_DIAGNOSTICS.md').write_text('\n'.join(lines))
    print(root/'OFFLINE_DIAGNOSTICS.md')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('qwen7b','phi35'))
    p.add_argument('--report',action='store_true');a=p.parse_args()
    if a.report:report()
    elif a.model:diagnose(a.model)
    else:p.error('Use --model or --report')
