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
from experiments.paired_evidence.evaluate import arrays,frozen_predictions,quality,stats,interval
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
    result=dict(model=key,post_action_information_used=True,actual_clean_used_only_in_named_oracle=True,
        usable_for_deployment=False,acceptance_decision_changed=False,
        n_base_questions=len(cases),offline_objectives=results,
        actual_error_logit_reduction_vs_actual_f1_gain_spearman=float(correlation.statistic),
        forecast_error_logit_transition_mse=float(np.mean((prediction['forecast_error_logits'][:,:,:4]-q[:,:,:4])**2)),
        oracle_action_f1=float(f1[:,:,:4].max(-1).mean()),prediction_audit=audit,
        source_sha256=file_digest(__file__),model_sha256=file_digest(OUT/key/'fitted/best.joblib'),
        time_utc=datetime.now(timezone.utc).isoformat(),
        note='Descriptive offline oracle decomposition, not a fresh policy claim or mitigation-success fitting')
    dump(REPO/'results/error_field'/f'{key}_offline_diagnostics.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=('qwen7b','phi35'),required=True)
    a=p.parse_args();diagnose(a.model)
