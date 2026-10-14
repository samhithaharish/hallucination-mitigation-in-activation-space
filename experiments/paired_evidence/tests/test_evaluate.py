import importlib
import numpy as np
import pytest


def module():
    try:return importlib.import_module('experiments.paired_evidence.evaluate')
    except ImportError:pytest.fail('Paired evaluation is not implemented')


def test_gap_contraction_can_be_maximal_when_both_prompts_lose_evidence():
    e=module();states=np.ones((4,3,6,2,4))
    states[:,1:,0]+=1
    states[:,:,5]=0
    score=e.gap_scores(states)
    assert (score[:,:,5]>0).all()
    np.testing.assert_allclose(score[:,:,5],score.max(-1))
    np.testing.assert_allclose(score[:,:,0],0)


def test_resampling_keeps_all_versions_of_a_base_together():
    e=module();values=np.tile([0.,1.,2.],(8,1))
    result=e.interval(values,draws=50)
    np.testing.assert_allclose([result['mean'],*result['ci95']],[1.,1.,1.])


def test_risk_shuffle_reestimates_action_frequencies_with_base_resampling():
    e=module();f1=np.broadcast_to([0.,1.,.5,.75],(8,3,4)).copy()
    choices=np.tile([0,1,3],(8,1));risk=np.arange(24).reshape(8,3)
    result=e.shuffle_interval(f1,choices,risk,draws=50,bins=1)
    np.testing.assert_allclose([result['mean'],*result['ci95']],[0.,0.,0.],atol=1e-12)


def test_prediction_written_after_mitigation_started_is_rejected():
    e=module()
    with pytest.raises(ValueError,match='before mitigation'):
        e.check_timing({'time_unix':100.},{'mitigation_start_time_unix':90.,'completed_time_unix':110.})
    e.check_timing({'time_unix':80.},{'mitigation_start_time_unix':90.,'completed_time_unix':110.})


def test_acceptance_requires_all_controls_and_distinguishes_field_added_value():
    e=module()
    positive={'ci95':[.01,.03]};negative={'ci95':[-.01,.03]}
    controls=['vs_keep','vs_matched_risk_fixed','vs_risk_matched_shuffle',
              'vs_observable_rule','vs_confidence_rule']
    policies={name:{k:positive for k in controls} for name in ('choices','class_rule')}
    policies['choices']['vs_class_rule']=negative
    result=e.acceptance(policies)
    assert result['routing']['choices'] and result['routing']['class_rule']
    assert not result['strategy_fields_add_value']
    policies['choices']['vs_confidence_rule']=negative
    assert not e.acceptance(policies)['routing']['choices']


def test_replication_requires_the_same_policy_on_both_backbones():
    e=module()
    q={'acceptance':{'routing':{'choices':True,'class_rule':False},'strategy_fields_add_value':True}}
    p={'acceptance':{'routing':{'choices':False,'class_rule':True},'strategy_fields_add_value':False}}
    assert not any(e.replicated_acceptance({'qwen7b':q,'phi35':p})['routing'].values())
    p['acceptance']=q['acceptance']
    assert e.replicated_acceptance({'qwen7b':q,'phi35':p})['routing']['choices']
    assert e.replicated_acceptance({'qwen7b':q,'phi35':p})['strategy_fields_add_value']


def test_matched_fixed_spends_a_call_budget_on_more_cheaper_actions():
    e=module();f1=np.zeros((1,3,6));f1[:,:,1]=.6;f1[:,:,4]=1.
    choices=np.array([[1,0,0]]);risk=np.array([[3.,2.,1.]])
    value,action,per_input=e.matched_fixed(f1,choices,risk)
    assert action==4
    np.testing.assert_allclose(per_input,[[1.,1.,0.]])
    assert value==pytest.approx(2/3)


def test_hindsight_fixed_can_leave_budget_unused_to_avoid_harm():
    e=module();f1=np.full((1,3,6),.5);f1[:,:,1]=.6;f1[:,:,4]=[1.,0.,0.]
    value,action,per_input=e.matched_fixed(f1,np.array([[1,0,0]]),np.array([[3.,2.,1.]]))
    assert action==4
    np.testing.assert_allclose(per_input,[[1.,.5,.5]])
    assert value==pytest.approx(2/3)
