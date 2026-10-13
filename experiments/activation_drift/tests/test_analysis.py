import numpy as np
from experiments.activation_drift import analyze


def test_risk_permutation_preserves_bin_action_mix_and_expected_quality():
    # A global shuffle would confound routing with the generic risk ordering.
    score = np.array([[0.,1.], [1.,0.], [0.,1.], [1.,0.]])
    risk = np.array([0.,1.,2.,3.]); choices = np.array([1,0,1,1])
    expected, bins = analyze.risk_matched_expectation(score,choices,risk,bins=2)
    np.testing.assert_allclose(expected,[.5,.5,1.,0.])
    assert bins.tolist()==[0,0,1,1]


def test_article_components_join_transitive_shared_supports():
    # Resampling questions independently misses dependence through shared titles.
    rows = [{'supporting_facts':{'title':x}} for x in [['a'],['a','b'],['b'],['c']]]
    labels = analyze.article_components(rows)
    assert labels[0]==labels[1]==labels[2]
    assert labels[0]!=labels[3]


def test_bootstrap_constant_difference_has_zero_uncertainty():
    result = analyze.interval(np.full(20,.125),draws=30)
    np.testing.assert_allclose([result['mean'],*result['ci95']],[.125,.125,.125])


def test_keep_has_no_extra_calls_and_harm_counts_only_initially_correct():
    f1=np.array([[1.,0.], [0.,1.], [.5,0.]])
    em=np.array([[1.,0.], [0.,1.], [0.,0.]])
    tokens=np.array([[2,10],[3,20],[4,30]])
    result=analyze.policy_stats(f1,em,np.array([1,1,0]),tokens)
    assert result['rescued']==1 and result['harmed']==1
    assert result['extra_calls']==2/3
    assert result['extra_output_tokens']==10.


def test_routing_ci_recomputes_empirical_action_mix_in_every_resample():
    # With constant quality per action, every assignment with the same action
    # mix has identical value. Treating estimated frequencies as fixed creates
    # spurious uncertainty and can distort the primary routing comparison.
    f1=np.tile([0.,1.],(4,1))
    result=analyze.risk_matched_interval(f1,np.array([0,1,1,1]),np.arange(4),bins=1,draws=100)
    np.testing.assert_allclose([result['mean'],*result['ci95']],[0.,0.,0.],atol=1e-15)
