import importlib
import numpy as np
import pytest


def module():
    try:return importlib.import_module('experiments.recovery_scale.audit')
    except ModuleNotFoundError:pytest.fail('Raw study audit is missing')


def test_article_components_follow_shared_support_transitively():
    a=module()
    rows=[{'supporting_facts':{'title':t}} for t in [['A','B'],['B','C'],['D'],['C','E'],['F']]]
    groups=a.article_groups(rows)
    assert groups[0]==groups[1]==groups[3]
    assert len(set(groups))==3


def test_cluster_bootstrap_weights_questions_and_constant_signal_exactly():
    a=module()
    ci=a.cluster_interval(np.array([0.,0.,0.,1.]),np.array([0,0,0,1]),repeats=500)
    assert ci['mean']==.25
    assert ci['groups']==2
    assert ci['low']==0. and ci['high']==1.
    constant=a.cluster_interval(np.ones(4)*.3,np.array([0,0,1,2]),repeats=100)
    assert constant['low']==pytest.approx(.3) and constant['high']==pytest.approx(.3)


def test_independent_score_audit_rejects_inflated_recovery_score():
    a=module()
    record={'text':'<final>Paris</final>','answer':'Paris','score':{'em':1.,'f1':1.},'stages':[]}
    # Known mismatch: test the audit boundary without a downloaded scorer.
    # Full experiment audits still load and checksum the official evaluator.
    def mismatch_f1(answer, gold):
        assert (answer, gold) == ('Paris', 'London')
        return 0., 0., 0.
    scorer = mismatch_f1, lambda answer, gold: answer == gold
    with pytest.raises(AssertionError):
        a.check_score('retrieve', record, 'London', scorer=scorer)
    record['score'] = {'em': 0., 'f1': 0.}
    a.check_score('retrieve', record, 'London', scorer=scorer)
