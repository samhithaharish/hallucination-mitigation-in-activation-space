import numpy as np
from experiments.retrieval_budget.evaluate import missing_margin,compare


def test_missing_margin_ranks_the_same_three_way_decision():
    z=np.array([[0.,0.],[1.,0.],[0.,1.],[.8,.7],[.4,.1]])
    margin=missing_margin(z)
    np.testing.assert_array_equal(margin>0,[False,True,False,True,False])


def test_budget_control_can_choose_fewer_calls_and_a_cheaper_action():
    # Two active two-call retrievals give a four-call budget. A one-call sham
    # fixes four cases, while the two-call action helps one and harms another.
    f1=np.full((4,1,6),.5);f1[:,0,4]=1.;f1[:,0,1]=[1.,0.,.5,.5]
    choices=np.array([[1],[1],[0],[0]])
    risk=np.array([[4.],[3.],[2.],[1.]])
    result=compare(f1,choices,risk,risk,risk,draws=40)
    assert result['vs_fixed_public_gate']['mean']==-.5
    assert result['vs_fixed_public_gate']['hindsight_action']=='sham'


def test_registration_detects_changed_imported_scoring_helper(monkeypatch,tmp_path):
    import pytest
    from experiments.retrieval_budget import evaluate as module
    helper=tmp_path/'experiments/paired_evidence/evaluate.py'
    helper.parent.mkdir(parents=True);helper.write_text('original statistic')
    monkeypatch.setattr(module,'REPO',tmp_path)
    expected={'experiments/paired_evidence/evaluate.py':module.file_digest(helper)}
    module.check_sources(expected)
    helper.write_text('changed statistic')
    with pytest.raises(ValueError,match='Changed registered source'):
        module.check_sources(expected)
