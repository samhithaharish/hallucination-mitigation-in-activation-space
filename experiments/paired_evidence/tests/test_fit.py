import importlib
import numpy as np
import pytest


def module():
    try:return importlib.import_module('experiments.paired_evidence.fit')
    except ImportError:pytest.fail('Activation-pair fitting is not implemented')


def synthetic(n=80):
    rng=np.random.default_rng(1)
    h=rng.normal(size=(n,3,6,2,12)).astype(np.float32)
    h[:,:,:,:,:2]=0
    for c in range(3):
        h[:,c]=h[:,0]
    h[:,1,:,0,0]=1;h[:,2,:,0,1]=1
    # Action 1 fixes missing evidence; action 3 fixes distractor interference.
    h[:,1,1,0,0]=0;h[:,2,3,0,1]=0
    return h


def test_pair_coordinates_generalize_without_answer_or_reward_arguments():
    f=module();h=synthetic()
    c=f.Coordinates(components=2).fit(h[:60,:,0],h[60:,:,0])
    z=c.transform(h[60:,:,0])[1]
    expected=np.tile([[0,0],[1,0],[0,1]],(20,1,1))
    assert np.mean((z-expected)**2)<.01
    with pytest.raises(TypeError):c.fit(h[:60,:,0],h[60:,:,0],action_rewards=np.zeros((60,3)))


def test_policy_uses_initial_states_and_is_equivariant_to_condition_order():
    f=module();h=synthetic()
    c=f.Coordinates(components=2).fit(h[:60,:,0],h[60:,:,0])
    p=f.Directions(c,kind='ridge',alpha=.01,features='state').fit(h[:60],h[60:])
    result=p.predict(h[60:,:,0])
    # Input diagnosis routes only the independently supported missing-evidence remedy.
    np.testing.assert_array_equal(result['class_rule'],np.tile([0,1,0],(20,1)))
    assert (result['choices'][:,1]==1).all()
    assert (result['choices'][:,2]==3).all()
    order=[2,0,1]
    shuffled=p.predict(h[60:,order,0])
    np.testing.assert_array_equal(shuffled['choices'],result['choices'][:,order])
    np.testing.assert_allclose(shuffled['scores'],result['scores'][:,order],atol=1e-6)
