import numpy as np
import pytest


def test_question_specific_reference_preserves_content_and_rejects_deletion():
    import importlib
    try:f=importlib.import_module('experiments.error_field.reference')
    except ImportError:pytest.fail('Paired clean-reference comparison is not implemented')
    rng=np.random.default_rng(9);n=90
    clean=np.stack([3+rng.random(n),np.zeros(n),np.zeros(n)],axis=1)
    x=np.repeat(clean[:,None,:],3,axis=1);x[:,1,1]=1.;x[:,2,2]=1.
    target=np.repeat(clean[:,None,:],3,axis=1)
    train=x[:60].reshape(-1,3);val=x[60:].reshape(-1,3)
    ref=f.Reference().fit(train,target[:60].reshape(-1,3),val,target[60:].reshape(-1,3))
    assert np.mean((ref.predict(val)-target[60:].reshape(-1,3))**2)<.001
    delta=np.zeros((n,3,6,3));delta[:,:,1,1]=-x[:,:,1];delta[:,:,2,2]=-x[:,:,2]
    delta[:,:,5]=-x
    dynamics=f.StateDelta(alpha=.01).fit(train,delta[:60].reshape(-1,6,3),val,delta[60:].reshape(-1,6,3))
    scores=dynamics.scores(val,val-ref.predict(val)).reshape(30,3,6)
    assert (scores[:,1,:4].argmax(-1)==1).all()
    assert (scores[:,2,:4].argmax(-1)==2).all()
    assert (scores[:,1:,5]<0).all()
    with pytest.raises(TypeError):ref.fit(train,train,val,val,action_success=np.ones(len(train)))
