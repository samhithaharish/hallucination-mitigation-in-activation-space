import numpy as np
import pytest


def module():
    import importlib
    try:return importlib.import_module('experiments.error_field.fit')
    except ImportError:pytest.fail('Error-field fitting is not implemented')


def test_truth_labels_score_only_the_original_answer(monkeypatch):
    f=module();seen=[]
    def score(answer,gold):
        assert answer!='strategy answer must stay unscored'
        seen.append(answer)
        return {'em':float(answer==gold)}
    monkeypatch.setattr(f,'score_answer',score)
    versions=[{'actions':[{'name':'keep','answer':a},{'name':'retrieve','answer':'strategy answer must stay unscored'}]}
              for a in ('yes','no','yes')]
    labels=f.baseline_targets([{'id':'x','versions':versions}],[{'id':'x','answer':'yes'}])
    np.testing.assert_array_equal(labels,[[0.,1.,0.]])
    assert seen==['yes','no','yes']


def test_delta_forecast_selects_response_and_rejects_noisy_apparent_gain():
    f=module();rng=np.random.default_rng(8);x=rng.normal(size=(120,3))
    delta=np.zeros((120,6));delta[:,1]=-x[:,0];delta[:,2]=x[:,0]
    delta[:,3]=-.01+rng.normal(size=120)
    model=f.DeltaModel(kind='ridge',alpha=.01).fit(x[:90],delta[:90],x[90:],delta[90:])
    scores=model.scores(x[90:]);choices=scores[:,:4].argmax(1)
    assert np.mean(choices==np.where(x[90:,0]>0,1,2))>.95
    assert (scores[:,3]<0).all()
    with pytest.raises(TypeError):model.fit(x,delta,x,delta,action_success=np.ones_like(delta))


def test_negative_control_cannot_select_or_change_primary_metric():
    f=module();x=np.arange(50)[:,None]/50
    d=np.zeros((50,6));d[:,1]=-.2;d[:,5]=-5.
    one=f.DeltaModel(kind='mean').fit(x,d,x,d)
    changed=d.copy();changed[:,5]=100.
    two=f.DeltaModel(kind='mean').fit(x,changed,x,changed)
    assert one.metric==two.metric
    assert (one.scores(x)[:,:4].argmax(1)==1).all()
    assert (one.scores(x).argmax(1)==5).all()


def test_router_is_single_input_and_batch_permutation_equivariant():
    from types import SimpleNamespace
    f=module();rng=np.random.default_rng(3);states=rng.normal(size=(8,3,1,3))
    coordinate=SimpleNamespace(transform=lambda s:(s[...,0,:],s[...,0,:2]))
    probe=SimpleNamespace(decision_function=lambda x:x[:,2])
    x,z=coordinate.transform(states);q=probe.decision_function(x.reshape(-1,3))
    target=np.zeros((24,6));target[:,1]=-x[...,0].ravel();target[:,2]=x[...,0].ravel()
    router=f.RiskRouter();router.coordinates=coordinate;router.probe=probe
    router.field=f.DeltaModel(alpha=.01).fit(f.design(coordinate,x,z,q,'state'),target,
        f.design(coordinate,x,z,q,'state'),target)
    router.previous=SimpleNamespace(predict=lambda s:{'choices':np.zeros(s.shape[:2],int)})
    result=router.predict(states);order=[2,0,1];permuted=router.predict(states[:,order])
    for name in result:np.testing.assert_allclose(permuted[name],result[name][:,order],atol=1e-10)
    assert router.predict(states[:1,:1])['choices'][0,0]==result['choices'][0,0]
