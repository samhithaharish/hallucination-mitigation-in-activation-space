import importlib
import joblib
import numpy as np
import pytest


def module():
    try:return importlib.import_module('experiments.recovery_scale.models')
    except ModuleNotFoundError:pytest.fail('Expanded supervised recovery models are missing')


def data(n=40):
    rng=np.random.default_rng(9)
    return rng.normal(size=(n,2,2,8)).astype(np.float32),rng.normal(size=(n,3)),[
        f'Evidence: sample {i%5} red blue\nQuestion: What item {i}?\nResponse:\nInitial answer: {i%3}' for i in range(n)]


def test_encoder_is_train_only_and_raw_features_restore(tmp_path):
    m=module();s,c,t=data()
    encoder=m.Encoder(components=4,text_components=5,projection_components=6,semantic=False).fit(s[:30],c[:30],t[:30])
    before=encoder.pcas[0].components_.copy()
    x=encoder.transform(s[30:],c[30:],t[30:])
    assert x['activation'].shape==(10,8)
    assert x['random'].shape==(10,6)
    assert np.array_equal(before,encoder.pcas[0].components_)
    assert 'unseenword' not in encoder.vectorizer.vocabulary_
    encoder.transform(s[30:]+1e4,c[30:],['unseenword']*10)
    assert 'unseenword' not in encoder.vectorizer.vocabulary_
    joblib.dump(encoder,tmp_path/'encoder.joblib')
    restored=joblib.load(tmp_path/'encoder.joblib').transform(s[30:],c[30:],t[30:])
    for key in x:np.testing.assert_allclose(restored[key],x[key],atol=1e-6)


def test_semantic_input_keeps_question_and_answer_ahead_of_long_evidence():
    m=module()
    text='Evidence: '+('distractor '*1000)+'\nQuestion: Where is Paris?\nResponse:\nInitial answer: France'
    view=m.semantic_view(text)
    assert 'Where is Paris?' in view[:150] and 'France' in view[:150]


def test_supervised_gain_model_and_degenerate_correctness_gate_are_finite(tmp_path):
    m=module();rng=np.random.default_rng(1);x=rng.normal(size=(100,4))
    y=np.column_stack([.2*x[:,0],-.2*x[:,0]])
    fit=m.GainModel('ridge',.01,1).fit(x[:80],y[:80])
    pred=fit.predict(x[80:]);assert np.mean((pred[:,1:]-y[80:])**2)<1e-4
    assert np.all(pred[:,0]==0)
    f1=np.column_stack([np.ones(100),np.zeros(100),np.ones(100)])
    gate=m.CorrectnessGate(1).fit(x,f1,f1)
    assert np.isfinite(gate.predict(x)).all()
    joblib.dump(fit,tmp_path/'fit.joblib')
    np.testing.assert_array_equal(pred,joblib.load(tmp_path/'fit.joblib').predict(x[80:]))
