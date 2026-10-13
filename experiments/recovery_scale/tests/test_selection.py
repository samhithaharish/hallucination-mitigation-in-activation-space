import importlib
import numpy as np
import pytest


def module():
    try:return importlib.import_module('experiments.recovery_scale.train')
    except ModuleNotFoundError:pytest.fail('Frozen candidate selection is missing')


def test_fixed_policy_can_spend_fractional_batch_budget_without_label_tie_breaks():
    m=module();rows=[{'id':str(i)} for i in range(100)]
    p=m.baselines(np.zeros(10),rows)['fixed_retry']
    from experiments.recovery_help.policy import allocate
    costs=np.tile([0,1,1,1,1,1,1,2,2,3],(100,1))
    choice,_=allocate(p,costs,50)
    assert (choice==1).sum()==50
    np.testing.assert_array_equal(p,m.baselines(np.zeros(10),rows)['fixed_retry'])


def test_menu_mask_cannot_choose_unavailable_new_action():
    m=module();p=np.zeros((2,10));p[:,1]=.1;p[:,9]=.9
    old=m.mask_menu(p,5)
    assert old.argmax(1).tolist()==[1,1]
    assert p.argmax(1).tolist()==[9,9]


def test_fit_never_reads_test_labels_and_frozen_artifacts_reject_drift(tmp_path,monkeypatch):
    import json
    m=module();rng=np.random.default_rng(17);access=[]
    def fake_load(key,split,limit=None):
        access.append(split)
        assert split in ('train','validation'), 'Fit accessed held-out outcomes'
        n=80 if split=='train' else 30
        states=rng.normal(size=(n,2,2,8)).astype(np.float32)
        em=rng.integers(2,size=(n,10)).astype(float)
        return {'rows':[{'id':f'{split}{i}'} for i in range(n)],'states':states,
            'metadata':rng.normal(size=(n,3)),'texts':[f'Question: color {i+20}?\nInitial answer: item {i+30}' for i in range(n)],
            'f1':em,'em':em,'calls':np.tile([0,1,1,1,1,1,1,2,2,3],(n,1)),
            'tokens':np.tile([0,100,200,300,400,200,150,900,1000,1200],(n,1))}
    monkeypatch.setattr(m,'OUT',tmp_path);monkeypatch.setattr(m,'load_split',fake_load)
    monkeypatch.setattr(m,'GRID',[('ridge',1.,0)]);monkeypatch.setattr(m,'GATE_CS',[.01])
    monkeypatch.setattr(m,'semantic_for',lambda key,split,data:np.ones((len(data['rows']),4),np.float32))
    root=tmp_path/'unit'/'main';root.mkdir(parents=True);(root/'protocol.json').write_text('{}')
    (tmp_path/'data_manifest.json').write_text('{}')
    m.fit('unit',80)
    selection=m.verify(tmp_path/'unit'/'prediction_80','unit')
    assert access==['train','validation']
    assert selection['train_size']==80 and set(selection['choices'])=={'5','10'}
    (tmp_path/'unit'/'prediction_80'/'models.joblib').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='artifact'):m.verify(tmp_path/'unit'/'prediction_80','unit')
