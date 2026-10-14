import json
from pathlib import Path


def test_failed_trial_is_recorded_and_later_trials_finish(monkeypatch,tmp_path):
    from experiments.paired_evidence import search as s
    monkeypatch.setenv('PYTHONHASHSEED','0')
    monkeypatch.setattr(s,'REPO',tmp_path)
    monkeypatch.setattr(s,'OUT',tmp_path/'runs')
    config=tmp_path/'predictor.json';config.write_text('{}')
    monkeypatch.setattr(s,'CONFIG',config)
    monkeypatch.setattr(s,'command',lambda args: '' if args[:3]==['git','status','--porcelain'] else 'abc')

    def verify(trial):
        if trial=='trial_01':raise RuntimeError('transient fitting error')
        for key in s.MODELS:
            path=s.OUT/key/'fitted'/f'{trial}.joblib';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(b'frozen test object')
            path.with_suffix('.json').write_text(json.dumps({'config':{'kind':'mean'}}))
        value=1. if trial=='trial_00' else 1.1
        return value,{key:value for key in s.MODELS}

    monkeypatch.setattr(s,'verify',verify)
    s.search()
    root=tmp_path/'autoresearch/loop-261013-paired'
    results=json.loads((root/'trials.json').read_text())
    assert len(results)==13 and results[1]['status']=='crash'
    assert results[1]['metric'] is None
    assert results[-1]['iteration']==12
    summary=json.loads((root/'summary.json').read_text())
    assert summary['selected_trial']=='trial_00'
    assert all((s.OUT/key/'fitted/best.joblib').exists() for key in s.MODELS)
