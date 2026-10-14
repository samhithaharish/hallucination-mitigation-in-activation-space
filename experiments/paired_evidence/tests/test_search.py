import json
from pathlib import Path
import pytest


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


def test_queued_search_rejects_source_changes_before_any_git_mutation(monkeypatch):
    from experiments.paired_evidence import search as s
    monkeypatch.setenv('PYTHONHASHSEED','0')
    monkeypatch.setattr(s,'LOADED_SOURCE_HASH','stale queued source',raising=False)
    def forbidden(args):pytest.fail('Git or fitting ran despite stale queued source')
    monkeypatch.setattr(s,'command',forbidden)
    with pytest.raises(ValueError,match='source changed'):
        s.search()


def test_search_restarts_from_published_winner_and_retains_valid_git_history(monkeypatch,tmp_path):
    import subprocess
    from experiments.paired_evidence import search as s
    config=tmp_path/'predictor.json'
    winner=dict(kind='ridge',alpha=100.,features='state')
    config.write_text(json.dumps(winner,sort_keys=True)+'\n')
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=tmp_path,text=True).strip()
    git('init','-q')
    git('config','user.name','Test User');git('config','user.email','test@example.invalid')
    git('add','predictor.json');git('commit','-qm','Initial published winner')
    # Generated experiment artifacts must not dirty the real temporary checkout.
    (tmp_path/'.git/info/exclude').write_text('autoresearch/\nruns/\n')
    monkeypatch.setenv('PYTHONHASHSEED','0')
    monkeypatch.setattr(s,'REPO',tmp_path);monkeypatch.setattr(s,'OUT',tmp_path/'runs')
    monkeypatch.setattr(s,'CONFIG',config)
    real_command=s.command
    monkeypatch.setattr(s,'command',lambda args: '' if args[:3]==[s.sys.executable,'-m','pytest'] else real_command(args))
    seen=[]
    def verify(trial):
        current=json.loads(config.read_text());seen.append(current)
        if trial=='trial_00':assert current==dict(kind='mean',alpha=0.,features='state')
        value=.1 if current==winner else (1. if current['kind']=='mean' else .8)
        for key in s.MODELS:
            p=s.OUT/key/'fitted'/f'{trial}.joblib';p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(trial.encode());p.with_suffix('.json').write_text(json.dumps(dict(config=current)))
        return value,{key:value for key in s.MODELS}
    monkeypatch.setattr(s,'verify',verify)
    s.search()
    assert len(seen)==13
    assert json.loads(config.read_text())==winner
    assert git('status','--porcelain')==''
    summary=json.loads((tmp_path/'autoresearch/loop-261013-paired/summary.json').read_text())
    assert summary['selected_trial']=='trial_07' and summary['initial_metric']==1.
