import hashlib
import json
import pytest


def test_identity_guard_rejects_replaced_tokenizer_and_weight_files(tmp_path):
    from experiments.recovery_scale.verified_collect import verify_identity
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    for name in ('config.json','tokenizer_config.json','model.safetensors'):
        (snapshot/name).write_text('original')
    manifest={p.name:{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
              for p in snapshot.iterdir()}
    path=tmp_path/'model_hashes.json';path.write_text(json.dumps(manifest))
    assert verify_identity(snapshot,path)['files']==3
    for name in ('tokenizer_config.json','model.safetensors'):
        (snapshot/name).write_text('replaced')
        with pytest.raises(ValueError,match='identity changed'):verify_identity(snapshot,path)
        (snapshot/name).write_text('original')
    (snapshot/'generation_config.json').write_text('new settings')
    with pytest.raises(ValueError,match='file set changed'):verify_identity(snapshot,path)


def test_identity_guard_detects_an_added_nested_chat_template(tmp_path):
    from experiments.recovery_scale.verified_collect import verify_identity
    snapshot=tmp_path/'snapshot';snapshot.mkdir()
    config=snapshot/'config.json';config.write_text('{}')
    path=tmp_path/'model_hashes.json'
    path.write_text(json.dumps({'config.json':{'bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}}))
    (snapshot/'chat_templates').mkdir()
    (snapshot/'chat_templates'/'tool_use.jinja').write_text('changed prompt')
    with pytest.raises(ValueError,match='file set changed'):verify_identity(snapshot,path)


def test_verified_collection_does_not_start_after_identity_failure(tmp_path,monkeypatch):
    from experiments.recovery_scale import verified_collect as v
    calls=[]
    monkeypatch.setattr(v,'verify_identity',lambda *a: (_ for _ in ()).throw(ValueError('identity changed')))
    monkeypatch.setattr(v,'collect_run',lambda *a,**k:calls.append(a))
    with pytest.raises(ValueError,match='identity changed'):v.run('qwen7b',['test'],out=tmp_path)
    assert calls==[]
