import hashlib
import importlib
import json

import pytest


def setup_module_api():
    try:
        return importlib.import_module('experiments.prepare')
    except ImportError:
        pytest.fail('Portable experiment setup has not been added')


def cohort(tmp_path):
    rows = [{'id': 'first', 'answer': 'one'}, {'id': 'second', 'answer': 'two'}]
    payload = json.dumps(rows[::-1], indent=2, allow_nan=False).encode()
    manifest = {'ids': {'test': ['second', 'first']},
                'split_hashes': {'test': hashlib.sha256(payload).hexdigest()}}
    path = tmp_path / 'cohorts.json'
    path.write_text(json.dumps(manifest))
    return path, rows, payload


def test_restore_preserves_frozen_order_bytes_and_resumes(tmp_path):
    api = setup_module_api()
    manifest, rows, expected = cohort(tmp_path)
    destination = tmp_path / 'run'
    for _ in range(2):
        api.restore_cohorts(manifest, destination, {r['id']: r for r in rows})
        assert (destination / 'data/test.json').read_bytes() == expected
        assert (destination / 'data_manifest.json').read_bytes() == manifest.read_bytes()


def test_restore_rejects_changed_corpus_before_writing(tmp_path):
    api = setup_module_api()
    manifest, rows, _ = cohort(tmp_path)
    rows[0]['answer'] = 'changed'
    destination = tmp_path / 'run'
    with pytest.raises(ValueError, match='checksum'):
        api.restore_cohorts(manifest, destination, {r['id']: r for r in rows})
    assert not destination.exists()


def test_setup_does_not_overwrite_an_existing_run(tmp_path):
    api = setup_module_api()
    path = tmp_path / 'frozen.json'
    path.write_bytes(b'original')
    with pytest.raises(ValueError, match='overwrite'):
        api.write_checked(path, b'changed')
    assert path.read_bytes() == b'original'
