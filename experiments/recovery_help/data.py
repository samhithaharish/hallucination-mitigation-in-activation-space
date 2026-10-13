"""Pinned local corpus, deterministic splits, and artifact helpers."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from .task import normalize

SEED = 20261008
from experiments.paths import PILOT_RUN as OUT, CORPUS, MODEL_PATHS


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def eligible_train(raw, heldout_support_titles, heldout_normalized_questions):
    return (not set(raw['supporting_facts']['title']) & heldout_support_titles
            and normalize(raw['question']) not in heldout_normalized_questions)


def prepare_data():
    manifest_path = OUT / 'data_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, sha in manifest['split_hashes'].items():
            if file_digest(OUT / 'data' / f'{name}.json') != sha:
                raise RuntimeError('Changed frozen split: ' + name)
        return manifest
    rng = np.random.default_rng(SEED)
    devfile = CORPUS / 'validation-00000-of-00001.parquet'
    trainfiles = sorted(CORPUS.glob('train-*.parquet'))
    dev = pq.read_table(devfile).to_pylist()
    seen_ids, seen_questions = set(), set()
    def take(rows, n, heldout_titles=None):
        selected = []
        for j in rng.permutation(len(rows)):
            row = rows[int(j)]
            if row['id'] in seen_ids or normalize(row['question']) in seen_questions:
                continue
            if heldout_titles and not eligible_train(row, heldout_titles, seen_questions):
                continue
            selected.append(row)
            seen_ids.add(row['id']); seen_questions.add(normalize(row['question']))
            if len(selected) == n:
                return selected
        raise RuntimeError('Insufficient eligible examples')
    heldout = take(dev, 900)
    titles = {t for r in heldout for t in r['supporting_facts']['title']}
    train = [row for p in trainfiles for row in pq.read_table(p).to_pylist()]
    development = take(train, 1596, titles)
    splits = {'calibration': development[:96], 'train': development[96:1296],
              'validation': development[1296:], 'test': heldout[:600], 'transfer': heldout[600:]}
    for name, rows in splits.items():
        dump(OUT / 'data' / f'{name}.json', rows)
    manifest = {'seed': SEED, 'source_hashes': {p.name: file_digest(p) for p in [*trainfiles, devfile]},
                'split_hashes': {name: file_digest(OUT / 'data' / f'{name}.json') for name in splits},
                'ids': {name: [r['id'] for r in rows] for name, rows in splits.items()},
                'counts': {name: len(rows) for name, rows in splits.items()},
                'heldout_support_title_count': len(titles)}
    dump(manifest_path, manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps({k: v for k, v in prepare_data().items() if k != 'ids'}, indent=2))
