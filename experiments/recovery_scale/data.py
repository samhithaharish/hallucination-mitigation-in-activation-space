"""Fresh, fixed cohorts with matched type/difficulty and training title embargo."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from experiments.recovery_help.data import CORPUS, MODEL_PATHS, dump, digest, file_digest, OUT as OLD_OUT
from experiments.recovery_help.task import normalize

SEED = 20261009
from experiments.paths import SCALE_RUN as OUT
COUNTS = {'calibration': 128, 'train': 6000, 'validation': 1000, 'test': 2000, 'transfer': 1000}
STRATA = {('easy','bridge'): .16, ('easy','comparison'): .04,
          ('medium','bridge'): .48, ('medium','comparison'): .12,
          ('hard','bridge'): .16, ('hard','comparison'): .04}


def sample(rows, n, used_ids, used_questions, embargo, seed=SEED, strata=None):
    chosen, seen_ids, seen_questions = [], set(used_ids), set(used_questions)
    have = Counter()
    for j in np.random.default_rng(seed).permutation(len(rows)):
        r = rows[int(j)]; q = normalize(r['question']); key = (r['level'], r['type'])
        if r['id'] in seen_ids or q in seen_questions or set(r['supporting_facts']['title']) & embargo:
            continue
        if strata is not None and have[key] >= round(n*strata.get(key, 0)):
            continue
        chosen.append(r); seen_ids.add(r['id']); seen_questions.add(q); have[key] += 1
        if len(chosen) == n: return chosen
    raise ValueError(f'Insufficient eligible questions: {len(chosen)}/{n}; strata={dict(have)}')


def prepare():
    path = OUT/'data_manifest.json'
    if path.exists():
        m = json.loads(path.read_text())
        for name, sha in m['split_hashes'].items():
            if file_digest(OUT/'data'/f'{name}.json') != sha: raise ValueError('Frozen data changed: '+name)
        return m
    files = sorted(CORPUS.glob('train-*.parquet'))
    train = [r for p in files for r in pq.read_table(p).to_pylist()]
    devfile = CORPUS/'validation-00000-of-00001.parquet'
    dev = pq.read_table(devfile).to_pylist()
    oldrows = [r for p in sorted((OLD_OUT/'data').glob('*.json')) for r in json.loads(p.read_text())]
    ids = {r['id'] for r in oldrows}; questions = {normalize(r['question']) for r in oldrows}
    splits = {}; embargo = set()
    for i, name in enumerate(('transfer', 'test', 'validation', 'calibration', 'train')):
        rows = sample(dev if name == 'transfer' else train, COUNTS[name], ids, questions,
                      embargo if name in ('train', 'calibration') else set(), seed=SEED+i,
                      strata=STRATA if name in ('train','validation','test') else None)
        splits[name] = rows; ids.update(r['id'] for r in rows)
        questions.update(normalize(r['question']) for r in rows)
        if name in ('test','validation','transfer'):
            embargo.update(t for r in rows for t in r['supporting_facts']['title'])
        dump(OUT/'data'/f'{name}.json', rows)
    manifest = {'seed':SEED, 'counts':COUNTS, 'prior_excluded_count':len(oldrows),
        'source_hashes':{p.name:file_digest(p) for p in [*files,devfile]},
        'ids':{s:[r['id'] for r in rows] for s,rows in splits.items()},
        'strata':{s:dict(Counter(r['level']+'/'+r['type'] for r in rows)) for s,rows in splits.items()},
        'embargo_title_count':len(embargo), 'old_manifest_hash':file_digest(OLD_OUT/'data_manifest.json'),
        'split_hashes':{s:file_digest(OUT/'data'/f'{s}.json') for s in splits}}
    dump(path, manifest)
    return manifest


if __name__ == '__main__':
    print(json.dumps({k:v for k,v in prepare().items() if k!='ids'}, indent=2))
