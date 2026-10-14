"""Fresh base-question splits; every controlled version stays with its base."""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from experiments.paths import REPO, RUNS, CORPUS
from experiments.recovery_help.data import dump, file_digest
from experiments.recovery_help.task import normalize, tokens
from .task import build_case, valid_support

OUT=RUNS/'paired_evidence'
COUNTS={'calibration':32,'pilot':200,'train':400,'validation':100,'test':800,'natural':200}
SEED=20261013


def prepare():
    manifest_path=OUT/'data_manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
        for name,value in m['hashes'].items():
            if file_digest(OUT/'data'/f'{name}.json')!=value:
                raise ValueError('Changed paired cohort: '+name)
        return m
    rows=[r for p in sorted(CORPUS.glob('train-*.parquet')) for r in pq.read_table(p).to_pylist()]
    if not rows:raise ValueError('No pinned HotpotQA training corpus found')
    excluded=set()
    for path in [REPO/'results'/n/'cohorts.json' for n in ('recovery_pilot','recovery_scale','activation_drift')]:
        m=json.loads(path.read_text());excluded.update(i for ids in m['ids'].values() for i in ids)
    excluded_questions={normalize(r['question']) for r in rows if r['id'] in excluded}
    eligible=[r for r in rows if r['id'] not in excluded and normalize(r['question']) not in excluded_questions
              and len(set(r['supporting_facts']['title']))==2 and len(set(r['context']['title']))==10
              and valid_support(r) and r['level']!='hard' and len(r['answer'].split())<=12]
    rng=np.random.default_rng(SEED);rng.shuffle(eligible)
    split_raw={};used=set();seen_questions=set();embargo=set()
    for name in ('test','validation','natural','pilot','calibration','train'):
        selected=[]
        for row in eligible:
            titles=set(row['supporting_facts']['title']);q=normalize(row['question'])
            if row['id'] in used or q in seen_questions or titles&embargo:continue
            if not titles.issubset(row['context']['title']):continue
            selected.append(row);used.add(row['id']);seen_questions.add(q)
            if len(selected)==COUNTS[name]:break
        if len(selected)!=COUNTS[name]:raise ValueError('Insufficient eligible '+name)
        split_raw[name]=selected
        embargo.update(t for r in selected for t in r['supporting_facts']['title'])
    # Filler passages come from a disjoint background pool, never from test bases.
    background=[r for r in eligible if r['id'] not in used]
    filler_pool=[];seen_titles=set()
    for r in background:
        for title,ss in zip(r['context']['title'],r['context']['sentences']):
            words=' '.join(ss).split()
            if title not in seen_titles and title not in embargo and len(words)>=180:
                filler_pool.append({'title':title,'text':' '.join(words[:180])});seen_titles.add(title)
        if len(filler_pool)>=5000:break
    if len(filler_pool)<100:raise ValueError('Insufficient independent filler passages')
    ids={};hashes={}
    for split,raw in split_raw.items():
        cases=[]
        for j,r in enumerate(raw):
            q=set(tokens(r['question'])); blocked=set(r['context']['title'])
            choices=rng.permutation(len(filler_pool));fillers=[]
            for k in choices:
                p=filler_pool[int(k)]
                if p['title'] in blocked or len(q&set(tokens(p['title'])))>1:continue
                fillers.append(p);blocked.add(p['title'])
                if len(fillers)==10:break
            cases.append(build_case(r,fillers,SEED+int(r['id'][:8],16)))
        dump(OUT/'data'/f'{split}.json',cases)
        ids[split]=[r['id'] for r in cases];hashes[split]=file_digest(OUT/'data'/f'{split}.json')
        if split=='natural':dump(OUT/'data'/'natural_original.json',raw)
    manifest={'seed':SEED,'counts':COUNTS,'ids':ids,'hashes':hashes,
              'excluded_previous_ids':len(excluded),'support_title_disjoint':True,
              'source_hashes':{p.name:file_digest(p) for p in sorted(CORPUS.glob('train-*.parquet'))}}
    dump(manifest_path,manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('ids','source_hashes')},indent=2))
    return manifest


if __name__=='__main__':prepare()
