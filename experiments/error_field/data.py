"""Reuse frozen development executions; reserve completely new test questions."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import pyarrow.parquet as pq

from experiments.paths import REPO,CORPUS
from experiments.paired_evidence.data import OUT
from experiments.paired_evidence.task import build_case,valid_support,CONDITIONS
from experiments.recovery_help.data import dump,file_digest
from experiments.recovery_help.task import normalize,tokens


def prepare(source):
    source=Path(source).resolve()
    if (OUT/'data_manifest.json').exists():
        raise ValueError('Preserve existing cohorts; use an empty run root')
    previous=json.loads((source/'data_manifest.json').read_text())
    old_cases=[r for split in previous['ids'] for r in json.loads((source/'data'/f'{split}.json').read_text())]
    old_titles={p['title'] for r in old_cases for p in r['versions']['distractor']['candidates']}
    excluded={i for ids in previous['ids'].values() for i in ids}
    for p in [REPO/'results'/n/'cohorts.json' for n in ('recovery_pilot','recovery_scale','activation_drift')]+[
              REPO/'results/paired_evidence/qualification_cohorts.json']:
        m=json.loads(p.read_text());excluded.update(i for ids in m['ids'].values() for i in ids)
    rows=[r for p in sorted(CORPUS.glob('train-*.parquet')) for r in pq.read_table(p).to_pylist()]
    excluded_questions={normalize(r['question']) for r in rows if r['id'] in excluded}
    eligible=[r for r in rows if r['id'] not in excluded and normalize(r['question']) not in excluded_questions
        and len(set(r['supporting_facts']['title']))==2 and len(set(r['context']['title']))==10
        and not (set(r['supporting_facts']['title'])&old_titles)
        and valid_support(r) and r['level']!='hard' and len(r['answer'].split())<=12]
    rng=np.random.default_rng(20261015);rng.shuffle(eligible)
    # Required sentences are kept intact; ensure the 180-word fillers can match them.
    selected=[];seen=set()
    for r in eligible:
        required={t:set() for t in r['supporting_facts']['title']}
        for t,i in zip(r['supporting_facts']['title'],r['supporting_facts']['sent_id']):required[t].add(i)
        paragraphs=dict(zip(r['context']['title'],r['context']['sentences']))
        if max(sum(len(paragraphs[t][i].split()) for i in ids) for t,ids in required.items())>180:continue
        q=normalize(r['question'])
        if q in seen:continue
        selected.append(r);seen.add(q)
        if len(selected)==800:break
    if len(selected)!=800:raise ValueError('Insufficient new support-disjoint test questions')
    test_ids={r['id'] for r in selected}
    blocked=old_titles|{t for r in selected for t in r['context']['title']}
    fillers=[]
    for r in rows:
        if r['id'] in test_ids:continue
        for title,ss in zip(r['context']['title'],r['context']['sentences']):
            words=' '.join(ss).split()
            if title not in blocked and len(words)>=180:
                fillers.append({'title':title,'text':' '.join(words[:180])});blocked.add(title)
        if len(fillers)>=5000:break
    if len(fillers)<100:raise ValueError('Insufficient disjoint background fillers')
    cases=[]
    for r in selected:
        q=set(tokens(r['question']));chosen=[]
        for j in rng.permutation(len(fillers)):
            f=fillers[int(j)]
            if len(q&set(tokens(f['title'])))>1:continue
            chosen.append(f)
            if len(chosen)==10:break
        case=build_case(r,chosen,20261015+int(r['id'][:8],16))
        lengths=[sum(len(p['text'].split()) for p in case['versions'][c]['current']) for c in CONDITIONS]
        if len(set(lengths))!=1:raise ValueError('Unmatched passage word counts')
        cases.append(case)
    ids={};hashes={}
    for split in ('train','validation'):
        target=OUT/'data'/f'{split}.json';target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source/'data'/f'{split}.json',target)
        ids[split]=previous['ids'][split];hashes[split]=file_digest(target)
    dump(OUT/'data/test.json',cases);ids['test']=[r['id'] for r in cases]
    hashes['test']=file_digest(OUT/'data/test.json')
    for key in ('qwen7b','phi35'):
        target=OUT/key;target.mkdir(parents=True,exist_ok=True)
        for split in ('train','validation'):
            (target/split).symlink_to(source/key/split,target_is_directory=True)
    manifest=dict(seed=20261015,ids=ids,hashes=hashes,counts={s:len(v) for s,v in ids.items()},
        source_hashes=previous['source_hashes'],source_development=str(source),
        source_manifest_sha256=file_digest(source/'data_manifest.json'),
        excluded_previous_ids=len(excluded),test_support_titles_absent_from_previous_candidate_pools=True,
        test_fillers_absent_from_previous_candidate_pools=True,
        previous_pool_scope='source v3 paired cohorts; earlier cohorts excluded by question IDs only',
        candidate_pool='same shuffled 20-article union in all conditions')
    dump(OUT/'data_manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('ids','source_hashes')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);a=p.parse_args()
    prepare(a.source)
