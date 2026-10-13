"""Post-collection checks without choosing or tuning any policy."""
import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer, GenerationConfig

from experiments.activation_drift import pilot
from experiments.activation_drift import analyze


def run(key):
    from experiments.recovery_help.backend import chat
    data,actions=pilot.helpers()
    analyze.validate_data()
    tok=AutoTokenizer.from_pretrained(data.MODEL_PATHS[key],local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token=tok.eos_token
    config=GenerationConfig.from_pretrained(data.MODEL_PATHS[key],local_files_only=True)
    eos=config.eos_token_id
    stops=set(([eos] if isinstance(eos,int) else list(eos or []))+[tok.pad_token_id])
    manifest=pilot.prepare(); identity=pilot.sha(data.OUT/key/'model_hashes.json')
    counts={}; max_error=0.; total_generated=0
    for split in ('train','validation','test'):
        raw=json.loads((pilot.ROOT/'data'/f'{split}.json').read_text())
        complete=json.loads((pilot.ROOT/key/split/'complete.json').read_text())
        assert complete['n']==len(raw)
        start=0
        for path in sorted((pilot.ROOT/key/split).glob('batch_*.json')):
            payload=json.loads(path.read_text()); rows=payload['rows']
            f=payload['fingerprint']
            assert f['source']==pilot.sha(pilot.CODE/'pilot.py')
            assert f['data']==manifest['hashes'][split]
            assert f['model_identity']==identity
            assert payload['states_sha256']==pilot.sha(path.with_suffix('.npz'))
            states=np.load(path.with_suffix('.npz'))['states']
            assert states.shape[:4]==(len(rows),len(pilot.ACTIONS),6,2)
            assert np.isfinite(states).all()
            chunk=raw[start:start+len(rows)]
            assert [r['id'] for r in rows]==[r['id'] for r in chunk]
            for row,example in zip(rows,chunk):
                pub=actions.public_example(example)
                baseline=actions.extract_answer(row['actions'][0]['text'])
                anchor=baseline or '(no valid initial answer)'
                assert [r['name'] for r in row['actions']]==list(pilot.ACTIONS)
                for a,item in zip(pilot.ACTIONS,row['actions']):
                    prompt=chat(tok,actions.prompt(pub,a,anchor if a!='keep' else ''))
                    assert item['prompt_ids']==tok.encode(prompt,add_special_tokens=False)
                    assert item['input_tokens']==len(item['prompt_ids'])
                    n=next((j for j,t in enumerate(item['output_ids']) if t in stops),len(item['output_ids']))
                    assert item['output_tokens']==n
                    assert item['at_cap']==(n==actions.cap_for(a))
                    assert n<=actions.cap_for(a)
                    answer=actions.extract_answer(item['text'],a in ('reason','both'))
                    assert item['answer']==answer
                    decoded=tok.decode(item['output_ids'],skip_special_tokens=True)
                    decoded_answer=actions.extract_answer(decoded,a in ('reason','both'))
                    assert actions.normalize(decoded_answer)==actions.normalize(answer)
                    # The collector stores no success labels, including calibration.
                    assert not {'gold','score','f1','em','correct'}&set(item)
                total_generated+=len(row['actions'])
            start+=len(rows)
        assert start==len(raw)
        counts[split]=start
    before=json.loads((pilot.ROOT/key/'identity_before.json').read_text())
    after=json.loads((pilot.ROOT/key/'identity_after.json').read_text())
    assert before['manifest_sha256']==after['manifest_sha256']==identity
    evaluation=json.loads((pilot.ROOT/key/'evaluation.json').read_text())
    assert evaluation['audit']['n']==counts['test']
    record={'model':key,'counts':counts,'generation_calls_excluding_smoke':total_generated,
        'all_prompts_reconstructed':True,'answers_match_exact_token_replays':True,
        'collector_contains_no_success_labels':True,'all_state_arrays_finite_and_hashed':True,
        'model_identity_checks_match':True,'source_and_data_fingerprints_match':True,
        'test_policy_reload':evaluation['audit']}
    pilot.save(pilot.ROOT/key/'audit.json',record)
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=('qwen7b','phi35'),required=True)
    run(p.parse_args().model)
