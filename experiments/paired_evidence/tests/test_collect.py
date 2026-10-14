import importlib
import numpy as np
import pytest
from .test_task import raw,fillers


def test_choices_are_saved_before_any_mitigation_and_collection_has_no_success_labels():
    try:c=importlib.import_module('experiments.paired_evidence.collect')
    except ImportError:pytest.fail('Paired collector is not implemented')
    from experiments.paired_evidence.task import build_case
    events=[]
    def generator(prompts,cap=64,capture=True):
        events.append('answer' if capture else 'preprocess')
        n=len(prompts)
        return {'texts':['<final>Paris</final>' if capture else '[1, 2]']*n,
                'states':np.ones((n,6,4),np.float16) if capture else None,
                'prompt_ids':[[1,2]]*n,'output_ids':[[3]]*n,
                'input_tokens':[2]*n,'output_tokens':[1]*n,
                'confidence':np.zeros((n,4)).tolist()}
    def decide(states,texts,confidence):
        events.append('saved');assert states.shape==(1,3,6,4)
    cases=[build_case(raw(),fillers(),2)]
    states,records=c.execute(cases,generator,decide)
    assert events[:3]==['answer','saved','preprocess']
    assert states.shape==(1,3,5,6,4)
    assert len(records)==1 and len(records[0]['versions'])==3
    for v in records[0]['versions']:
        for action in v['actions']:
            assert not {'gold','score','f1','em','correct'}&set(action)
