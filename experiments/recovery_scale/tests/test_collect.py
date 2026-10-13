import importlib
import copy
import numpy as np
import pytest
from .test_actions import raw


def module():
    try: return importlib.import_module('experiments.recovery_scale.collect')
    except ModuleNotFoundError: pytest.fail('Independent expanded action collection is missing')


class Tokenizer:
    def encode(self, text, **kwargs): return text.split()
    def apply_chat_template(self, messages, **kwargs): return messages[0]['content']


def generator(seen):
    def generate(model, tok, prompts, cap, capture=False, seed=None):
        seen.extend((p, cap, capture, seed) for p in prompts)
        if capture: texts = ['<final>BASE_ANCHOR</final>'] * len(prompts)
        elif prompts[0].startswith('Write a short search query'): texts = ['SEARCH_INTERMEDIATE'] * len(prompts)
        elif prompts[0].startswith('Check the initial answer against'): texts = ['CRITIQUE_INTERMEDIATE'] * len(prompts)
        else: texts = ['<final>RECOVERY_ANSWER</final>'] * len(prompts)
        result = {'texts':texts, 'lengths':[2]*len(prompts),
                  'input_lengths':[len(tok.encode(p)) for p in prompts],
                  'at_cap':[False]*len(prompts), 'layers':[1,2]}
        if capture:
            result.update(states=np.ones((len(prompts),2,2,4),np.float16),
                          confidence=np.ones((len(prompts),13),np.float32))
        return result
    return generate


def test_every_branch_starts_from_baseline_and_multicall_costs_are_complete():
    c = module(); seen=[]
    rows, arrays = c.collect_batch(None, Tokenizer(), [raw()], 2, seed=4, generate=generator(seen))
    r=rows[0]
    assert len(seen) == 14
    assert sum(p[2] for p in seen) == 1
    assert all('GOLD_SECRET' not in p[0] for p in seen)
    assert all('RECOVERY_ANSWER' not in p[0] for p in seen)
    assert r['actions']['query']['calls'] == 2
    assert r['actions']['verify']['calls'] == 2
    assert r['actions']['vote']['calls'] == 3
    assert r['actions']['keep']['calls'] == 0
    assert r['actions']['vote']['answer'] == 'RECOVERY_ANSWER'
    for a, value in r['actions'].items():
        if a!='keep':
            assert value['reserved_tokens'] >= value['input_tokens'] + value['output_tokens']
    assert arrays['states'].shape == (1,2,2,4)
    changed=raw();changed['answer']='another gold'
    seen2=[]
    c.collect_batch(None, Tokenizer(), [changed], 2, seed=4, generate=generator(seen2))
    assert seen == seen2


def test_resume_rejects_code_data_and_row_order_drift():
    c=module()
    payload={'fingerprint':'abc','data_hash':'def','rows':[{'id':'a'},{'id':'b'}]}
    c.validate_batch(payload, 'abc', 'def', ['a','b'])
    for fp,dh,ids in [('xyz','def',['a','b']),('abc','xyz',['a','b']),('abc','def',['b','a'])]:
        with pytest.raises(ValueError,match='cached'):
            c.validate_batch(payload,fp,dh,ids)
