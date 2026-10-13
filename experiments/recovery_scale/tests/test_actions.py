import importlib
import copy
import pytest


def module():
    try:
        return importlib.import_module('experiments.recovery_scale.actions')
    except ModuleNotFoundError:
        pytest.fail('Expanded recovery action behavior is missing')


def raw():
    return {'id': 'x', 'question': 'Where was Ada born?', 'answer': 'GOLD_SECRET',
            'supporting_facts': {'title': ['Ada'], 'sent_id': [0]},
            'context': {'title': ['Ada', 'Noise'],
                        'sentences': [['Ada was born in London.', 'She wrote algorithms.'],
                                      ['Bananas are yellow.', 'Many ships sail.']]}}


def test_online_prompts_are_gold_invariant_and_fresh_removes_anchor():
    a = module()
    r = raw(); r2 = copy.deepcopy(r); r2['answer'] = 'CHANGED_SECRET'
    r2['supporting_facts']['title'] = ['Noise']
    for action in a.ACTIONS:
        p = a.prompt(a.public_example(r), action, 'ANCHOR_SECRET', 2)
        assert p == a.prompt(a.public_example(r2), action, 'ANCHOR_SECRET', 2)
        assert 'GOLD_SECRET' not in p and 'CHANGED_SECRET' not in p
        assert ('ANCHOR_SECRET' in p) == (action not in ('keep', 'fresh'))


def test_vote_uses_normalized_answers_without_gold_or_empty_majority():
    a = module()
    assert a.majority(['<final>The London</final>', '<final>London.</final>', '<final>Paris</final>']) == 'The London'
    assert a.majority(['bad rationale', '<final>Paris</final>', '<final>Rome</final>']) == 'Paris'
    assert a.majority(['bad', 'missing tags', '']) == ''


def test_pruning_keeps_relevant_evidence_and_short_candidate_pool_works():
    a = module(); e = a.public_example(raw())
    p = a.prompt(e, 'prune', 'Paris', 2)
    assert 'Ada was born in London.' in p
    assert 'Bananas are yellow.' not in p
    assert len(a.metadata(e)) == 8
    assert 'London' in a.prompt(e, 'query', 'Paris', 2, aux='Ada birthplace London', stage=2)


def test_dynamic_input_bound_preserves_question_and_rejects_impossible_cap():
    a = module(); e = a.public_example(raw())
    class Tokenizer:
        def encode(self, text, **kwargs): return text.split()
        def apply_chat_template(self, messages, **kwargs): return messages[0]['content']
    tok = Tokenizer()
    p = a.bounded_prompt(tok, e, 'verify', 'Paris', 2, 'irrelevant ' * 300, 2, 130)
    assert len(tok.encode(p)) <= 130
    assert e['question'] in p
    with pytest.raises(ValueError, match='question'):
        a.bounded_prompt(tok, e, 'verify', 'Paris', 2, '', 2, 2)


def test_search_uses_first_query_line_without_following_explanation():
    a=module();e=a.public_example(raw())
    assert hasattr(a,'search_query'), 'Search-query extraction is missing'
    assert a.search_query('Search query: Ada London\n\nBananas yellow ships sail')=='Ada London'
    p=a.prompt(e,'query','Paris',2,aux='Search query: Ada London\n\nBananas yellow ships sail',stage=2)
    assert p==a.prompt(e,'query','Paris',2,aux='Ada London',stage=2)
