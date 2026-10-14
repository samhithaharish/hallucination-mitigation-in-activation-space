import importlib
import copy
import pytest


def module():
    try:
        return importlib.import_module('experiments.paired_evidence.task')
    except ImportError:
        pytest.fail('Paired evidence task is not implemented')


def raw():
    titles=['A','B']+[f'D{i}' for i in range(8)]
    return {'id':'case','question':'Where was the artist who made Z born?',
        'answer':'Paris','supporting_facts':{'title':['A','B'],'sent_id':[0,0]},
        'context':{'title':titles,'sentences':[['A made Z.'],['A was born in Paris.']]+
            [[f'D{i} is an unrelated artist with a different work.'] for i in range(8)]}}


def fillers():
    return [{'title':f'N{i}','text':'Trees and stones form a landscape with streams and fields.'}
            for i in range(12)]


def test_conditions_preserve_question_and_complete_support_without_public_labels():
    t=module(); r=raw(); case=t.build_case(r,fillers(),seed=7)
    assert set(case['versions'])=={'clean','missing','distractor'}
    assert case['answer']=='Paris'
    assert len({tuple(p['title'] for p in v['current']) for v in case['versions'].values()})==3
    for name,public in case['versions'].items():
        assert public['question']==r['question'] and len(public['current'])==10
        assert set(public)=={'id','question','current','candidates'}
        assert len(public['candidates'])==10
        assert len({p['title'] for p in public['current']})==10
        support=set(p['title'] for p in public['current'])&{'A','B'}
        assert len(support)==(1 if name=='missing' else 2)
    clean=case['versions']['clean']['current']; bad=case['versions']['missing']['current']
    assert sum(a!=b for a,b in zip(clean,bad))==1


def test_case_is_deterministic_and_does_not_mutate_raw():
    t=module(); r=raw(); before=copy.deepcopy(r)
    assert t.build_case(r,fillers(),3)==t.build_case(r,fillers(),3)
    assert r==before


def test_selection_uses_only_valid_unique_passage_ids():
    t=module(); e=t.build_case(raw(),fillers(),2)['versions']['clean']
    chosen,valid=t.select_passages(e,'[2, 2, 100, 1, -1]')
    assert valid and chosen==[e['current'][1],e['current'][0]]
    fallback,valid=t.select_passages(e,'the answer is Paris')
    assert not valid and len(fallback)==4


def test_retrieval_preserves_current_evidence_and_adds_only_public_candidates():
    t=module(); e=t.build_case(raw(),fillers(),2)['versions']['missing']
    result=t.expand(e,'artist Z birth')
    assert result[:10]==e['current']
    assert len({p['title'] for p in result})==len(result)
    assert all(p in e['candidates'] for p in result[10:])


def test_final_prompts_share_suffix_and_do_not_expose_gold_or_condition():
    t=module(); case=t.build_case(raw(),fillers(),4)
    for e in case['versions'].values():
        prompt=t.answer_prompt(e,e['current'])
        assert prompt.endswith('Response:')
        assert 'supporting_facts' not in prompt and 'condition:' not in prompt.lower()
    assert '<final>' in t.answer_prompt(case['versions']['clean'],[])


def test_invalid_support_annotations_are_rejected_before_building_pairs():
    t=module(); r=raw();r['supporting_facts']['sent_id'][0]=99
    assert not t.valid_support(r)
    with pytest.raises(ValueError,match='support annotation'):
        t.build_case(r,fillers(),3)


def test_focus_preserves_every_passage_and_moves_selected_evidence_to_end():
    t=module();e=t.build_case(raw(),fillers(),3)['versions']['distractor']
    selected=[e['current'][3],e['current'][1]]
    result=t.focus(e,selected)
    assert result[-2:]==selected
    assert sorted((p['title'],p['text']) for p in result)==sorted((p['title'],p['text']) for p in e['current'])
    assert len({p['title'] for p in result})==10
