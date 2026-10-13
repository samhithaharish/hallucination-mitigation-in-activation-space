import copy


def example():
    return {'id':'q1','question':'Where was Ada born?',
      'context':{'title':['Ada','Other','Town','Music','Art','Noise'],
                 'sentences':[['Ada was born in Paris.'],['Ben lives in Rome.'],['Paris is a city.'],['Music is an art.'],['A painting.'],['Noise has volume.']]},
      'answer':'Paris','supporting_facts':{'title':['Ada'],'sent_id':[0]}}


def test_prompts_and_ranking_ignore_gold_and_support_annotations():
    from experiments.recovery_help.task import public_example,build_prompt,ACTIONS
    original=example();changed=copy.deepcopy(original);changed['answer']='SECRET'
    changed['supporting_facts']={'title':['Noise'],'sent_id':[99]}
    assert public_example(original)==public_example(changed)
    for action in ACTIONS:
        a=build_prompt(public_example(original),action,'Rome')
        assert a==build_prompt(public_example(changed),action,'Rome')
        assert 'SECRET' not in a


def test_retrieval_adds_evidence_and_reasoning_keeps_initial_evidence():
    from experiments.recovery_help.task import public_example,build_prompt,rank_passages
    e=public_example(example());rank=rank_passages(e['question'],e['paragraphs'])
    assert rank[0]==0
    reason=build_prompt(e,'reason','Rome');retrieve=build_prompt(e,'retrieve','Rome')
    assert 'Initial answer: Rome' in reason and 'Initial answer: Rome' in retrieve
    assert reason.count('[Passage ')==2 and retrieve.count('[Passage ')==5
    assert '<final>short answer</final>' not in reason  # prevent literal placeholder copying


def test_reasoning_extraction_does_not_score_answer_in_rationale():
    from experiments.recovery_help.task import extract_answer,score_answer
    text='Paris could be relevant. <final>Rome</final>'
    assert score_answer(extract_answer(text,True),'Paris')=={'em':0.,'f1':0.}
    assert extract_answer('The answer is Paris but I need to check.',True)==''
    assert extract_answer('<final>Paris',True)==''
    assert extract_answer('<final>Paris</final>',True)=='Paris'


def test_hotpot_scoring_and_special_yes_no_rules():
    from experiments.recovery_help.task import score_answer
    assert score_answer('The Pacific Ocean.','Pacific Ocean')=={'em':1.,'f1':1.}
    assert score_answer('France and Germany','France')['f1']==.5
    assert score_answer('yes because it is','yes')['f1']==0.


def test_split_embargo_removes_shared_titles_and_questions():
    from experiments.recovery_help.data import eligible_train
    a=example();b=copy.deepcopy(a);b['id']='b';b['question']='Different question'
    assert not eligible_train(a,{'Ada'},set())
    assert not eligible_train(a,set(),{'where was ada born'})
    assert eligible_train(b,{'Another title'},set())


def test_retrieval_metadata_has_fixed_width_for_short_candidate_pools():
    from experiments.recovery_help.task import retrieval_metadata, public_example
    e = public_example(example())
    complete = retrieval_metadata(e)
    e['paragraphs'] = e['paragraphs'][:2]
    short = retrieval_metadata(e)
    assert len(short) == len(complete) == 8
    assert short[2:5] == [0., 0., 0.]
    assert short[-1] == 2
