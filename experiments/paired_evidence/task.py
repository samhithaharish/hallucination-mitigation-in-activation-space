"""Controlled inputs and public recovery actions; annotations never enter prompts."""
import copy
import hashlib
import json
import re
import numpy as np
from experiments.recovery_help.task import retrieval_scores

CONDITIONS = ('clean','missing','distractor')
ACTIONS = ('keep','retrieve','filter','sham','empty')


def limited(paragraph, n=120):
    return {'title':paragraph['title'], 'text':' '.join(paragraph['text'].split()[:n])}


def valid_support(raw):
    paragraphs=dict(zip(raw['context']['title'],raw['context']['sentences']))
    return all(title in paragraphs and 0 <= index < len(paragraphs[title])
        for title,index in zip(raw['supporting_facts']['title'],raw['supporting_facts']['sent_id']))


def build_case(raw, fillers, seed=20261013):
    if not valid_support(raw):
        raise ValueError('Invalid support annotation')
    rng=np.random.default_rng(seed)
    required={}
    for title, index in zip(raw['supporting_facts']['title'],raw['supporting_facts']['sent_id']):
        required.setdefault(title,set()).add(index)
    if len(required)!=2:
        raise ValueError('Need exactly two annotated supporting articles')
    paragraphs=[]
    for title,sentences in zip(raw['context']['title'],raw['context']['sentences']):
        ordered=[sentences[i] for i in sorted(required.get(title,set()))]
        if title in required:
            # Keep every annotated support sentence intact; add ordinary context
            # only up to the approximate common word cap.
            words=' '.join(ordered).split()
            for i,s in enumerate(sentences):
                if i not in required[title] and len(words)<120:
                    words.extend(s.split()[:120-len(words)])
            text=' '.join(words)
        else:
            text=' '.join(' '.join(sentences).split()[:120])
        paragraphs.append({'title':title,'text':text})
    support=[p for p in paragraphs if p['title'] in required]
    distractors=[p for p in paragraphs if p['title'] not in required]
    if len(support)!=2 or len(distractors)!=8 or len(fillers)<10:
        raise ValueError('Expected ten distinct candidate articles and at least ten fillers')
    # Random public distractors are paired with word-matched unrelated passages.
    order=rng.permutation(10)
    neutral=[limited(fillers[i],len(p['text'].split())) for i,p in enumerate(distractors)]
    clean=support+neutral
    missing=copy.deepcopy(clean)
    removed=int(rng.integers(2))
    missing[removed]=limited(fillers[8],len(support[removed]['text'].split()))
    noisy=support+distractors
    versions={name:{'id':raw['id'],'question':raw['question'],
                   'current':[copy.deepcopy(items[int(i)]) for i in order],
                   'candidates':copy.deepcopy(paragraphs)}
              for name,items in zip(CONDITIONS,(clean,missing,noisy))}
    return {'id':raw['id'],'question':raw['question'],'answer':raw['answer'],
            'removed_title':support[removed]['title'],
            'support_titles':list(required),'versions':versions}


def evidence(paragraphs):
    return '\n\n'.join(f"[Passage {i+1}] {p['title']}\n{p['text']}"
                        for i,p in enumerate(paragraphs))


def answer_prompt(example, paragraphs, sham=False):
    instruction=('Answer the question using the supplied evidence and your knowledge. '
                 'Give only the short answer inside <final>...</final>; do not explain.')
    if sham:
        instruction+=' Read the passages carefully before answering.'
    return f"{instruction}\n\n{evidence(paragraphs)}\n\nQuestion: {example['question']}\nResponse:"


def query_prompt(example):
    return ('Write one short search query for additional facts needed to answer the question. '
            'Mention specific entities and relations. Return only the query, without an answer.\n\n'
            +evidence(example['current'])+'\n\nQuestion: '+example['question']+'\nSearch query:')


def selection_prompt(example):
    return ('Select at most four passages needed to answer the question. Include passages '
            'needed to connect entities across facts; discard passages about different entities '
            'or unrelated relations. Return only a JSON array of passage numbers, such as [2, 5]. '
            'Do not answer the question.\n\n'+evidence(example['current'])+
            '\n\nQuestion: '+example['question']+'\nPassage numbers:')


def expand(example, query):
    query=' '.join(query.split()[:64])
    scores=retrieval_scores(example['question']+' '+query,example['candidates'])
    rank=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    result=copy.deepcopy(example['current']); seen={p['title'] for p in result}
    for i in rank:
        p=example['candidates'][i]
        if p['title'] not in seen:
            result.append(copy.deepcopy(p));seen.add(p['title'])
            if len(result)==len(example['current'])+4:break
    return result


def select_passages(example, text):
    match=re.search(r'\[[^\]]*\]',text)
    try:
        ids=json.loads(match.group()) if match else []
        selected=[]
        for i in ids:
            if isinstance(i,int) and not isinstance(i,bool) and 1<=i<=len(example['current']) and i-1 not in selected:
                selected.append(i-1)
        if selected:
            return [copy.deepcopy(example['current'][i]) for i in selected[:4]],True
    except (ValueError,TypeError):
        pass
    scores=retrieval_scores(example['question'],example['current'])
    rank=sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:4]
    return [copy.deepcopy(example['current'][i]) for i in sorted(rank)],False
