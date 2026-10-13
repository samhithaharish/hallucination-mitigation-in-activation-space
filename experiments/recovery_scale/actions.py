"""Public-input recovery actions. No labels are accepted by action selection."""
import re
from collections import Counter
from experiments.recovery_help.task import (
    normalize, extract_answer, score_answer, retrieval_scores, retrieval_metadata)
from experiments.recovery_help.backend import chat

ACTIONS = ('keep', 'retry', 'retrieve', 'reason', 'both', 'fresh', 'prune', 'query', 'verify', 'vote')
CALLS = (0, 1, 1, 1, 1, 1, 1, 2, 2, 3)
DIRECT_CAP, REASON_CAP, QUERY_CAP, CRITIQUE_CAP, DYNAMIC_INPUT_CAP = 64, 256, 64, 192, 2048


def public_example(raw):
    return {'id': raw['id'], 'question': raw['question'], 'paragraphs': [
        {'title': title, 'text': ' '.join(' '.join(sentences).split()[:180])}
        for title, sentences in zip(raw['context']['title'], raw['context']['sentences'])]}


def metadata(example):
    return retrieval_metadata(example)


def ranked(example, query=None):
    # The inherited BM25 addition order is deterministic with PYTHONHASHSEED=0.
    scores = retrieval_scores(query or example['question'], example['paragraphs'])
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def search_query(text):
    line=next((line.strip() for line in text.splitlines() if line.strip()),'')
    line=re.sub(r'^(search\s+)?query\s*:\s*','',line,flags=re.I)
    return ' '.join(line.strip('"\' ').split()[:64])


def prompt(example, action='keep', baseline='', initial_k=2, aux='', stage=1, passage_words=180):
    if action not in ACTIONS:
        raise ValueError(action)
    expanded = example['question'] + ' ' + search_query(aux)
    rank = ranked(example, expanded if action == 'query' and stage == 2 else None)
    k = initial_k if action in ('keep', 'retry', 'reason') or (action == 'query' and stage == 1) else 5
    paragraphs = [dict(example['paragraphs'][i]) for i in rank[:k]]
    if action == 'prune':
        sentences = [{'title': p['title'], 'text': s} for p in paragraphs
                     for s in re.split(r'(?<=[.!?])\s+', p['text']) if s.strip()]
        score = retrieval_scores(example['question'], sentences)
        chosen = sorted(range(len(sentences)), key=lambda i: (-score[i], i))[:6]
        paragraphs = [sentences[i] for i in sorted(chosen) if score[i] > 0]
    evidence = '\n\n'.join(f"[Passage {i+1}] {p['title']}\n{' '.join(p['text'].split()[:passage_words])}"
                           for i, p in enumerate(paragraphs) if passage_words)
    instruction = 'Answer the question using the evidence and your knowledge.'
    reasoning = action in ('reason', 'both', 'vote') or (action in ('query', 'verify') and stage == 2)
    if action == 'query' and stage == 1:
        instruction = ('Write a short search query to find missing evidence needed to answer the question. '
                       'Use specific entity names and relations. Return only the search query, not an answer.')
    elif action == 'verify' and stage == 1:
        instruction = ('Check the initial answer against the evidence. Identify any unsupported claim, '
                       'conflicting fact, or missing reasoning step. Give a brief critique in at most three sentences.')
    elif reasoning:
        instruction += (' Reason carefully in at most three sentences, then put the short answer '
                        'between the opening tag <final> and the closing tag </final>.')
    else:
        instruction += ' Give only the short answer inside <final>...</final>; do not explain.'
    anchor = '' if action in ('keep', 'fresh') else f'\nInitial answer: {baseline}\nCheck the initial answer and return your best answer.\n'
    if action in ('query','verify') and stage==1:
        anchor=f'\nInitial answer (may be wrong): {baseline}\n'
    critique = f'\nCritique to consider (may be mistaken): {aux}\n' if action == 'verify' and stage == 2 else ''
    return f"{instruction}\n\n{evidence}\n\nQuestion: {example['question']}\n{anchor}{critique}Response:"


def bounded_prompt(tok, example, action, baseline, initial_k, aux, stage=2, cap=DYNAMIC_INPUT_CAP):
    # Preserve task, question and initial answer. Reduce evidence first, then critique.
    if action=='query':aux=search_query(aux)
    for aux_words in (128, 64, 16, 0):
        shortened = ' '.join(aux.split()[:aux_words])
        for words in (180, 90, 45, 20, 0):
            p = chat(tok, prompt(example, action, baseline, initial_k, shortened, stage, words))
            if len(tok.encode(p, add_special_tokens=False)) <= cap:
                return p
    raise ValueError('Cannot preserve question and instructions within dynamic input cap')


def majority(texts):
    answers = [extract_answer(t, reasoning=True) for t in texts]
    valid = [a for a in answers if normalize(a)]
    if not valid:
        return ''
    counts = Counter(normalize(a) for a in valid)
    winner = max(counts, key=counts.get)
    return next(a for a in valid if normalize(a) == winner)


def cap_for(action, stage=1):
    if action == 'query' and stage == 1: return QUERY_CAP
    if action == 'verify' and stage == 1: return CRITIQUE_CAP
    if action in ('reason', 'both', 'query', 'verify', 'vote'): return REASON_CAP
    return DIRECT_CAP


def costs(tok, example, baseline, k):
    reserved = [0]
    for action in ACTIONS[1:]:
        p = chat(tok, prompt(example, action, baseline, k))
        first = len(tok.encode(p, add_special_tokens=False)) + cap_for(action)
        if action in ('query', 'verify'): first += DYNAMIC_INPUT_CAP + REASON_CAP
        if action == 'vote': first *= 3
        reserved.append(first)
    return reserved
