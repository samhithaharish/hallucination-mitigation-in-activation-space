"""Gold-free recovery prompts and official HotpotQA answer scoring."""
import math
import re
import string
from collections import Counter

ACTIONS = ('keep', 'retry', 'retrieve', 'reason', 'both')


def normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def score_answer(prediction, gold):
    p, g = normalize(prediction), normalize(gold)
    em = float(p == g)
    if any(x in ('yes', 'no', 'noanswer') for x in (p, g)) and p != g:
        return {'em': em, 'f1': 0.}
    pt, gt = p.split(), g.split()
    overlap = sum((Counter(pt) & Counter(gt)).values())
    return {'em': em, 'f1': 2. * overlap / (len(pt) + len(gt)) if overlap else 0.}


def public_example(raw):
    return {'id': raw['id'], 'question': raw['question'], 'paragraphs': [
        {'title': title, 'text': ' '.join(' '.join(sentences).split()[:180])}
        for title, sentences in zip(raw['context']['title'], raw['context']['sentences'])]}


def tokens(text):
    return re.findall(r'\w+', text.lower())


def retrieval_scores(question, paragraphs):
    docs = [tokens(p['title'] + ' ' + p['title'] + ' ' + p['text']) for p in paragraphs]
    counts = [Counter(d) for d in docs]
    avg = sum(map(len, docs)) / max(len(docs), 1)
    scores = [0.] * len(docs)
    for term in set(tokens(question)):
        df = sum(term in c for c in counts)
        idf = math.log(1 + (len(docs) - df + .5) / (df + .5))
        for i, c in enumerate(counts):
            tf = c[term]
            scores[i] += idf * tf * 2.5 / (tf + 1.5 * (.25 + .75 * len(docs[i]) / max(avg, 1)))
    return scores


def rank_passages(question, paragraphs):
    scores = retrieval_scores(question, paragraphs)
    return sorted(range(len(paragraphs)), key=lambda i: (-scores[i], i))


def retrieval_metadata(example):
    scores = sorted(retrieval_scores(example['question'], example['paragraphs']), reverse=True)
    top = (scores + [0.] * 5)[:5]
    return top + [top[0]-top[1], top[1]-top[2], len(scores)]


def build_prompt(example, action='keep', baseline_answer='', initial_k=2):
    if action not in ACTIONS:
        raise ValueError(action)
    rank = rank_passages(example['question'], example['paragraphs'])
    k = 5 if action in ('retrieve', 'both') else initial_k
    passages = '\n\n'.join(
        f"[Passage {j+1}] {example['paragraphs'][i]['title']}\n{example['paragraphs'][i]['text']}"
        for j, i in enumerate(rank[:k]))
    instruction = 'Answer the question using the evidence and your knowledge.'
    if action in ('reason', 'both'):
        instruction += ' Reason carefully through the needed facts, then give the short answer inside <final>...</final>.'
    else:
        instruction += ' Give only the short answer inside <final>...</final>; do not explain.'
    initial = '' if action == 'keep' else (
        '\nInitial answer: ' + baseline_answer + '\nCheck the initial answer and return your best answer.\n')
    ending = ('Give brief reasoning in at most three sentences. Then put your actual answer '
              'between the opening tag <final> and the closing tag </final>.\n'
              if action in ('reason', 'both') else '')
    return f"{instruction}\n\n{passages}\n\nQuestion: {example['question']}\n{initial}{ending}Answer:"


def extract_answer(text, reasoning=False):
    finals = re.findall(r'<final>\s*(.*?)\s*</final>', text, flags=re.S | re.I)
    if finals:
        return finals[-1].strip()
    if reasoning or '<final' in text.lower():
        return ''
    # Direct answers can be concise even if the model omits markup. Never mine
    # an explanation for a reference answer, and never use gold to parse text.
    text = text.strip()
    return text if '\n' not in text and len(text.split()) <= 24 else ''
