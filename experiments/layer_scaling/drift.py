"""Input-drift study (docs/prereg_input_drift.md): add text to the input, capture every layer, judge the answers.

    PYTHONHASHSEED=0 python -m experiments.layer_scaling.drift collect --model llama32_3b --n 1000
    python -m experiments.layer_scaling.drift judge --model llama32_3b          # Qwen2.5-7B yes/no judge
"""
import argparse
import json
import os
import random

os.environ.setdefault('ACTIVATION_LAYERS', 'all')

import numpy as np
import torch

from experiments.layer_scaling.collect import OUT as SCALING_OUT, load
from experiments.paths import SCALE_RUN
from experiments.recovery_help.backend import chat, generate_batch
from experiments.recovery_help.data import dump
from experiments.recovery_scale.actions import (DIRECT_CAP, extract_answer, prompt, public_example, ranked,
                                                score_answer)

OUT = SCALING_OUT.parent / 'input_drift'
CONDITIONS = ('base', 'more_evidence', 'oracle_evidence', 'metadata', 'warning', 'filler', 'misleading')
WARNING = ('Be careful: questions like this are often answered incorrectly. '
           'Check each step against the evidence.')
JUDGE = 'Qwen/Qwen2.5-7B-Instruct'


def with_line(text, line):
    """Insert one line directly before the question, keeping Som's prompt otherwise unchanged."""
    head, sep, tail = text.rpartition('\n\nQuestion: ')
    if not sep:
        raise ValueError('prompt has no question marker')
    return f'{head}\n\n{line}{sep}{tail}'


def condition_prompts(raw, pool, rng):
    """The seven input variants for one HotpotQA row (raw), using Som's keep prompt."""
    e = public_example(raw)
    top = ranked(e)
    gold_titles = set(raw['supporting_facts']['title'])
    out = {'base': prompt(e, 'keep', initial_k=2), 'more_evidence': prompt(e, 'keep', initial_k=5)}
    support = [p for p in e['paragraphs'] if p['title'] in gold_titles]
    others = [e['paragraphs'][i] for i in top if e['paragraphs'][i]['title'] not in gold_titles][:2]
    oracle = dict(e, paragraphs=support + others)
    out['oracle_evidence'] = prompt(oracle, 'keep', initial_k=len(oracle['paragraphs']))
    out['metadata'] = with_line(out['base'], f"Question type: {raw['type']}. Difficulty: {raw['level']}.")
    out['warning'] = with_line(out['base'], WARNING)
    other = public_example(pool[rng.randrange(len(pool))])
    while other['id'] == e['id']:
        other = public_example(pool[rng.randrange(len(pool))])
    filler = dict(e, paragraphs=[e['paragraphs'][i] for i in top[:2]] + other['paragraphs'][:3])
    out['filler'] = prompt(filler, 'keep', initial_k=5)
    gold = raw['answer'].strip()
    if gold.lower() in ('yes', 'no'):
        hint = 'no' if gold.lower() == 'yes' else 'yes'
    else:
        hint = gold
        while hint.lower() == gold.lower() or hint.lower() in ('yes', 'no'):
            hint = pool[rng.randrange(len(pool))]['answer'].strip()
    out['misleading'] = with_line(out['base'], f'Hint: the answer might be {hint}.')
    return out


def collect(key, n, batch_size):
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Run with PYTHONHASHSEED=0 for stable BM25 accumulation')
    rows = json.loads((SCALE_RUN / 'data' / 'test.json').read_text())[:n]
    pool = json.loads((SCALE_RUN / 'data' / 'test.json').read_text())
    rng = random.Random(20261020)
    prompts = [condition_prompts(r, pool, rng) for r in rows]    # built once, before any generation
    model, tok = load(key)
    for c in CONDITIONS:
        d = OUT / key / c
        d.mkdir(parents=True, exist_ok=True)
        for s in range(0, n, batch_size):
            stem = d / f'batch_{s:05d}'
            if stem.with_suffix('.json').exists():
                continue
            chunk = rows[s:s + batch_size]
            res = generate_batch(model, tok, [chat(tok, p[c]) for p in prompts[s:s + batch_size]], DIRECT_CAP,
                                 capture=True)
            answers = [extract_answer(t) for t in res['texts']]
            tmp = stem.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f, states=res['states'], confidence=res['confidence'])
            tmp.replace(stem.with_suffix('.npz'))
            dump(stem.with_suffix('.json'), {'layers': res['layers'], 'rows': [
                {'id': r['id'], 'question': r['question'], 'gold': r['answer'], 'type': r['type'],
                 'level': r['level'], 'prompt': p[c], 'text': t, 'answer': a,
                 'f1': score_answer(a, r['answer'])['f1'], 'input_tokens': il}
                for r, p, t, a, il in zip(chunk, prompts[s:s + batch_size], res['texts'], answers,
                                          res['input_lengths'])]})
            print(json.dumps({'model': key, 'condition': c, 'done': s + len(chunk), 'total': n}), flush=True)


@torch.no_grad()
def judge(key, batch_size=32):
    """Same yes/no prompt and logit rule as FACR's scripts/judge_audit.py."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16).to('cuda').eval()
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ('yes', 'Yes', ' yes', ' Yes')]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ('no', 'No', ' no', ' No')]
    for c in CONDITIONS:
        path = OUT / key / c / 'judged.json'
        if path.exists():
            continue
        items = [r for jp in sorted((OUT / key / c).glob('batch_*.json')) for r in json.loads(jp.read_text())['rows']]
        verdicts = []
        for s in range(0, len(items), batch_size):
            msgs = [tok.apply_chat_template([{'role': 'user', 'content':
                    f"Question: {it['question']}\nReference answers: {it['gold']}\n"
                    f"Proposed answer: {it['answer']}\n\nIs the proposed answer correct, i.e. does it refer to the "
                    f"same thing as one of the reference answers? Reply with only yes or no."}],
                    tokenize=False, add_generation_prompt=True) for it in items[s:s + batch_size]]
            enc = tok(msgs, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
            lg = model(**enc).logits[:, -1].float()
            verdicts += (lg[:, yes].logsumexp(-1) > lg[:, no].logsumexp(-1)).int().tolist()
        dump(path, {'judge': JUDGE, 'ids': [it['id'] for it in items], 'correct': verdicts})
        print(json.dumps({'model': key, 'condition': c, 'judged': len(verdicts),
                          'accuracy': float(np.mean(verdicts))}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['collect', 'judge'])
    ap.add_argument('--model', default='llama32_3b')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=24)
    a = ap.parse_args()
    collect(a.model, a.n, a.batch_size) if a.step == 'collect' else judge(a.model)
