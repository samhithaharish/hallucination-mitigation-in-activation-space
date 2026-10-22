"""Equal-size context selection pilot (docs/prereg_context_select.md): every paragraph pair, greedy answer, judge.

    PYTHONHASHSEED=0 ACTIVATION_LAYERS=12,15,18,21,24 python -m experiments.layer_scaling.context_select collect
    python -m experiments.layer_scaling.context_select judge
"""
import argparse
import itertools
import json
import os

import numpy as np
import torch

from experiments.layer_scaling.collect import OUT as SCALING_OUT, load
from experiments.layer_scaling.drift import JUDGE
from experiments.paths import SCALE_RUN
from experiments.recovery_help.backend import chat, generate_batch
from experiments.recovery_help.data import dump
from experiments.recovery_scale.actions import DIRECT_CAP, extract_answer, prompt, public_example, score_answer

OUT = SCALING_OUT.parent / 'context_select'


def items(n):
    """(question row, paragraph pair) for every pair of every question, in a fixed order."""
    rows = json.loads((SCALE_RUN / 'data' / 'test.json').read_text())[:n]
    out = []
    for qi, r in enumerate(rows):
        e = public_example(r)
        for i, j in itertools.combinations(range(len(e['paragraphs'])), 2):
            pair = dict(e, paragraphs=[e['paragraphs'][i], e['paragraphs'][j]])
            out.append({'qi': qi, 'id': r['id'], 'question': r['question'], 'gold': r['answer'],
                        'pair': [i, j], 'titles': [e['paragraphs'][i]['title'], e['paragraphs'][j]['title']],
                        'support': sorted(set(r['supporting_facts']['title'])),
                        'prompt': prompt(pair, 'keep', initial_k=2)})
    return out


def collect(key, n, batch_size):
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Run with PYTHONHASHSEED=0 for stable BM25 accumulation')
    if not os.environ.get('ACTIVATION_LAYERS') or os.environ['ACTIVATION_LAYERS'] == 'all':
        raise ValueError('Set ACTIVATION_LAYERS to the five pre-registered layers')
    its = items(n)
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    model, tok = load(key)
    for s in range(0, len(its), batch_size):
        stem = d / f'batch_{s:06d}'
        if stem.with_suffix('.json').exists():
            continue
        chunk = its[s:s + batch_size]
        res = generate_batch(model, tok, [chat(tok, it['prompt']) for it in chunk], DIRECT_CAP, capture=True)
        tmp = stem.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, states=res['states'], confidence=res['confidence'])
        tmp.replace(stem.with_suffix('.npz'))
        rows = []
        for it, t in zip(chunk, res['texts']):
            a = extract_answer(t)
            rows.append({**{k: v for k, v in it.items() if k != 'prompt'}, 'text': t, 'answer': a,
                         'f1': score_answer(a, it['gold'])['f1']})
        dump(stem.with_suffix('.json'), {'layers': res['layers'], 'rows': rows})
        if (s // batch_size) % 50 == 0:
            print(json.dumps({'model': key, 'done': s + len(chunk), 'total': len(its)}), flush=True)


@torch.no_grad()
def judge(key, batch_size=48):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = OUT / key / 'judged.json'
    if path.exists():
        return
    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16).to('cuda').eval()
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ('yes', 'Yes', ' yes', ' Yes')]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ('no', 'No', ' no', ' No')]
    its = [r for jp in sorted((OUT / key).glob('batch_*.json')) for r in json.loads(jp.read_text())['rows']]
    verdicts = []
    for s in range(0, len(its), batch_size):
        msgs = [tok.apply_chat_template([{'role': 'user', 'content':
                f"Question: {it['question']}\nReference answers: {it['gold']}\n"
                f"Proposed answer: {it['answer']}\n\nIs the proposed answer correct, i.e. does it refer to the "
                f"same thing as one of the reference answers? Reply with only yes or no."}],
                tokenize=False, add_generation_prompt=True) for it in its[s:s + batch_size]]
        enc = tok(msgs, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
        lg = model(**enc).logits[:, -1].float()
        verdicts += (lg[:, yes].logsumexp(-1) > lg[:, no].logsumexp(-1)).int().tolist()
    dump(path, {'judge': JUDGE, 'keys': [[it['id'], *it['pair']] for it in its], 'correct': verdicts})
    print(json.dumps({'model': key, 'judged': len(verdicts), 'accuracy': float(np.mean(verdicts))}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['collect', 'judge'])
    ap.add_argument('--model', default='llama32_3b')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=24)
    a = ap.parse_args()
    collect(a.model, a.n, a.batch_size) if a.step == 'collect' else judge(a.model)
