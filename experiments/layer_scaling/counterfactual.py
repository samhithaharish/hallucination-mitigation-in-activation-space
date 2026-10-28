"""Matched original / counterfactual contexts (docs/prereg_grounding_errors.md, design B).

    PYTHONHASHSEED=0 python -m experiments.layer_scaling.counterfactual plant --split validation   # Qwen-7B proposals
    PYTHONHASHSEED=0 ACTIVATION_LAYERS=all python -m experiments.layer_scaling.counterfactual run --model llama32_3b
    python -m experiments.layer_scaling.counterfactual summary --model llama32_3b

O: the question with its two gold supporting passages (Som's keep prompt). CF: identical prompt except every
occurrence of the gold answer inside the passages is replaced by a type-matched false answer.
"""
import argparse
import json
import random
import re

import numpy as np
import torch

from experiments.layer_scaling.collect import OUT as SCALING_OUT, load
from experiments.layer_scaling.drift import JUDGE
from experiments.paths import SCALE_RUN
from experiments.recovery_help.backend import chat, generate_batch
from experiments.recovery_help.data import dump
from experiments.recovery_scale.actions import DIRECT_CAP, extract_answer, prompt, public_example

OUT = SCALING_OUT.parent / 'counterfactual'
MARK = '\n\nQuestion: '


def norm(t):
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', (t or '').lower()).split())


def contains(text, ans):
    a = norm(ans)
    return bool(a) and f' {a} ' in f' {norm(text)} '


def candidates(split, limit):
    """Questions whose gold answer appears verbatim in both-or-either gold supporting passage (shown in full)."""
    out = []
    for r in json.loads((SCALE_RUN / 'data' / f'{split}.json').read_text()):
        gold = r['answer'].strip()
        if norm(gold) in ('yes', 'no') or not gold or contains(r['question'], gold):
            continue
        e = public_example(r)
        support = [p for p in e['paragraphs'] if p['title'] in set(r['supporting_facts']['title'])]
        if len(support) != 2 or not any(gold in p['text'] for p in support):
            continue
        o = prompt(dict(e, paragraphs=support), 'keep', initial_k=2)
        head, _, tail = o.partition(MARK)
        if gold not in head:
            continue
        out.append({'id': r['id'], 'question': r['question'], 'gold': gold, 'type': r['type'], 'level': r['level'],
                    'prompt_O': o, 'evidence': head})
        if len(out) == limit:
            break
    return out


def rule_plant(gold, evidence, rng):
    """Years, dates with a year, and plain numbers: a nearby different value not present in the passages."""
    def shift_year(y):
        for _ in range(50):
            new = str(int(y) + rng.choice([-1, 1]) * rng.randint(1, 9))
            if new not in evidence:
                return new
        return None
    if re.fullmatch(r'1\d{3}|20\d{2}', gold):
        return shift_year(gold)
    m = re.search(r'\b(1\d{3}|20\d{2})\b', gold)
    if m and re.search(r'[A-Za-z]', gold):                       # a date such as "22 November 1985"
        y = shift_year(m.group(1))
        return gold[:m.start(1)] + y + gold[m.end(1):] if y else None
    if re.fullmatch(r'[\d,\.]+', gold):
        digits = gold.replace(',', '')
        try:
            v = float(digits)
        except ValueError:
            return None
        new = v * rng.choice([0.7, 0.8, 1.2, 1.3])
        new = f'{int(round(new)):,}' if ',' in gold else (str(int(round(new))) if '.' not in gold else f'{new:.1f}')
        return new if new != gold and new not in evidence else None
    return None


@torch.no_grad()
def plant(split, limit, batch_size=16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = OUT / f'planted_{split}.json'
    if path.exists():
        return
    cands = candidates(split, limit)
    rng = random.Random(20261028)
    need_llm = []
    for c in cands:
        c['planted'] = rule_plant(c['gold'], c['evidence'], rng)
        c['plant_source'] = 'rule' if c['planted'] else None
        if not c['planted'] and not re.search(r'\d', c['gold']):
            need_llm.append(c)
    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = 'left'
    m = AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16).to('cuda').eval()
    for s in range(0, len(need_llm), batch_size):
        chunk = need_llm[s:s + batch_size]
        msgs = [tok.apply_chat_template([{'role': 'user', 'content':
                f"Question: {c['question']}\nCorrect answer: {c['gold']}\n\nGive a different, plausible but incorrect "
                f"answer of exactly the same type (a person for a person, a city for a city, a film for a film, an "
                f"organisation for an organisation, and so on). It must not be the correct answer. Reply with only "
                f"the alternative answer."}], tokenize=False, add_generation_prompt=True) for c in chunk]
        enc = tok(msgs, return_tensors='pt', padding=True, add_special_tokens=False).to(m.device)
        g = m.generate(**enc, max_new_tokens=24, do_sample=False, temperature=None, top_p=None, top_k=None,
                       pad_token_id=tok.eos_token_id)
        for c, t in zip(chunk, tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            a = t.strip().splitlines()[0].strip().strip('"\'').rstrip('.') if t.strip() else ''
            ok = (a and 1 <= len(a.split()) <= 8 and not contains(a, c['gold']) and not contains(c['gold'], a)
                  and not contains(c['evidence'], a))
            c['planted'], c['plant_source'] = (a, 'llm') if ok else (None, 'rejected')
    kept = [c for c in cands if c['planted']]
    for c in kept:
        head, sep, tail = c['prompt_O'].partition(MARK)
        c['prompt_CF'] = head.replace(c['gold'], c['planted']) + sep + tail
        del c['evidence']
    dump(path, {'split': split, 'n_candidates': len(cands), 'n_planted': len(kept),
                'sources': {k: sum(1 for c in cands if c['plant_source'] == k) for k in ('rule', 'llm', 'rejected')},
                'rows': kept})
    print(json.dumps({'split': split, 'candidates': len(cands), 'planted': len(kept)}), flush=True)


def run(key, split, keep, batch_size):
    rows = json.loads((OUT / f'planted_{split}.json').read_text())['rows']
    d = OUT / key / split
    d.mkdir(parents=True, exist_ok=True)
    model, tok = load(key)

    def answer(prompts):
        res = {'texts': [], 'states': []}
        for s in range(0, len(prompts), batch_size):
            r = generate_batch(model, tok, [chat(tok, p) for p in prompts[s:s + batch_size]], DIRECT_CAP, capture=True)
            res['texts'] += r['texts']
            res['states'].append(r['states'])
        res['states'] = np.concatenate(res['states'])
        return res

    o = answer([r['prompt_O'] for r in rows])
    ans_o = [extract_answer(t) for t in o['texts']]
    ok = [i for i, a in enumerate(ans_o) if contains(a, rows[i]['gold'])][:keep]
    cf = answer([rows[i]['prompt_CF'] for i in ok])
    ans_cf = [extract_answer(t) for t in cf['texts']]
    out = []
    for j, i in enumerate(ok):
        r, a = rows[i], ans_cf[j]
        hp, hg = contains(a, r['planted']), contains(a, r['gold'])
        label = 'G' if hp and not hg else 'C' if hg and not hp else 'U'
        out.append({'id': r['id'], 'question': r['question'], 'gold': r['gold'], 'planted': r['planted'],
                    'plant_source': r['plant_source'], 'answer_O': ans_o[i], 'answer_CF': a, 'label': label})
    np.savez_compressed(d / 'states.npz', O=o['states'][ok], CF=cf['states'])
    dump(d / 'rows.json', {'model': key, 'n_O_run': len(rows), 'n_O_correct': sum(1 for a, r in zip(ans_o, rows)
                                                                              if contains(a, r['gold'])),
                           'rows': out})
    summary(key, split)


def summary(key, split='validation'):
    b = json.loads((OUT / key / split / 'rows.json').read_text())
    lab = [r['label'] for r in b['rows']]
    counts = {k: lab.count(k) for k in 'GCU'}
    print(json.dumps({'model': key, 'O_run': b['n_O_run'], 'O_correct': b['n_O_correct'], 'pairs': len(lab),
                      'counts': counts, 'gate_passed': counts['G'] >= 40 and counts['C'] >= 40}), flush=True)
    for k in 'GCU':
        for r in [r for r in b['rows'] if r['label'] == k][:4]:
            print(f"  {k} | gold {r['gold'][:25]:25s} | planted {r['planted'][:25]:25s} | CF answer: {r['answer_CF'][:50]}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['plant', 'run', 'summary'])
    ap.add_argument('--split', default='validation')
    ap.add_argument('--model', default='llama32_3b')
    ap.add_argument('--limit', type=int, default=400)
    ap.add_argument('--keep', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=16)
    a = ap.parse_args()
    if a.step == 'plant':
        plant(a.split, a.limit)
    elif a.step == 'run':
        run(a.model, a.split, a.keep, a.batch_size)
    else:
        summary(a.model, a.split)
