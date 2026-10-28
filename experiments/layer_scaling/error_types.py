"""Label baseline answers as correct / concept drift / hallucination / refusal (docs/prereg_error_types.md).

    PYTHONHASHSEED=0 python -m experiments.layer_scaling.error_types label --model llama32_3b   # GPU, Qwen-7B judge
    python -m experiments.layer_scaling.error_types spot --model llama32_3b                     # 25 per label -> md
"""
import argparse
import json
import random

import numpy as np
import torch

from experiments.layer_scaling.collect import OUT as SCALING_OUT
from experiments.layer_scaling.drift import JUDGE
from experiments.paths import SCALE_RUN
from experiments.recovery_help.data import dump
from experiments.recovery_scale.actions import public_example, ranked

OUT = SCALING_OUT.parent / 'error_types'
RESULTS = SCALING_OUT.parents[1] / 'results' / 'error_types'
SPLITS = ('train', 'validation', 'test')

LABELS = ('correct', 'refusal', 'concept_drift', 'hallucination', 'no_answer', 'yesno_wrong')
# Staged yes/no questions (a 4-way prompt mislabelled empty answers and refusals in the 48-answer pilot).
Q_CORRECT = ("Is the proposed answer correct, i.e. does it refer to the same thing as the reference answer? "
             "Reply with only yes or no.")
Q_REFUSAL = ("Does the proposed answer say that it cannot answer, that the information is not available, or that "
             "it does not know? Reply with only yes or no.")
Q_RELATED = ("The proposed answer is wrong. Is it nevertheless a real, closely related concept, for example an entity "
             "mentioned in the question or the passages, or a related entity in the wrong role (like 'Sydney' for the "
             "capital of Australia)? Answer no if it is invented, unrelated or unsupported. Reply with only yes or no.")


def _norm(t):
    import re
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', t.lower()).split())


def items(model):
    """Every baseline answer for the model with the two passages it saw (Som's top-2 BM25)."""
    out = []
    for split in SPLITS:
        raw = {r['id']: r for r in json.loads((SCALE_RUN / 'data' / f'{split}.json').read_text())}
        k = 1 if split == 'transfer' else 2
        for jp in sorted((SCALING_OUT / model / split).glob('batch_*.json')):
            for r in json.loads(jp.read_text())['rows']:
                e = public_example(raw[r['id']])
                passages = [e['paragraphs'][i] for i in ranked(e)[:k]]
                keep = r['actions']['keep']
                out.append({'split': split, 'id': r['id'], 'question': raw[r['id']]['question'],
                            'gold': r['gold'], 'answer': keep['answer'], 'f1': keep['score']['f1'],
                            'passages': [f"{p['title']}: {' '.join(p['text'].split()[:180])}" for p in passages]})
    return out


def message(it, q):
    ev = '\n\n'.join(f'[Passage {i + 1}] {p}' for i, p in enumerate(it['passages']))
    return (f"Question: {it['question']}\n\nPassages the model saw:\n{ev}\n\nReference answer: {it['gold']}\n"
            f"Proposed answer: {it['answer']}\n\n{q}")


@torch.no_grad()
def yes_prob(m, tok, its, q, batch_size=16):
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ('yes', 'Yes', ' yes', ' Yes')]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ('no', 'No', ' no', ' No')]
    out = []
    for s in range(0, len(its), batch_size):
        msgs = [tok.apply_chat_template([{'role': 'user', 'content': message(it, q)}], tokenize=False,
                                        add_generation_prompt=True) for it in its[s:s + batch_size]]
        enc = tok(msgs, return_tensors='pt', padding=True, add_special_tokens=False).to(m.device)
        lg = m.lm_head(m.model(**enc).last_hidden_state[:, -1]).float()     # last position only
        out += torch.sigmoid(lg[:, yes].logsumexp(-1) - lg[:, no].logsumexp(-1)).cpu().tolist()
    return out


def load_judge():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(JUDGE)
    tok.padding_side = 'left'
    return AutoModelForCausalLM.from_pretrained(JUDGE, dtype=torch.bfloat16).to('cuda').eval(), tok


def classify(m, tok, its):
    """Rules where the case is clear-cut, the judge only for what is genuinely fuzzy (pilot, 2026-10-28):
    empty -> no_answer; gold contained in answer -> correct; else judge 'correct?'; wrong yes/no -> yesno_wrong;
    judge 'refusal?'; wrong answer found in the question or passages -> concept_drift; else judge 'related?'."""
    n = len(its)
    ans = [_norm(it['answer'] or '') for it in its]
    gold = [_norm(it['gold']) for it in its]
    ctx = [_norm(' '.join(it['passages']) + ' ' + it['question']) for it in its]
    in_passages = [bool(a) and f' {a} ' in f' {c} ' for a, c in zip(ans, ctx)]
    lab = [None] * n
    p = {k: [None] * n for k in ('correct', 'refusal', 'related')}
    for i in range(n):
        if not ans[i]:
            lab[i] = 'no_answer'
        elif gold[i] and f' {gold[i]} ' in f' {ans[i]} ':
            lab[i] = 'correct'

    def ask(key, q, decide):
        idx = [i for i in range(n) if lab[i] is None]
        for i, v in zip(idx, yes_prob(m, tok, [its[i] for i in idx], q)):
            p[key][i] = round(v, 4)
            lab[i] = decide(i, v)

    ask('correct', Q_CORRECT, lambda i, v: 'correct' if v > 0.5 else None)
    for i in range(n):
        if lab[i] is None and gold[i] in ('yes', 'no'):
            lab[i] = 'yesno_wrong'
    refusal_text = ('none', 'none mentioned', 'not mentioned', 'unknown', 'n a', 'not specified', 'not stated')
    for i in range(n):
        if lab[i] is None and (ans[i] in refusal_text or ans[i].startswith(('there is no information',
                                                                              'no information', 'none of the'))):
            lab[i] = 'refusal'
    ask('refusal', Q_REFUSAL, lambda i, v: 'refusal' if v > 0.5 else None)
    for i in range(n):
        if lab[i] is None and in_passages[i]:
            lab[i] = 'concept_drift'
    ask('related', Q_RELATED, lambda i, v: 'concept_drift' if v > 0.5 else 'hallucination')
    return lab, p, in_passages


def label(model):
    path = OUT / model / 'labels.json'
    if path.exists():
        return
    its = items(model)
    m, tok = load_judge()
    lab, p, inp = classify(m, tok, its)
    dump(path, {'judge': JUDGE, 'labels': LABELS, 'rows': [
        {'split': it['split'], 'id': it['id'], 'f1': it['f1'], 'label': l, 'p_correct': p['correct'][i],
         'p_refusal': p['refusal'][i], 'p_related': p['related'][i], 'answer_in_passages': inp[i]}
        for i, (it, l) in enumerate(zip(its, lab))]})
    counts = {sp: {l: sum(1 for it, x in zip(its, lab) if it['split'] == sp and x == l) for l in LABELS}
              for sp in SPLITS}
    print(json.dumps({'model': model, 'counts': counts}), flush=True)


def spot(model, per=25):
    its = {(it['split'], it['id']): it for it in items(model)}
    rows = json.loads((OUT / model / 'labels.json').read_text())['rows']
    rng = random.Random(0)
    lines = [f'# Spot check: {model} (judge {JUDGE})', '']
    for lab in LABELS:
        pick = [r for r in rows if r['label'] == lab]
        lines += [f'## {lab} ({len(pick)} total; {min(per, len(pick))} shown)', '']
        for r in rng.sample(pick, min(per, len(pick))):
            it = its[(r['split'], r['id'])]
            lines.append(f"- **Q:** {it['question']}  \n  **Gold:** {it['gold']}  |  **Answer:** {it['answer']}  "
                         f"|  F1 {it['f1']:.2f}  |  in passages: {r['answer_in_passages']}")
        lines.append('')
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f'spot_check_{model}.md').write_text('\n'.join(lines))
    print('wrote', RESULTS / f'spot_check_{model}.md')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['label', 'spot'])
    ap.add_argument('--model', default='llama32_3b')
    a = ap.parse_args()
    label(a.model) if a.step == 'label' else spot(a.model)
