"""Prior-work baselines for context selection (docs/prereg_prior_work_baselines.md). GPU steps:

    python -m experiments.layer_scaling.baselines forward --model llama32_3b   # prompt-end states, singles + pairs
    python -m experiments.layer_scaling.baselines bge                          # bge-reranker-v2-m3 paragraph scores
    python -m experiments.layer_scaling.baselines seakr --model llama32_3b      # SeaKR-style uncertainty per pair

Only the test questions (qi 500-999) of the context-selection pilot are used.
"""
import argparse
import json
import os

import numpy as np
import torch

from experiments.layer_scaling.collect import load
from experiments.layer_scaling.context_select import OUT as CTX_OUT, items
from experiments.paths import SCALE_RUN
from experiments.recovery_help.backend import active_lengths, chat
from experiments.recovery_help.data import dump
from experiments.recovery_scale.actions import DIRECT_CAP, prompt, public_example

OUT = CTX_OUT.parent / 'baselines'
AXIS = {'llama32_3b': 18, 'phi35': 22, 'qwen25_3b': 28}       # drift-study probe layers (prompt end)
TEST_Q = range(500, 1000)
BGE = 'BAAI/bge-reranker-v2-m3'


def test_rows():
    return json.loads((SCALE_RUN / 'data' / 'test.json').read_text())


def single_items():
    rows = test_rows()
    out = []
    for qi in TEST_Q:
        e = public_example(rows[qi])
        for i, p in enumerate(e['paragraphs']):
            out.append({'qi': qi, 'id': rows[qi]['id'], 'para': i,
                        'prompt': prompt(dict(e, paragraphs=[p]), 'keep', initial_k=1)})
    return out


def pair_items():
    return [it for it in items(1000) if it['qi'] in TEST_Q]


@torch.no_grad()
def prompt_end_states(model, tok, prompts, layers, bs=32):
    out = []
    for s in range(0, len(prompts), bs):
        enc = tok([chat(tok, p) for p in prompts[s:s + bs]], return_tensors='pt', padding=True,
                  add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states   # hs[l] = output of block l
        out.append(torch.stack([hs[l][:, -1] for l in layers], 1).half().cpu().numpy())
    return np.concatenate(out)


def forward(key):
    model, tok = load(key)
    L = model.config.num_hidden_layers
    layers = [L // 2, AXIS[key]]
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    for name, its in (('singles', single_items()), ('pairs', pair_items())):
        if (d / f'{name}.npz').exists():
            continue
        S = prompt_end_states(model, tok, [it['prompt'] for it in its], layers)
        keys = [[it['qi'], it['para']] if name == 'singles' else [it['qi'], *it['pair']] for it in its]
        np.savez_compressed(d / f'{name}.npz', states=S, layers=np.array(layers), keys=np.array(keys))
        print(json.dumps({'model': key, 'set': name, 'n': len(its), 'layers': layers}), flush=True)


@torch.no_grad()
def bge():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = OUT / 'bge_scores.json'
    if path.exists():
        return
    tok = AutoTokenizer.from_pretrained(BGE)
    m = AutoModelForSequenceClassification.from_pretrained(BGE, dtype=torch.float16).to('cuda').eval()
    rows = test_rows()
    scores = {}
    for qi in TEST_Q:
        e = public_example(rows[qi])
        enc = tok([[e['question'], p['title'] + '. ' + p['text']] for p in e['paragraphs']], padding=True,
                  truncation=True, max_length=512, return_tensors='pt').to('cuda')
        scores[qi] = m(**enc).logits.view(-1).float().cpu().tolist()
    dump(path, {'model': BGE, 'scores': {str(k): v for k, v in scores.items()}})
    print(json.dumps({'bge': len(scores)}), flush=True)


def inside_score(Z, alpha=1e-3):
    """INSIDE eigenscore: (1/K) sum log eig(Z^T J Z + alpha I), Z (K, d), J centres the feature dimension."""
    Z = Z.astype(np.float64)
    Zc = Z - Z.mean(axis=1, keepdims=True)                  # J_d: centre each embedding over its d features
    S = Zc @ Zc.T / Z.shape[1]
    ev = np.linalg.eigvalsh(S + alpha * np.eye(len(S)))
    return float(np.mean(np.log(np.clip(ev, 1e-12, None))))


@torch.no_grad()
def seakr(key, k=8, pairs_per_batch=12):
    model, tok = load(key)
    layer = model.config.num_hidden_layers // 2
    d = OUT / key
    d.mkdir(parents=True, exist_ok=True)
    path = d / 'seakr.json'
    done = json.loads(path.read_text()) if path.exists() else {'k': k, 'layer': layer, 'keys': [], 'score': []}
    its = pair_items()[len(done['keys']):]
    eos = model.generation_config.eos_token_id
    stops = [s for s in ([eos] if isinstance(eos, int) else list(eos or [])) + [tok.pad_token_id] if s is not None]
    for s in range(0, len(its), pairs_per_batch):
        chunk = its[s:s + pairs_per_batch]
        enc = tok([chat(tok, it['prompt']) for it in chunk], return_tensors='pt', padding=True,
                  add_special_tokens=False).to(model.device)
        torch.manual_seed(20261023 + s)
        seq = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9, top_k=0, max_new_tokens=DIRECT_CAP,
                             num_return_sequences=k, pad_token_id=tok.pad_token_id)
        P = enc.input_ids.shape[1]
        gen = seq[:, P:]
        n = active_lengths(gen, stops)                             # generated tokens before EOS/pad
        mask = torch.cat([enc.attention_mask.repeat_interleave(k, 0),
                          (torch.arange(gen.shape[1], device=gen.device)[None] < n[:, None] + 1).long()], 1)
        hs = model(input_ids=seq, attention_mask=mask, output_hidden_states=True).hidden_states[layer]
        last = (P + n - 1).clamp(min=P - 1)                        # last generated token (prompt end if empty)
        H = hs[torch.arange(len(seq)), last].float().cpu().numpy().reshape(len(chunk), k, -1)
        for it, Z in zip(chunk, H):
            done['keys'].append([it['qi'], *it['pair']])
            done['score'].append(inside_score(Z))
        if (s // pairs_per_batch) % 50 == 0:
            dump(path, done)
            print(json.dumps({'model': key, 'done': len(done['keys']), 'total': len(pair_items())}), flush=True)
    dump(path, done)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['forward', 'bge', 'seakr'])
    ap.add_argument('--model', default='llama32_3b')
    a = ap.parse_args()
    {'forward': lambda: forward(a.model), 'bge': bge, 'seakr': lambda: seakr(a.model)}[a.step]()
