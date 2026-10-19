"""Layer-wise collection on Som's recovery_scale cohorts: every layer, more backbones, one GPU.

Reuses recovery_scale.collect.collect_batch unchanged (same prompts, ten actions, scoring), with a
Transformers-only engine so a 24 GB card can run 1.5-8B models. States are captured at every block
output (ACTIVATION_LAYERS=all) at the prompt end and at the last answer token.

    PYTHONHASHSEED=0 python -m experiments.layer_scaling.collect --model phi35 --split train --limit 3000
"""
import argparse
import json
import os
import time

os.environ.setdefault('ACTIVATION_LAYERS', 'all')

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.paths import RUNS, SCALE_RUN
from experiments.recovery_help.data import dump, digest, file_digest
from experiments.recovery_scale.backend import generate
from experiments.recovery_scale.collect import collect_batch
from experiments.recovery_scale.data import SEED

OUT = RUNS / 'layer_scaling'
# key -> (hub id, pinned revision or None for the cached snapshot)
MODELS = {
    'phi35': ('microsoft/Phi-3.5-mini-instruct', '2fe192450127e6a83f7441aef6e3ca586c338b77'),
    'llama32_3b': ('meta-llama/Llama-3.2-3B-Instruct', None),
    'qwen25_3b': ('Qwen/Qwen2.5-3B-Instruct', None),
    'qwen25_7b': ('Qwen/Qwen2.5-7B-Instruct', 'a09a35458c702b33eeacc393d103063234e8bc28'),
    'llama31_8b': ('meta-llama/Llama-3.1-8B-Instruct', None),
    'olmo2_7b': ('allenai/OLMo-2-1124-7B-Instruct', None),
}


def load(key):
    hub, rev = MODELS[key]
    torch.set_num_threads(4)
    torch.manual_seed(20261008)
    tok = AutoTokenizer.from_pretrained(hub, revision=rev)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hub, revision=rev, dtype=torch.bfloat16,
                                                 attn_implementation='sdpa').to('cuda').eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def run(key, split, limit, batch_size):
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Run with PYTHONHASHSEED=0 for stable BM25 accumulation')
    raw = json.loads((SCALE_RUN / 'data' / f'{split}.json').read_text())[:limit]
    directory = OUT / key / split
    directory.mkdir(parents=True, exist_ok=True)
    proto = {'model': MODELS[key], 'split': split, 'limit': limit, 'batch_size': batch_size,
             'layers': 'all', 'engine': 'transformers', 'seed': SEED,
             'data_hash': file_digest(SCALE_RUN / 'data' / f'{split}.json')}
    if (directory / 'protocol.json').exists() and json.loads((directory / 'protocol.json').read_text()) != proto:
        raise ValueError('Protocol changed; use a fresh ACTIVATION_RUNS directory')
    dump(directory / 'protocol.json', proto)
    model, tok = load(key)
    k = 1 if split == 'transfer' else 2
    t0 = time.monotonic()
    for start in range(0, len(raw), batch_size):
        stem = directory / f'batch_{start:05d}'
        if stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
            continue
        chunk = raw[start:start + batch_size]
        rows, arrays = collect_batch(model, tok, chunk, k, SEED + start, generate=generate)
        layers = arrays.pop('layers')
        tmp = stem.with_suffix('.npz.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f, **arrays)
        tmp.replace(stem.with_suffix('.npz'))
        dump(stem.with_suffix('.json'), {'protocol': digest(proto), 'layers': layers, 'rows': rows})
        print(json.dumps({'model': key, 'split': split, 'done': start + len(chunk), 'total': len(raw),
                          'seconds': round(time.monotonic() - t0)}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--batch-size', type=int, default=32)
    a = ap.parse_args()
    run(a.model, a.split, a.limit, a.batch_size)
