"""TriviaQA generation and factuality labelling (token-F1 against gold aliases).

Label convention used everywhere in this repo:
    label 1 = hallucinated, label 0 = factual.

Note: the dataset id must be namespaced ("mandarjoshi/trivia_qa"); the bare
"trivia_qa" id is rejected by recent huggingface_hub versions.
"""
from __future__ import annotations

import collections
import json
import os
import re

import torch


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision, recall = n_same / len(p), n_same / len(g)
    return 2 * precision * recall / (precision + recall)


def best_f1(pred: str, golds: list[str]) -> float:
    return max(token_f1(pred, g) for g in golds)


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 32) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def build_records(model, tokenizer, cfg, device: str) -> list[dict]:
    """Generate answers for TriviaQA and label them. Cached to disk as records.json.

    Each record keeps its gold aliases so steering can later be scored against
    ground truth rather than against the model's own original output.
    """
    from datasets import load_dataset

    os.makedirs(cfg.cache_dir, exist_ok=True)
    path = os.path.join(cfg.cache_dir, "records.json")
    if os.path.exists(path):
        with open(path) as f:
            records = json.load(f)
        print(f"Loaded {len(records)} cached records.")
        return records

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    ds = ds.select(range(cfg.n_samples))

    records = []
    for ex in ds:
        q = ex["question"]
        golds = ex["answer"]["aliases"] + [ex["answer"]["value"]]
        prompt = f"Question: {q}\nAnswer:"
        gen = greedy_generate(model, tokenizer, prompt, device, cfg.max_new_tokens)
        f1 = best_f1(gen, golds)
        records.append({
            "question": q,
            "prompt": prompt,
            "gold": golds,
            "generation": gen,
            "f1": f1,
            "label": 1 if f1 < cfg.f1_threshold else 0,
        })

    with open(path, "w") as f:
        json.dump(records, f)
    n_h = sum(r["label"] for r in records)
    print(f"Generated + labelled {len(records)} records. "
          f"Hallucination rate: {n_h}/{len(records)} ({100 * n_h / len(records):.1f}%)")
    return records
