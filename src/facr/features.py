"""Hidden-state extraction (mean-pooled over prompt tokens, every layer, disk-cached)."""
from __future__ import annotations

import os

import numpy as np
import torch


@torch.no_grad()
def extract_hidden_states(model, tokenizer, records: list[dict], cfg, device: str):
    """Return (hidden, labels) with hidden shape (n_samples, n_layers + 1, hidden_size).

    Index 0 along axis 1 is the embedding output; index k is the output of
    transformer block k-1. All layer indices in this repo use that convention.
    """
    os.makedirs(cfg.cache_dir, exist_ok=True)
    h_path = os.path.join(cfg.cache_dir, "hidden_states.npy")
    l_path = os.path.join(cfg.cache_dir, "labels.npy")

    if os.path.exists(h_path) and os.path.exists(l_path):
        hidden, labels = np.load(h_path), np.load(l_path)
        print(f"Loaded cached hidden states: {hidden.shape}")
        return hidden, labels

    hidden, labels = [], []
    for r in records:
        inputs = tokenizer(r["prompt"], return_tensors="pt").to(device)
        out = model(**inputs, output_hidden_states=True)
        reps = [h[0].mean(dim=0).float().cpu().numpy() for h in out.hidden_states]
        hidden.append(np.stack(reps))
        labels.append(r["label"])

    hidden, labels = np.stack(hidden), np.array(labels)
    np.save(h_path, hidden)
    np.save(l_path, labels)
    print(f"Extracted hidden states: {hidden.shape}")
    return hidden, labels
