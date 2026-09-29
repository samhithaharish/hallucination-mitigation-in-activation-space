"""CAA steering vectors, forward hook, and steering evaluation."""
from __future__ import annotations

import numpy as np
import torch

from .data import best_f1


def get_blocks(model):
    """Transformer blocks for Llama / Qwen / Mistral style HF models."""
    return model.model.layers


def build_steering_vectors(hidden: np.ndarray, labels: np.ndarray, device: str,
                           dtype=torch.bfloat16, normalize: bool = False) -> dict[int, torch.Tensor]:
    """CAA vector per layer = mean(factual activations) - mean(hallucinated activations).

    Adding it pushes the residual stream toward the factual cluster.

    normalize=False reproduces the original notebooks (raw difference). Raw norms
    grow with depth, so one fixed alpha is a larger relative perturbation at some
    layers than others; normalize=True rescales each vector to unit norm so alpha
    is comparable across layers.
    """
    vectors = {}
    for layer in range(hidden.shape[1]):
        X = hidden[:, layer, :]
        v = X[labels == 0].mean(axis=0) - X[labels == 1].mean(axis=0)
        if normalize:
            v = v / (np.linalg.norm(v) + 1e-8)
        vectors[layer] = torch.tensor(v, dtype=dtype, device=device)
    return vectors


def make_steer_hook(vector: torch.Tensor, alpha: float):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            h = out[0] + alpha * vector.to(out[0].dtype)
            return (h,) + out[1:]
        return out + alpha * vector.to(out.dtype)
    return hook


@torch.no_grad()
def steered_generate(model, tokenizer, prompt, steer_layer, vector, alpha, device, max_new_tokens=32):
    """Greedy generation with `vector` added at hidden_states index `steer_layer`
    (= output of block steer_layer - 1). steer_layer 0 (embeddings) is skipped."""
    handle = None
    if steer_layer > 0:
        handle = get_blocks(model)[steer_layer - 1].register_forward_hook(
            make_steer_hook(vector, alpha))
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        if handle is not None:
            handle.remove()


def eval_steering_accuracy(model, tokenizer, records, steering_vectors, steer_layer,
                           eval_idx, cfg, device) -> float:
    """Fraction of eval examples whose *steered* answer matches gold (token-F1 >= threshold).

    NOTE: the original notebooks scored 'correct' as "steered output differs from the
    original generation" (see the placeholder comment in the 8B notebook). That rewards
    any change, including changes to a wrong answer, so it is not accuracy. This version
    scores against gold aliases, which records.json now stores.
    """
    correct = 0
    for i in eval_idx:
        r = records[int(i)]
        gen = steered_generate(model, tokenizer, r["prompt"], steer_layer,
                               steering_vectors[steer_layer], cfg.alpha, device,
                               cfg.max_new_tokens)
        correct += int(best_f1(gen, r["gold"]) >= cfg.f1_threshold)
    return correct / len(eval_idx)


def baseline_accuracy(records, eval_idx, cfg) -> float:
    """Unsteered accuracy on the same examples (label 0 = factual), for the delta."""
    return float(np.mean([records[int(i)]["label"] == 0 for i in eval_idx]))


def steer_layer_sweep(model, tokenizer, records, steering_vectors, eval_idx, cfg, device):
    accs = []
    for layer in range(1, len(steering_vectors)):
        acc = eval_steering_accuracy(model, tokenizer, records, steering_vectors,
                                     layer, eval_idx, cfg, device)
        accs.append(acc)
        print(f"steer layer {layer:2d}  accuracy {acc:.3f}")
    accs = np.array(accs)
    steer_layer = int(np.argmax(accs)) + 1  # sweep starts at layer 1
    return accs, steer_layer
