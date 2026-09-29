"""CPU smoke tests on a tiny random Llama (no downloads).

    python tests/test_steering.py

Checks: F1 helpers, hidden-state shapes, that the steering hook changes activations from
the hooked layer onward (and not before), that steer_layer=0 is a no-op, and that hooks are
removed afterwards. It does not check that steering *improves* anything; that needs real models.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM

from facr.config import RunConfig
from facr.data import best_f1, token_f1
from facr.features import extract_hidden_states
from facr.steering import build_steering_vectors, steered_generate

torch.manual_seed(0)
N_LAYERS, HID = 4, 32


class FakeTok:
    """Whitespace tokenizer over a fixed hash vocab, just enough for generate()."""
    def __call__(self, text, return_tensors="pt"):
        ids = [1 + (hash(w) % 200) for w in text.split()] or [1]
        t = torch.tensor([ids])
        return BatchEncoding({"input_ids": t, "attention_mask": torch.ones_like(t)})

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


model = LlamaForCausalLM(LlamaConfig(
    vocab_size=256, hidden_size=HID, intermediate_size=64, num_hidden_layers=N_LAYERS,
    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)).eval()
tok, dev = FakeTok(), "cpu"

# --- F1 helpers
assert token_f1("Eiffel Tower", "eiffel tower") == 1.0
# normalize() does not strip articles (same as the original notebooks), so "the" costs precision
assert abs(token_f1("The Eiffel Tower", "eiffel tower") - 0.8) < 1e-9
assert token_f1("Paris", "London") == 0.0
assert best_f1("Paris France", ["London", "Paris"]) > 0.6

# --- hidden-state extraction shape + cache
cfg = RunConfig(model_key="llama3.2-3b", cache_root=tempfile.mkdtemp())
recs = [{"prompt": f"Question: q{i} ?\nAnswer:", "label": i % 2} for i in range(6)]
hidden, labels = extract_hidden_states(model, tok, recs, cfg, dev)
assert hidden.shape == (6, N_LAYERS + 1, HID), hidden.shape

# --- steering vectors: factual mean - hallucinated mean
vecs = build_steering_vectors(hidden, labels, dev, dtype=torch.float32)
expect = hidden[labels == 0, 2].mean(0) - hidden[labels == 1, 2].mean(0)
assert np.allclose(vecs[2].numpy(), expect, atol=1e-5)
unit = build_steering_vectors(hidden, labels, dev, dtype=torch.float32, normalize=True)
assert abs(float(unit[2].norm()) - 1.0) < 1e-4

# --- hook effect: steering block k-1 (steer_layer=k) must change hidden_states[k:] only
ids = tok("Question: q ?\nAnswer:")["input_ids"]
with torch.no_grad():
    base = model(ids, output_hidden_states=True).hidden_states
K = 2
big = torch.ones(HID) * 5.0
h = model.model.layers[K - 1].register_forward_hook(
    lambda m, i, o: (o[0] + big,) + o[1:] if isinstance(o, tuple) else o + big)
with torch.no_grad():
    steered = model(ids, output_hidden_states=True).hidden_states
h.remove()
for layer in range(N_LAYERS + 1):
    same = torch.allclose(base[layer], steered[layer], atol=1e-6)
    if layer < K:
        assert same, f"hidden_states[{layer}] changed by a hook on a later block"
    elif layer > K:
        assert not same, f"hidden_states[{layer}] unaffected by steering at block {K - 1}"
    # layer == K is deliberately unchecked: newer transformers record hidden_states[K] with
    # their own hook *before* ours runs, older versions record it after. Either way the
    # steering reaches every later block, which is what the experiment relies on.

# --- steered_generate: layer 0 is a no-op, layer>0 changes output, hooks are cleaned up
prompt = "Question: q ?\nAnswer:"
# transformers itself may register hooks on blocks (for output_hidden_states), so compare counts
hooks_before = [len(b._forward_hooks) for b in model.model.layers]
plain = steered_generate(model, tok, prompt, 0, big, 4.0, dev, max_new_tokens=6)
again = steered_generate(model, tok, prompt, 0, big, 4.0, dev, max_new_tokens=6)
assert plain == again
steer = steered_generate(model, tok, prompt, 2, big * 20, 4.0, dev, max_new_tokens=6)
assert steer != plain, "large steering vector should change greedy output"
assert [len(b._forward_hooks) for b in model.model.layers] == hooks_before, "hook leaked"
after = steered_generate(model, tok, prompt, 0, big, 4.0, dev, max_new_tokens=6)
assert after == plain

print("all smoke tests passed")
