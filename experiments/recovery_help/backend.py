"""One baseline generation supplies both observable and activation features."""
import contextlib
import os
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import MODEL_PATHS

CONFIDENCE_NAMES = ['logp_mean', 'logp_min', 'logp_max', 'logp_first', 'logp_last',
                    'entropy_mean', 'entropy_first', 'entropy_last',
                    'margin_mean', 'margin_first', 'margin_last', 'answer_tokens', 'at_cap']


def load(key):
    torch.set_num_threads(4)
    torch.manual_seed(20261008)
    tok = AutoTokenizer.from_pretrained(MODEL_PATHS[key], local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATHS[key], local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tok


def chat(tok, prompt):
    return tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False,
                                   add_generation_prompt=True)


def layer_grid(model):
    n = model.config.num_hidden_layers
    # ACTIVATION_LAYERS=all captures every block output (layer-wise separability study);
    # unset keeps the six-depth grid the recorded experiments used.
    spec = os.environ.get('ACTIVATION_LAYERS')
    if spec == 'all':
        return list(range(1, n + 1))
    if spec:                                    # explicit comma list, e.g. "12,15,18"
        return sorted({int(x) for x in spec.split(',')})
    return sorted(set(max(1, min(n-1, round(n*f))) for f in (1/6, 1/3, 1/2, 2/3, 5/6, 11/12)))


def active_lengths(ids, stop_ids):
    mask = torch.ones_like(ids, dtype=torch.bool)
    for token in stop_ids:
        mask &= ids != token
    return mask.long().cumprod(dim=1).sum(dim=1)


@torch.inference_mode()
def generate_batch(model, tok, prompts, cap, capture=False):
    enc = tok(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
    layers = layer_grid(model)
    captures = {layer: [] for layer in layers}
    with contextlib.ExitStack() as stack:
        if capture:
            for layer in layers:
                def hook(module, inputs, output, layer=layer):
                    hidden = output[0] if isinstance(output, tuple) else output
                    captures[layer].append(hidden[:, -1].detach().half().cpu())
                handle = model.model.layers[layer-1].register_forward_hook(hook)
                stack.callback(handle.remove)
        generated = model.generate(**enc, max_new_tokens=cap, do_sample=False,
            temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id,
            return_dict_in_generate=True, output_scores=capture)
    ids = generated.sequences[:, enc.input_ids.shape[1]:]
    eos = model.generation_config.eos_token_id
    stop_ids = ([eos] if isinstance(eos, int) else list(eos or [])) + [tok.pad_token_id]
    lengths = active_lengths(ids, [s for s in stop_ids if s is not None])
    result = {'texts': tok.batch_decode(ids, skip_special_tokens=True), 'ids': ids.cpu().tolist(),
              'lengths': lengths.cpu().tolist(), 'input_lengths': enc.attention_mask.sum(1).cpu().tolist(),
              'at_cap': (lengths == cap).cpu().tolist(), 'layers': layers}
    if capture:
        steps = len(generated.scores)
        if any(len(captures[layer]) != steps for layer in layers):
            raise RuntimeError('Generation hook count differs from output steps')
        last = (lengths.cpu()-1).clamp(min=0)
        row = torch.arange(len(prompts))
        result['states'] = torch.stack([
            torch.stack([captures[layer][0], torch.stack(captures[layer])[last, row]], dim=1)
            for layer in layers], dim=1).numpy()
        per_step = []
        for t, logits in enumerate(generated.scores):
            logp = logits.float().log_softmax(-1)
            prob = logp.exp()
            chosen = logp.gather(1, ids[:, t:t+1]).squeeze(1)
            entropy = -torch.where(prob > 0, prob * logp, 0.).sum(-1)
            top = prob.topk(2, dim=-1).values
            per_step.append(torch.stack([chosen, entropy, top[:, 0]-top[:, 1]], dim=-1).cpu().numpy())
        values = np.stack(per_step, axis=1)
        features = []
        for i, length in enumerate(result['lengths']):
            v = values[i, :max(length, 1)]
            features.append([v[:, 0].mean(), v[:, 0].min(), v[:, 0].max(), v[0, 0], v[-1, 0],
                             v[:, 1].mean(), v[0, 1], v[-1, 1], v[:, 2].mean(), v[0, 2], v[-1, 2],
                             length, float(length == cap)])
        result['confidence'] = np.asarray(features, dtype=np.float32)
    return result
