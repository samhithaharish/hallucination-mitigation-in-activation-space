"""Reuse audited greedy capture; seeded sampled generations for the voting action."""
import torch
from experiments.recovery_help.backend import generate_batch, active_lengths, layer_grid, load as load_baseline
from .data import MODEL_PATHS, SEED


class Hybrid:
    def __init__(self, baseline, recovery):
        self.baseline=baseline;self.recovery=recovery


def load(key):
    from vllm import LLM
    baseline,tok=load_baseline(key)
    recovery=LLM(model=MODEL_PATHS[key],dtype='bfloat16',max_model_len=4096,
        kv_cache_memory_bytes=16<<30,gpu_memory_utilization=.5,max_num_seqs=64,
        max_num_batched_tokens=4096,enforce_eager=True,enable_prefix_caching=True,
        seed=SEED,disable_log_stats=True)
    return Hybrid(baseline,recovery),tok


def unpack_fast(outputs,cap,stop_ids):
    lengths=[]
    for row in outputs:
        ids=row.outputs[0].token_ids
        lengths.append(next((i for i,t in enumerate(ids) if t in stop_ids),len(ids)))
    return {'texts':[r.outputs[0].text for r in outputs],'lengths':lengths,
        'input_lengths':[len(r.prompt_token_ids) for r in outputs],
        'at_cap':[n==cap for n in lengths]}


@torch.inference_mode()
def generate(model, tok, prompts, cap, capture=False, seed=None):
    if isinstance(model,Hybrid):
        if capture:
            return generate_batch(model.baseline,tok,prompts,cap,True)
        from vllm import SamplingParams
        params=[SamplingParams(temperature=.7 if seed is not None else 0.,top_p=.9 if seed is not None else 1.,
            top_k=-1,max_tokens=cap,seed=seed+i if seed is not None else None) for i in range(len(prompts))]
        inputs=[{'prompt_token_ids':tok.encode(p,add_special_tokens=False)} for p in prompts]
        output=model.recovery.generate(inputs,params,use_tqdm=False)
        eos=model.baseline.generation_config.eos_token_id
        stops=([eos] if isinstance(eos,int) else list(eos or []))+[tok.pad_token_id]
        return unpack_fast(output,cap,set(s for s in stops if s is not None))
    if seed is None:
        return generate_batch(model, tok, prompts, cap, capture)
    if capture:
        raise ValueError('Baseline features must come from ordinary greedy execution')
    enc = tok(prompts, padding=True, add_special_tokens=False, return_tensors='pt').to(model.device)
    with torch.random.fork_rng(devices=[model.device.index]):
        torch.manual_seed(seed)
        ids = model.generate(**enc, max_new_tokens=cap, do_sample=True, temperature=.7,
            top_p=.9, top_k=0, pad_token_id=tok.pad_token_id)[:, enc.input_ids.shape[1]:]
    eos = model.generation_config.eos_token_id
    stops = ([eos] if isinstance(eos,int) else list(eos or [])) + [tok.pad_token_id]
    lengths = active_lengths(ids,[s for s in stops if s is not None])
    return {'texts':tok.batch_decode(ids,skip_special_tokens=True), 'lengths':lengths.cpu().tolist(),
            'input_lengths':enc.attention_mask.sum(1).cpu().tolist(), 'at_cap':(lengths==cap).cpu().tolist(),
            'layers':layer_grid(model)}
