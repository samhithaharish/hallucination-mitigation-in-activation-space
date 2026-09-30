"""Type-matched control for the knowledge-nearby claim (see docs/prereg_type_control.md).

    python scripts/type_control.py --key llama3.2-3b
    python scripts/type_control.py --aggregate
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from go_nogo import generate, is_correct, load_model, normalize  # noqa: E402
from correctability_pilot import batch_nll_inputs, generate_with, optimize_delta  # noqa: E402
from mixed_vs_separated import MODELS  # noqa: E402

SRC = "results/mixed_vs_separated"
CONDS = ["gold", "sampled_wrong", "prompted_wrong", "random"]
EPS = [0.05, 0.1]
MAX_WORDS = 6        # deviation 1: short control targets only (gold answers are short)


def hits(output, target):
    return is_correct(output, [target])


def run(key, args):
    out_dir = os.path.join(args.out, key)
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "type_control.json")
    if os.path.exists(res_path):
        print(f"[{key}] already done")
        return
    hf_id, size = MODELS[key]
    recs = json.load(open(os.path.join(SRC, key, "records.json")))
    y = np.array([r["label"] for r in recs])
    tr, _ = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    rng = np.random.default_rng(2)
    idx = rng.choice([i for i in tr if y[i] == 1], size=args.n, replace=False)

    model, tok = load_model(hf_id)
    for p in model.parameters():
        p.requires_grad_(False)
    dev, L = model.device, model.config.num_hidden_layers
    layer = int(round(0.45 * L))
    ebs = 6 if size == "large" else 12

    # prompted-wrong targets
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").select(range(len(recs)))
    ask = [tok.apply_chat_template([{"role": "user", "content":
                                     f"Question: {ds[int(i)]['question']}\nGive one plausible but incorrect answer. "
                                     f"Answer with only the answer, at most 5 words."}], tokenize=False, add_generation_prompt=True)
           for i in idx]
    prompted = generate(model, tok, ask, 24, 32)

    short = lambda t: t if t and len(t.split()) <= MAX_WORDS else None
    targets = {c: [] for c in CONDS}
    for j, i in enumerate(idx):
        r, g = recs[i], normalize(recs[i]["generation"])
        targets["gold"].append(r["value"])
        wrong = [s for s in r["samples"] if s and normalize(s) and normalize(s) != g and not is_correct(s, r["gold"])]
        wrong = [w for w in wrong if short(w)]
        targets["sampled_wrong"].append(collections.Counter(wrong).most_common(1)[0][0] if wrong else None)
        pw = prompted[j].strip()
        ok = pw and normalize(pw) and normalize(pw) != g and not is_correct(pw, r["gold"])
        targets["prompted_wrong"].append(short(pw) if ok else None)
        targets["random"].append(recs[idx[(j + 1) % len(idx)]]["value"])

    # median token norm at the edit layer (for the budget)
    with torch.no_grad():
        norms = []
        for s in range(0, len(idx), 32):
            enc = tok([recs[i]["prompt"] for i in idx[s:s + 32]], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            h = model(**enc, output_hidden_states=True).hidden_states[layer]
            mask = enc["attention_mask"].clone()
            mask[torch.arange(mask.shape[0]), mask.float().argmax(1)] = 0
            norms.append(h.float().norm(dim=-1)[mask.bool()].cpu().numpy())
    tok_norm = float(np.median(np.concatenate(norms)))

    res = {"key": key, "size": size, "layer": layer, "n": len(idx), "targets": targets,
           "target_len": {c: [len(tok(t, add_special_tokens=False)["input_ids"]) if t else None for t in targets[c]]
                          for c in CONDS}}
    t0 = time.time()
    for c in CONDS:
        have = [j for j, t in enumerate(targets[c]) if t]
        for e in EPS:
            radius = e * tok_norm
            hit = [None] * len(idx)
            for s in range(0, len(have), ebs):
                js = have[s:s + ebs]
                P = [recs[idx[j]]["prompt"] for j in js]
                T = [targets[c][j] for j in js]
                x, yy, m = batch_nll_inputs(tok, P, T, dev)
                d = optimize_delta(model, x, yy, m, layer, radius, 30, lr=radius / 10)
                gens = generate_with(model, tok, P, layer, d, args)
                for j, g, t in zip(js, gens, T):
                    hit[j] = int(hits(g, t))
            res[f"{c}:{e}"] = hit
            print(f"[{key}] {c:15s} eps {e}: n={len(have)} hit {np.mean([h for h in hit if h is not None]):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(res, open(res_path, "w"))


def aggregate(args):
    B = 2000
    out, k1 = {}, 0
    for key in MODELS:
        p = os.path.join(args.out, key, "type_control.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        row = {"size": r["size"]}
        for e in EPS:
            g = np.array([np.nan if v is None else v for v in r[f"gold:{e}"]], float)
            row[f"gold:{e}"] = float(np.nanmean(g))
            for c in CONDS[1:]:
                x = np.array([np.nan if v is None else v for v in r[f"{c}:{e}"]], float)
                both = ~np.isnan(g) & ~np.isnan(x)
                d = g[both] - x[both]
                bs = np.random.default_rng(0).integers(0, len(d), size=(B, len(d)))
                row[f"{c}:{e}"] = float(x[both].mean())
                row[f"gold_minus_{c}:{e}"] = float(d.mean())
                row[f"gold_minus_{c}:{e}_ci"] = [float(np.percentile(d[bs].mean(1), q)) for q in (2.5, 97.5)]
                row[f"n_{c}"] = int(both.sum())
                if c != "random":
                    lg = np.array([np.nan if v is None else v for v in r["target_len"]["gold"]], float)[both]
                    lc = np.array([np.nan if v is None else v for v in r["target_len"][c]], float)[both]
                    mt = np.abs(lg - lc) <= 1
                    row[f"len_matched_gold_minus_{c}:{e}"] = float(d[mt].mean()) if mt.any() else None
                    row[f"n_len_matched_{c}"] = int(mt.sum())
        row["K1"] = all(row[f"gold_minus_{c}:0.05_ci"][0] > 0 for c in ("sampled_wrong", "prompted_wrong"))
        k1 += row["K1"]
        out[key] = row
    agg = {"per_model": out, "K1_count": k1, "of": len(out), "K1_holds": k1 >= 8}
    json.dump(agg, open(os.path.join(args.out, "aggregate.json"), "w"), indent=1)
    print(f"{'model':14s} {'eps':4s} gold  samp_wrong  prompt_wrong  random   (gold-samp CI | gold-prompt CI)  K1")
    for k, r in out.items():
        for e in EPS:
            print(f"{k:14s} {e:<4} {r[f'gold:{e}']:.2f}  {r[f'sampled_wrong:{e}']:.2f} ({r['n_sampled_wrong']})  "
                  f"{r[f'prompted_wrong:{e}']:.2f} ({r['n_prompted_wrong']})  {r[f'random:{e}']:.2f}  "
                  f"{np.round(r[f'gold_minus_sampled_wrong:{e}_ci'], 2)} | {np.round(r[f'gold_minus_prompted_wrong:{e}_ci'], 2)}"
                  f"  {r['K1'] if e == 0.05 else ''}")
    print(f"K1: {k1}/{len(out)} -> {'holds' if agg['K1_holds'] else 'fails'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default="results/type_control")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()
    aggregate(args) if args.aggregate else run(args.key, args)


if __name__ == "__main__":
    main()
