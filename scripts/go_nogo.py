"""Pre-registered detect-layer vs steer-layer test (see docs/prereg_go_nogo.md).

    python scripts/go_nogo.py                         # full run, Llama-3.2-3B
    python scripts/go_nogo.py --n 64 --out results/go_nogo_smoke --layers 1,14,24   # smoke test

Every stage is cached under --out, so a crashed run resumes where it stopped.
Layer convention: hidden_states index k = output of block k-1; steering at k hooks block k-1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import string
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

INSTRUCTION = "Answer with only the answer, no explanation."
ALPHAS = [0.05, 0.1, 0.2, 0.4]


# ---------- labelling ----------

def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(answer: str, golds: list[str]) -> bool:
    a = f" {normalize(answer)} "
    return any(g and f" {g} " in a for g in (normalize(x) for x in golds))


# ---------- model helpers ----------

def load_model(hf_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return model, tok


def build_prompt(tok, q):
    msgs = [{"role": "user", "content": f"{INSTRUCTION}\nQuestion: {q}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens, bs):
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           temperature=None, top_p=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


# ---------- stages ----------

def stage_label(model, tok, args):
    path = os.path.join(args.out, "records.json")
    if os.path.exists(path):
        return json.load(open(path))
    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").select(range(args.n))
    prompts = [build_prompt(tok, ex["question"]) for ex in ds]
    t = time.time()
    gens = generate(model, tok, prompts, args.max_new_tokens, args.bs)
    recs = []
    for ex, p, g in zip(ds, prompts, gens):
        golds = list(dict.fromkeys(ex["answer"]["aliases"] + [ex["answer"]["value"]]))
        recs.append({"question": ex["question"], "prompt": p, "gold": golds, "generation": g,
                     "label": 0 if is_correct(g, golds) else 1})
    json.dump(recs, open(path, "w"))
    fr = 1 - np.mean([r["label"] for r in recs])
    print(f"[label] {len(recs)} records in {time.time() - t:.0f}s, factual rate {fr:.3f}")
    return recs


@torch.no_grad()
def stage_features(model, tok, recs, args):
    path = os.path.join(args.out, "features.npz")
    if os.path.exists(path):
        z = np.load(path)
        return z["mean"], z["last"], z["tok_norm"]
    L = model.config.num_hidden_layers + 1
    means, lasts, norms = [], [], []
    prompts = [r["prompt"] for r in recs]
    for i in range(0, len(prompts), args.bs):
        enc = tok(prompts[i:i + args.bs], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states  # L x (B, T, H)
        mask = enc["attention_mask"].clone()
        first = mask.float().argmax(dim=1)                     # left padding: first real token = BOS
        mask[torch.arange(mask.shape[0]), first] = 0
        m = mask.unsqueeze(-1).float()
        H = torch.stack(hs, 1).float()                          # (B, L, T, H)
        means.append(((H * m.unsqueeze(1)).sum(2) / m.sum(1).unsqueeze(1)).half().cpu().numpy())
        lasts.append(H[:, :, -1].half().cpu().numpy())
        tn = H.norm(dim=-1)                                     # (B, L, T)
        norms.append(torch.stack([tn[:, l][mask.bool()] for l in range(L)]).cpu().numpy())  # (L, ntok)
    mean, last = np.concatenate(means), np.concatenate(lasts)
    tok_norm = np.median(np.concatenate(norms, axis=1), axis=1)
    np.savez(path, mean=mean, last=last, tok_norm=tok_norm)
    print(f"[features] mean {mean.shape}")
    return mean, last, tok_norm


def split(labels, seed=0):
    idx = np.arange(len(labels))
    tr, rest = train_test_split(idx, test_size=0.30, random_state=seed, stratify=labels)
    tu, te = train_test_split(rest, test_size=0.50, random_state=seed, stratify=labels[rest])
    return tr, tu, te


def _fold_auroc(X, y, tr, te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    clf.fit(X[tr], y[tr])
    return roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1])


def stage_detect(feats, labels, tr_idx, name, args):
    path = os.path.join(args.out, f"detect_{name}.json")
    if os.path.exists(path):
        return json.load(open(path))
    X, y = feats[tr_idx].astype(np.float32), labels[tr_idx]
    cv = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0).split(X[:, 0], y))
    L = X.shape[1]
    jobs = [(l, f) for l in range(L) for f in range(len(cv))]
    res = Parallel(n_jobs=args.jobs)(delayed(_fold_auroc)(X[:, l], y, *cv[f]) for l, f in jobs)
    A = np.array(res).reshape(L, len(cv))                       # (layers, 20 folds)
    per_repeat = A.reshape(L, 4, 5).mean(2)                     # (layers, 4 repeats)
    out = {"auroc_mean": A.mean(1).tolist(), "auroc_std": A.std(1).tolist(),
           "D": int(np.argmax(A.mean(1)[1:]) + 1),
           "argmax_per_repeat": [int(np.argmax(per_repeat[1:, r]) + 1) for r in range(4)]}
    json.dump(out, open(path, "w"))
    print(f"[detect/{name}] D={out['D']}  per-repeat argmax={out['argmax_per_repeat']}  "
          f"AUROC@D={out['auroc_mean'][out['D']]:.3f}")
    return out


def steering_vectors(feats, labels, tr_idx, tok_norm, device):
    X, y = feats[tr_idx].astype(np.float32), labels[tr_idx]
    v = X[y == 0].mean(0) - X[y == 1].mean(0)                   # (L, H)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    v *= tok_norm[:, None]
    return torch.tensor(v, dtype=torch.bfloat16, device=device)


def steered_correct(model, tok, recs, idx, layer, vec, alpha, args):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            return (out[0] + alpha * vec,) + out[1:]
        return out + alpha * vec
    h = model.model.layers[layer - 1].register_forward_hook(hook)
    try:
        gens = generate(model, tok, [recs[i]["prompt"] for i in idx], args.max_new_tokens, args.bs)
    finally:
        h.remove()
    return [int(is_correct(g, recs[i]["gold"])) for g, i in zip(gens, idx)]


def stage_steer(model, tok, recs, idx, V, layers, alphas_for, split_name, args):
    """alphas_for(layer) -> list of alphas. Results appended to a JSON cache per (layer, alpha)."""
    path = os.path.join(args.out, f"steer_{split_name}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for l in layers:
        for a in alphas_for(l):
            k = f"{l}:{a}"
            if k in res:
                continue
            t = time.time()
            res[k] = steered_correct(model, tok, recs, idx, l, V[l], a, args)
            json.dump(res, open(path, "w"))
            print(f"[steer/{split_name}] layer {l:2d} alpha {a:<4} acc {np.mean(res[k]):.3f} "
                  f"({time.time() - t:.0f}s)", flush=True)
    return res


# ---------- analysis ----------

def analyse(recs, det, det_last, tune, test, tu_idx, te_idx, layers, args):
    base_tu = np.array([1 - recs[i]["label"] for i in tu_idx])
    base_te = np.array([1 - recs[i]["label"] for i in te_idx])
    best_alpha, tune_gain = {}, {}
    for l in layers:
        g = {a: np.mean(tune[f"{l}:{a}"]) - base_tu.mean() for a in ALPHAS}
        best_alpha[l] = max(g, key=g.get)
        tune_gain[l] = g[best_alpha[l]]
    S = max(tune_gain, key=tune_gain.get)
    D = det["D"]
    C = {l: np.array(test[f"{l}:{best_alpha[l]}"]) for l in layers}
    gain = {l: C[l].mean() - base_te.mean() for l in layers}

    rng = np.random.default_rng(0)
    B = rng.integers(0, len(te_idx), size=(2000, len(te_idx)))
    diffs = {l: C[l] - base_te for l in layers}
    boot_S = diffs[S][B].mean(1)
    boot_SD = (diffs[S] - diffs[D])[B].mean(1) if D in diffs else None
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    boot_argmax = np.array(layers)[np.argmax(np.stack([diffs[l][B].mean(1) for l in layers]), 0)]

    c1 = ci(boot_S)[0] > 0
    c2 = abs(S - D) >= 4
    c3 = boot_SD is not None and ci(boot_SD)[0] > 0
    out = {
        "n_records": len(recs), "factual_rate": float(1 - np.mean([r["label"] for r in recs])),
        "D": D, "D_argmax_per_repeat": det["argmax_per_repeat"], "D_last_token": det_last["D"],
        "S": S, "alpha_S": best_alpha[S], "alpha_D": best_alpha.get(D),
        "baseline_test_acc": float(base_te.mean()),
        "gain_S": float(gain[S]), "gain_S_ci": ci(boot_S),
        "gain_D": float(gain.get(D, np.nan)),
        "gain_S_minus_D_ci": ci(boot_SD) if boot_SD is not None else None,
        "test_argmax_bootstrap_top": {int(k): int(v) for k, v in
                                      zip(*np.unique(boot_argmax, return_counts=True))},
        "tune_gain": {int(l): float(tune_gain[l]) for l in layers},
        "test_gain": {int(l): float(gain[l]) for l in layers},
        "best_alpha": {int(l): best_alpha[l] for l in layers},
        "criteria": {"1_steering_helps": c1, "2_far_apart": c2, "3_S_beats_D": c3},
        "verdict": "GO" if (c1 and c2 and c3) else "NO-GO",
    }
    json.dump(out, open(os.path.join(args.out, "verdict.json"), "w"), indent=1)
    plot(det, det_last, out, layers, args)
    return out


def plot(det, det_last, out, layers, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(det["auroc_mean"]))
    m, s = np.array(det["auroc_mean"]), np.array(det["auroc_std"])
    ax[0].plot(x[1:], m[1:], label="mean-pooled (primary)")
    ax[0].fill_between(x[1:], (m - s)[1:], (m + s)[1:], alpha=0.25)
    ax[0].plot(x[1:], np.array(det_last["auroc_mean"])[1:], ls="--", label="last token")
    ax[0].axvline(out["D"], color="k", lw=0.8)
    ax[0].set(xlabel="layer", ylabel="probe AUROC (5x4 CV, train split)", title=f"Detection: D = {out['D']}")
    ax[0].legend(fontsize=8)
    ax[1].plot(layers, [out["tune_gain"][l] * 100 for l in layers], label="tune (alpha chosen here)")
    ax[1].plot(layers, [out["test_gain"][l] * 100 for l in layers], label="test (held out)")
    ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].axvline(out["S"], color="k", lw=0.8)
    ax[1].axvline(out["D"], color="k", lw=0.8, ls=":")
    ax[1].set(xlabel="steer layer", ylabel="accuracy gain (pts)",
              title=f"Correction: S = {out['S']}  ({out['verdict']})")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "detect_vs_steer.png"), dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="results/go_nogo_llama3.2-3b")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--layers", default=None, help="comma list; default all 1..L")
    ap.add_argument("--stop-after", choices=["label", "detect", "tune"], default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    model, tok = load_model(args.model)
    recs = stage_label(model, tok, args)
    labels = np.array([r["label"] for r in recs])
    fr = 1 - labels.mean()
    if not 0.25 <= fr <= 0.75:
        raise SystemExit(f"Sanity gate failed: factual rate {fr:.3f} outside [0.25, 0.75]. Fix labelling.")
    if args.stop_after == "label":
        return

    mean, last, tok_norm = stage_features(model, tok, recs, args)
    tr, tu, te = split(labels)
    det = stage_detect(mean, labels, tr, "mean", args)
    det_last = stage_detect(last, labels, tr, "last", args)
    if args.stop_after == "detect":
        return

    L = model.config.num_hidden_layers
    layers = [int(x) for x in args.layers.split(",")] if args.layers else list(range(1, L + 1))
    V = steering_vectors(mean, labels, tr, tok_norm, model.device)
    tune = stage_steer(model, tok, recs, tu, V, layers, lambda l: ALPHAS, "tune", args)
    if args.stop_after == "tune":
        return

    best = {l: max(ALPHAS, key=lambda a: np.mean(tune[f"{l}:{a}"])) for l in layers}
    test_layers = sorted(set(layers) | ({det["D"]} if det["D"] >= 1 else set()))
    for l in test_layers:  # D may be outside --layers in smoke runs
        if l not in best:
            tune.update(stage_steer(model, tok, recs, tu, V, [l], lambda _: ALPHAS, "tune", args))
            best[l] = max(ALPHAS, key=lambda a: np.mean(tune[f"{l}:{a}"]))
    test = stage_steer(model, tok, recs, te, V, test_layers, lambda l: [best[l]], "test", args)
    out = analyse(recs, det, det_last, tune, test, tu, te, test_layers, args)
    print(json.dumps({k: out[k] for k in ["factual_rate", "D", "D_argmax_per_repeat", "D_last_token", "S",
                                          "baseline_test_acc", "gain_S", "gain_S_ci", "gain_D",
                                          "gain_S_minus_D_ci", "criteria", "verdict"]}, indent=1))


if __name__ == "__main__":
    main()
