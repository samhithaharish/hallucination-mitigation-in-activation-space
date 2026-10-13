"""Correctability-map pilot (see docs/prereg_correctability_pilot.md).

    python scripts/correctability_pilot.py            # 120 questions, 9 layers
    python scripts/correctability_pilot.py --smoke    # 8 questions, 2 layers, 2 budgets
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from go_nogo import generate, is_correct, load_model, split  # noqa: E402

LAYERS = [2, 4, 8, 10, 13, 16, 20, 24, 27]
EPS = [0.05, 0.1, 0.2, 0.4, 0.8]
TRANSFER_EPS = [0.2, 0.4]


class Edit:
    """Adds a per-example vector delta (B, H) at the output of one block while active."""

    def __init__(self, model, layer):
        self.block, self.delta, self.h = model.model.layers[layer - 1], None, None

    def __enter__(self):
        def hook(m, i, o):
            d = self.delta[:, None, :].to(o[0].dtype if isinstance(o, tuple) else o.dtype)
            return (o[0] + d,) + o[1:] if isinstance(o, tuple) else o + d
        self.h = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


def answer_sep(tok, prompt):
    """Text to put between a chat prompt and a stored (stripped) answer so the answer is tokenized the way the
    model generates it. Mistral's template ends in '[/INST]' and its answers start with a '▁word' token; joined
    without the space, 'Ross' becomes 'R'+'oss', tokens Mistral never produces. Other models' templates end in a
    newline or their answers start without a space marker, so they get ''."""
    return " " if "mistral" in tok.name_or_path.lower() and not prompt[-1:].isspace() else ""


def batch_nll_inputs(tok, prompts, targets, device):
    ids, labels = [], []
    for p, t in zip(prompts, targets):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ti = tok(p + answer_sep(tok, p) + t, add_special_tokens=False)["input_ids"][len(pi):]
        ids.append(pi + ti)
        labels.append([-100] * len(pi) + ti)
    T = max(map(len, ids))
    pad = tok.pad_token_id
    x = torch.tensor([[pad] * (T - len(s)) + s for s in ids], device=device)
    y = torch.tensor([[-100] * (T - len(s)) + s for s in labels], device=device)
    m = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in ids], device=device)
    return x, y, m


def optimize_delta(model, x, y, m, layer, radius, steps, lr):
    B, H = x.shape[0], model.config.hidden_size
    delta = torch.zeros(B, H, device=x.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    with Edit(model, layer) as e:
        for _ in range(steps):
            e.delta = delta
            logits = model(input_ids=x, attention_mask=m).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y[:, 1:], ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                n = delta.norm(dim=1, keepdim=True).clamp_min(1e-8)
                delta.mul_(torch.clamp(radius / n, max=1.0))
    return delta.detach()


def generate_with(model, tok, prompts, layer, delta, args):
    with Edit(model, layer) as e:
        e.delta = delta
        return generate(model, tok, prompts, args.max_new_tokens, len(prompts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/go_nogo_llama3.2-3b")
    ap.add_argument("--out", default="results/correctability_pilot_llama3.2-3b")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    layers, eps = LAYERS, EPS
    if args.smoke:
        args.out, args.n, layers, eps = args.out + "_smoke", 8, [8, 24], [0.2, 0.8]
    os.makedirs(args.out, exist_ok=True)

    recs = json.load(open(os.path.join(args.src, "records.json")))
    tok_norm = np.load(os.path.join(args.src, "features.npz"))["tok_norm"]
    y = np.array([r["label"] for r in recs])
    tr, _, _ = split(y)
    rng = np.random.default_rng(1)
    idx = rng.choice([i for i in tr if y[i] == 1], size=args.n, replace=False)
    ctrl = rng.permutation(idx)                               # control target = another question's gold
    while np.any(ctrl == idx):
        ctrl = rng.permutation(idx)

    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").select(range(len(recs)))
    gold_val = {int(i): ds[int(i)]["answer"]["value"] for i in np.r_[idx, ctrl]}

    model, tok = load_model(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    dev = model.device
    prompts = [recs[i]["prompt"] for i in idx]

    # ---- knowledge status: 10 samples at T=1 ----
    hk_path = os.path.join(args.out, "hk.json")
    if os.path.exists(hk_path):
        hk = json.load(open(hk_path))
    else:
        hk = []
        with torch.no_grad():
            for s in range(0, len(idx), args.bs):
                P = prompts[s:s + args.bs]
                enc = tok(P, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=10,
                                   max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
                txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for j, i in enumerate(idx[s:s + args.bs]):
                    hk.append(sum(is_correct(t, recs[i]["gold"]) for t in txt[j * 10:(j + 1) * 10]))
        json.dump(hk, open(hk_path, "w"))
        print(f"[hk] knows(>=2)={sum(h >= 2 for h in hk)}  doesn't(0)={sum(h == 0 for h in hk)}", flush=True)

    # ---- per-layer, per-target, per-budget optimization ----
    res_path = os.path.join(args.out, "fix.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    for layer in layers:
        for target in ["gold", "control"]:
            tgt_idx = idx if target == "gold" else ctrl
            tgt_text = [gold_val[int(i)] for i in tgt_idx]
            tgt_golds = [recs[i]["gold"] for i in tgt_idx]
            for e in eps:
                key = f"{layer}:{target}:{e}"
                if key in res:
                    continue
                t = time.time()
                radius = float(e * tok_norm[layer])
                fixed, deltas = [], []
                for s in range(0, len(idx), args.bs):
                    x, yy, m = batch_nll_inputs(tok, prompts[s:s + args.bs], tgt_text[s:s + args.bs], dev)
                    d = optimize_delta(model, x, yy, m, layer, radius, args.steps, lr=radius / 10)
                    gens = generate_with(model, tok, prompts[s:s + args.bs], layer, d, args)
                    fixed += [int(is_correct(g, gl)) for g, gl in zip(gens, tgt_golds[s:s + args.bs])]
                    deltas.append(d.cpu())
                res[key] = fixed
                if target == "gold" and e in TRANSFER_EPS:
                    D = torch.cat(deltas).to(dev)
                    Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    cos = (Dn @ Dn.T).float().cpu().numpy()
                    res[f"{layer}:cos:{e}"] = float(cos[np.triu_indices(len(D), 1)].mean())
                    half = len(D) // 2
                    tf = []
                    for a, b in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
                        mu = Dn[a].mean(0)
                        mu = mu / mu.norm() * radius
                        Pb = prompts[b]
                        for s in range(0, len(Pb), args.bs):
                            gens = generate_with(model, tok, Pb[s:s + args.bs], layer,
                                                 mu.expand(len(Pb[s:s + args.bs]), -1), args)
                            gb = [recs[i]["gold"] for i in idx[b]][s:s + args.bs]
                            tf += [int(is_correct(g, gl)) for g, gl in zip(gens, gb)]
                    # order: second half first, then first half -> restore question order
                    res[f"{layer}:transfer:{e}"] = tf[len(D) - half:] + tf[:len(D) - half]
                json.dump(res, open(res_path, "w"))
                print(f"[fix] layer {layer:2d} {target:<7} eps {e:<4} fix {np.mean(fixed):.3f} "
                      f"({time.time() - t:.0f}s)", flush=True)

    analyse(res, hk, layers, eps, args)


def analyse(res, hk, layers, eps, args):
    from scipy.stats import mannwhitneyu
    D = json.load(open(os.path.join(args.src, "detect_mean.json")))["D"]
    hk = np.array(hk)
    n = len(hk)

    def mincost(layer, target):
        c = np.full(n, np.inf)
        for e in sorted(eps, reverse=True):
            c[np.array(res[f"{layer}:{target}:{e}"]) == 1] = e
        return c

    e_ref = 0.2 if 0.2 in eps else eps[0]
    fr = {l: np.array(res[f"{l}:gold:{e_ref}"]) for l in layers}
    S = max(layers, key=lambda l: fr[l].mean())
    B = np.random.default_rng(0).integers(0, n, size=(2000, n))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    out = {"D": D, "S_star": S, "eps_ref": e_ref,
           "fixrate_gold": {l: {e: float(np.mean(res[f"{l}:gold:{e}"])) for e in eps} for l in layers},
           "fixrate_control": {l: {e: float(np.mean(res[f"{l}:control:{e}"])) for e in eps} for l in layers},
           "cos": {l: {e: res.get(f"{l}:cos:{e}") for e in TRANSFER_EPS} for l in layers},
           "transfer_fixrate": {l: {e: (float(np.mean(res[f"{l}:transfer:{e}"])) if f"{l}:transfer:{e}" in res else None)
                                    for e in TRANSFER_EPS} for l in layers},
           "hk_counts": {"knows": int((hk >= 2).sum()), "doesnt": int((hk == 0).sum())}}
    if D in fr:
        out["H1_S_minus_D_ci"] = ci((fr[S] - fr[D])[B].mean(1))
        out["H1"] = bool(abs(S - D) >= 4 and out["H1_S_minus_D_ci"][0] > 0)
    ctl = np.array(res[f"{S}:control:{e_ref}"])
    out["H2_gold_minus_control_ci"] = ci((fr[S] - ctl)[B].mean(1))
    out["H2"] = bool(out["H2_gold_minus_control_ci"][0] > 0)
    c = mincost(S, "gold")
    kp, km = c[hk >= 2], c[hk == 0]
    if len(kp) and len(km):
        p = mannwhitneyu(kp, km, alternative="less").pvalue
        out.update({"H3_p": float(p), "H3": bool(p < 0.05),
                    "H3_median_cost_knows": float(np.median(kp)), "H3_median_cost_doesnt": float(np.median(km)),
                    "H3_fixrate_knows": float(np.mean(fr[S][hk >= 2])), "H3_fixrate_doesnt": float(np.mean(fr[S][hk == 0]))})
    cos_ok = all(v is None or v < 0.2 for l in layers for v in out["cos"][l].values())
    tr_ok = all(out["transfer_fixrate"][l][e] is None or
                out["transfer_fixrate"][l][e] < out["fixrate_gold"][l][e] / 3
                for l in layers for e in TRANSFER_EPS if e in eps)
    out["H4"] = bool(cos_ok and tr_ok)
    out["verdict"] = "GO" if out.get("H1") and out["H2"] else "NO-GO"
    json.dump(out, open(os.path.join(args.out, "verdict_pilot.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict)}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    det = json.load(open(os.path.join(args.src, "detect_mean.json")))["auroc_mean"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for e in eps:
        ax[0].plot(layers, [out["fixrate_gold"][l][e] for l in layers], marker="o", label=f"gold eps={e}")
        ax[0].plot(layers, [out["fixrate_control"][l][e] for l in layers], ls=":", color=ax[0].lines[-1].get_color())
    ax[0].axvline(D, color="k", ls=":", lw=0.8)
    ax[0].set(xlabel="layer", ylabel="fix rate", title="Correctability (solid gold, dotted control)")
    ax[0].legend(fontsize=7)
    ax2 = ax[1]
    ax2.plot(range(1, len(det)), det[1:], color="C0")
    ax2.set(xlabel="layer", ylabel="detection AUROC", title=f"Detection (D={D}) vs correction (S*={S})")
    ax3 = ax2.twinx()
    ax3.plot(layers, [out["fixrate_gold"][l][e_ref] for l in layers], color="C3", marker="o")
    ax3.set_ylabel(f"gold fix rate at eps={e_ref}", color="C3")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "correctability_map.png"), dpi=150)


if __name__ == "__main__":
    main()
