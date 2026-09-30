"""Learned corrector, stage 1 (see docs/prereg_corrector_stage1.md).

    python scripts/corrector_stage1.py
    python scripts/corrector_stage1.py --smoke
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
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from go_nogo import generate, is_correct, load_model, split  # noqa: E402

READ = WRITE = 13


# ---------- interventions ----------

class Corrector(nn.Module):
    """delta = eps * n * tanh(|u|) * u/|u|, u = MLP(h_read). fixed=True: u is one learned vector."""

    def __init__(self, H, eps, norm, fixed=False, hidden=256):
        super().__init__()
        self.scale, self.fixed = eps * norm, fixed
        if fixed:
            self.u = nn.Parameter(torch.randn(H) * 1e-3)
        else:
            self.net = nn.Sequential(nn.Linear(H, hidden), nn.GELU(), nn.Linear(hidden, H))
            nn.init.normal_(self.net[2].weight, std=1e-3)
            nn.init.zeros_(self.net[2].bias)

    def forward(self, h):                                   # h: (B, H) float32
        u = self.u.expand(h.shape[0], -1) if self.fixed else self.net(h)
        n = u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return self.scale * torch.tanh(n) * u / n


class Hook:
    """Reads the layer output at read_pos, adds corrector(read) at every position.
    At generation, delta is computed on the prefill call (seq len > 1) and reused for decode steps."""

    def __init__(self, model, corr):
        self.corr, self.read_pos, self.delta = corr, None, None
        self.h = model.model.layers[WRITE - 1].register_forward_hook(self.fn)

    def fn(self, m, i, o):
        x = o[0] if isinstance(o, tuple) else o
        if x.shape[1] > 1 or self.delta is None:
            pos = self.read_pos if self.read_pos is not None else torch.full(
                (x.shape[0],), x.shape[1] - 1, device=x.device)
            h = x[torch.arange(x.shape[0], device=x.device), pos].float()
            self.delta = self.corr(h)
        y = x + self.delta[:, None, :].to(x.dtype)
        return (y,) + o[1:] if isinstance(o, tuple) else y

    def remove(self):
        self.h.remove()


class LoRALinear(nn.Module):
    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base, self.s = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A.T) @ self.B.T * self.s).to(x.dtype)


def add_lora(model):
    params = []
    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            lin = LoRALinear(getattr(layer.self_attn, name))
            setattr(layer.self_attn, name, lin)
            params += [lin.A, lin.B]
    return params


def remove_lora(model):
    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            m = getattr(layer.self_attn, name)
            if isinstance(m, LoRALinear):
                setattr(layer.self_attn, name, m.base)


# ---------- data ----------

def encode(tok, prompts, targets, device):
    ids, labels, rpos = [], [], []
    for p, t in zip(prompts, targets):
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ti = tok(p + t + tok.eos_token, add_special_tokens=False)["input_ids"][len(pi):]
        ids.append(pi + ti)
        labels.append([-100] * len(pi) + ti)
        rpos.append(len(pi) - 1)
    T = max(map(len, ids))
    x = torch.tensor([[tok.pad_token_id] * (T - len(s)) + s for s in ids], device=device)
    y = torch.tensor([[-100] * (T - len(s)) + s for s in labels], device=device)
    m = torch.tensor([[0] * (T - len(s)) + [1] * len(s) for s in ids], device=device)
    rp = torch.tensor([T - len(s) + r for s, r in zip(ids, rpos)], device=device)
    return x, y, m, rp


def train_epochs(model, tok, params, data, epochs, args, hook=None, on_epoch=None):
    """data: list of (prompt, target, weight). Calls on_epoch(ep) after each epoch."""
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(data))
        t, tot = time.time(), 0.0
        for s in range(0, len(order), args.train_bs):
            b = [data[k] for k in order[s:s + args.train_bs]]
            x, y, m, rp = encode(tok, [d[0] for d in b], [d[1] for d in b], model.device)
            w = torch.tensor([d[2] for d in b], device=model.device)
            if hook is not None:
                hook.read_pos, hook.delta = rp, None
            logits = model(input_ids=x, attention_mask=m).logits[:, :-1].float()
            tl = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y[:, 1:], ignore_index=-100,
                                                   reduction="none")
            per_ex = tl.sum(1) / (y[:, 1:] != -100).sum(1).clamp_min(1)
            loss = (per_ex * w).sum() / w.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if hook is not None:
            hook.read_pos = None
        print(f"    epoch {ep} loss {tot / len(data):.3f} ({time.time() - t:.0f}s)", flush=True)
        on_epoch(ep)


def correct_on(model, tok, recs, idx, args, hook=None):
    if hook is not None:
        hook.read_pos, hook.delta = None, None
    gens = generate(model, tok, [recs[i]["prompt"] for i in idx], args.max_new_tokens, args.bs)
    return [int(is_correct(g, recs[i]["gold"])) for g, i in zip(gens, idx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/go_nogo_llama3.2-3b")
    ap.add_argument("--out", default="results/corrector_stage1_llama3.2-3b")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--train-bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    eps_grid, max_ep = [0.05, 0.1, 0.2], 3
    if args.smoke:
        args.out += "_smoke"
        eps_grid, max_ep = [0.1], 1
    os.makedirs(args.out, exist_ok=True)

    recs = json.load(open(os.path.join(args.src, "records.json")))
    tok_norm = np.load(os.path.join(args.src, "features.npz"))["tok_norm"]
    y = np.array([r["label"] for r in recs])
    tr, tu, te = split(y)
    if args.smoke:
        tr, tu, te = tr[:200], tu[:64], te[:64]
    model, tok = load_model(args.model)
    for p in model.parameters():
        p.requires_grad_(False)
    dev, H = model.device, model.config.hidden_size

    # ---- 10 samples per question ----
    sp = os.path.join(args.out, "samples.json")
    if os.path.exists(sp):
        samples = {int(k): v for k, v in json.load(open(sp)).items()}
    else:
        samples, t = {}, time.time()
        allidx = np.r_[tr, tu, te]
        with torch.no_grad():
            for s in range(0, len(allidx), 32):
                b = allidx[s:s + 32]
                enc = tok([recs[i]["prompt"] for i in b], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(dev)
                g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=10,
                                   max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
                txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for j, i in enumerate(b):
                    samples[int(i)] = [x.strip() for x in txt[j * 10:(j + 1) * 10]]
        json.dump(samples, open(sp, "w"))
        print(f"[samples] {len(samples)} questions in {time.time() - t:.0f}s", flush=True)
    n_ok = {i: sum(is_correct(s, recs[i]["gold"]) for s in v) for i, v in samples.items()}

    # ---- training data ----
    R, K = [], []
    for i in tr:
        if y[i] == 1:
            ok = [s for s in samples[int(i)] if s and is_correct(s, recs[i]["gold"])]
            if ok:
                R.append((recs[i]["prompt"], collections.Counter(ok).most_common(1)[0][0]))
        else:
            K.append((recs[i]["prompt"], recs[i]["generation"].strip()))
    data = [(p, t, 1.0 / len(R)) for p, t in R] + [(p, t, 1.0 / len(K)) for p, t in K]
    print(f"[data] R (recoverable errors) = {len(R)}, K (keep) = {len(K)}", flush=True)

    res_path = os.path.join(args.out, "tune.json")
    tune = json.load(open(res_path)) if os.path.exists(res_path) else {}
    save = lambda: json.dump(tune, open(res_path, "w"))
    base_tu = [1 - int(y[i]) for i in tu]
    tune["none"] = base_tu

    # ---- corrector and fixed vector: train, evaluate tune after each epoch, keep weights per config ----
    states = {}
    for kind in ["corrector", "fixed"]:
        for e in eps_grid:
            if all(f"{kind}:{e}:{ep}" in tune for ep in range(1, max_ep + 1)) and \
                    os.path.exists(os.path.join(args.out, f"{kind}_{e}.pt")):
                states[(kind, e)] = torch.load(os.path.join(args.out, f"{kind}_{e}.pt"))
                continue
            print(f"[train] {kind} eps={e}", flush=True)
            torch.manual_seed(0)
            corr = Corrector(H, e, float(tok_norm[READ]), fixed=(kind == "fixed")).to(dev)
            hook = Hook(model, corr)
            snaps = {}

            def on_epoch(ep, kind=kind, e=e, corr=corr, hook=hook, snaps=snaps):
                tune[f"{kind}:{e}:{ep}"] = correct_on(model, tok, recs, tu, args, hook)
                snaps[ep] = {k: v.detach().cpu().clone() for k, v in corr.state_dict().items()}
                save()
                print(f"    tune acc {np.mean(tune[f'{kind}:{e}:{ep}']):.3f} (base {np.mean(base_tu):.3f})", flush=True)

            train_epochs(model, tok, list(corr.parameters()), data, max_ep, args, hook, on_epoch)
            hook.remove()
            states[(kind, e)] = snaps
            torch.save(snaps, os.path.join(args.out, f"{kind}_{e}.pt"))

    # ---- LoRA ----
    if not all(f"lora:{ep}" in tune for ep in range(1, max_ep + 1)):
        print("[train] lora", flush=True)
        torch.manual_seed(0)
        params = add_lora(model)
        lsnaps = {}

        def on_epoch_l(ep):
            tune[f"lora:{ep}"] = correct_on(model, tok, recs, tu, args)
            lsnaps[ep] = [p.detach().cpu().clone() for p in params]
            save()
            print(f"    tune acc {np.mean(tune[f'lora:{ep}']):.3f}", flush=True)

        train_epochs(model, tok, params, data, max_ep, args, None, on_epoch_l)
        remove_lora(model)
        torch.save(lsnaps, os.path.join(args.out, "lora.pt"))
    lsnaps = torch.load(os.path.join(args.out, "lora.pt"))

    # ---- selection on tune ----
    def best_of(prefix):
        keys = [k for k in tune if k.startswith(prefix + ":")]
        return max(keys, key=lambda k: np.mean(tune[k]))
    sel = {"corrector": best_of("corrector"), "fixed": best_of("fixed"), "lora": best_of("lora")}
    print(f"[select] {sel}", flush=True)

    # ---- test (once) ----
    test_path = os.path.join(args.out, "test.json")
    test = json.load(open(test_path)) if os.path.exists(test_path) else {}
    test["none"] = [1 - int(y[i]) for i in te]
    for kind in ["corrector", "fixed"]:
        if kind in test:
            continue
        _, e, ep = sel[kind].split(":")
        corr = Corrector(H, float(e), float(tok_norm[READ]), fixed=(kind == "fixed")).to(dev)
        corr.load_state_dict(states[(kind, float(e))][int(ep)])
        hook = Hook(model, corr)
        test[kind] = correct_on(model, tok, recs, te, args, hook)
        hook.remove()
        json.dump(test, open(test_path, "w"))
    if "lora" not in test:
        params = add_lora(model)
        for p, v in zip(params, lsnaps[int(sel["lora"].split(":")[1])]):
            p.data.copy_(v.to(p.device))
        test["lora"] = correct_on(model, tok, recs, te, args)
        remove_lora(model)
        json.dump(test, open(test_path, "w"))

    # ---- analysis ----
    B = np.random.default_rng(0).integers(0, len(te), size=(2000, len(te)))
    ci = lambda d: [float(np.percentile(d[B].mean(1), 2.5)), float(np.percentile(d[B].mean(1), 97.5))]
    A = {k: np.array(v) for k, v in test.items()}
    base = A["none"]
    knows = np.array([n_ok[int(i)] >= 2 for i in te])
    doesnt = np.array([n_ok[int(i)] == 0 for i in te])
    out = {"selected": sel, "n_R": len(R), "n_K": len(K),
           "test_acc": {k: float(v.mean()) for k, v in A.items()},
           "corrector_minus_none_ci": ci(A["corrector"] - base),
           "corrector_minus_fixed_ci": ci(A["corrector"] - A["fixed"]),
           "corrector_minus_lora_ci": ci(A["corrector"] - A["lora"]),
           "fix_rate": {k: float(v[base == 0].mean()) for k, v in A.items() if k != "none"},
           "break_rate": {k: float(1 - v[base == 1].mean()) for k, v in A.items() if k != "none"},
           "fix_rate_knows": {k: float(v[(base == 0) & knows].mean()) for k, v in A.items() if k != "none"},
           "fix_rate_doesnt": {k: float(v[(base == 0) & doesnt].mean()) for k, v in A.items() if k != "none"},
           "n_test_wrong_knows": int(((base == 0) & knows).sum()),
           "n_test_wrong_doesnt": int(((base == 0) & doesnt).sum())}
    out["Q1"] = out["corrector_minus_none_ci"][0] > 0
    out["Q2"] = out["corrector_minus_fixed_ci"][0] > 0
    out["Q3"] = out["corrector_minus_lora_ci"][0] > -0.02
    out["verdict"] = "GO" if out["Q1"] and out["Q2"] else "NO-GO"
    json.dump(out, open(os.path.join(args.out, "verdict_stage1.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
