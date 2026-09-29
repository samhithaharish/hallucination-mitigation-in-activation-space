"""Pre-registered v2 stronger-steering test (see docs/prereg_steering_v2.md).

Reuses records, features and the split from scripts/go_nogo.py (run that first).

    python scripts/steering_v2.py
    python scripts/steering_v2.py --smoke        # 2 layers, 1 alpha, few questions, separate out dir
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from go_nogo import generate, is_correct, load_model, split  # noqa: E402

ALPHAS = [0.05, 0.1, 0.2, 0.4, 0.8]
ITI_K, ITI_ALPHA = [16, 48], [5, 10, 15]
FRACS = [1.0, 0.5, 0.3, 0.2, 0.1]
LAYER_METHODS = ["caa_mean", "caa_last", "probe_dir", "contrastive"]


def nwords(s):
    return len(s.split())


def fit_probe(X, y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
    return clf.fit(X, y)


def probe_direction(clf):
    sc, lr = clf[0], clf[1]
    w = lr.coef_[0] / sc.scale_                      # raw-space gradient of the logit
    return -w                                        # away from "will hallucinate"


# ---------- teacher-forced gold vs wrong activations ----------

@torch.no_grad()
def teacher_forced(model, tok, prompts, answers, bs):
    """Returns (mean-over-answer-token hidden states (N, L+1, H), last-answer-token head inputs (N, L, H))."""
    L = model.config.num_hidden_layers
    heads, cur = [], {}
    hooks = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
        lambda m, a, l=l: cur.__setitem__(l, a[0][:, -1].float().cpu())) for l in range(L)]
    H, O = [], []
    try:
        for i in range(0, len(prompts), bs):
            P, A = prompts[i:i + bs], answers[i:i + bs]
            n_ans = [len(tok(p + a, add_special_tokens=False)["input_ids"]) -
                     len(tok(p, add_special_tokens=False)["input_ids"]) for p, a in zip(P, A)]
            enc = tok([p + a for p, a in zip(P, A)], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states, 1).float()  # B,L+1,T,H
            T = hs.shape[2]
            m = torch.zeros(len(P), T, device=hs.device)
            for j, n in enumerate(n_ans):
                m[j, T - max(n, 1):] = 1
            H.append(((hs * m[:, None, :, None]).sum(2) / m.sum(1)[:, None, None]).half().cpu().numpy())
            O.append(torch.stack([cur[l] for l in range(L)], 1).half().numpy())
    finally:
        for h in hooks:
            h.remove()
    return np.concatenate(H), np.concatenate(O)


# ---------- steering hooks ----------

def run_block_steer(model, tok, prompts, layer, vec, args):
    h = model.model.layers[layer - 1].register_forward_hook(
        lambda m, i, o: (o[0] + vec,) + o[1:] if isinstance(o, tuple) else o + vec)
    try:
        return generate(model, tok, prompts, args.max_new_tokens, args.bs)
    finally:
        h.remove()


def run_iti(model, tok, prompts, deltas, args):
    hooks = [model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(
        lambda m, a, d=d: (a[0] + d,) + a[1:]) for l, d in deltas.items()]
    try:
        return generate(model, tok, prompts, args.max_new_tokens, args.bs)
    finally:
        for h in hooks:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/go_nogo_llama3.2-3b")
    ap.add_argument("--out", default="results/steering_v2_llama3.2-3b")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.out += "_smoke"
    os.makedirs(args.out, exist_ok=True)

    recs = json.load(open(os.path.join(args.src, "records.json")))
    z = np.load(os.path.join(args.src, "features.npz"))
    mean, last, tok_norm = z["mean"].astype(np.float32), z["last"].astype(np.float32), z["tok_norm"]
    D = json.load(open(os.path.join(args.src, "detect_mean.json")))["D"]
    y = np.array([r["label"] for r in recs])
    tr, tu, te = split(y)
    layers = list(range(1, mean.shape[1]))
    alphas, itik, itia = ALPHAS, ITI_K, ITI_ALPHA
    if args.smoke:
        tu, te, layers, alphas, itik, itia = tu[:64], te[:64], [8, 13], [0.2], [16], [10]

    model, tok = load_model(args.model)
    dev = model.device
    base_c = {s: np.array([1 - recs[i]["label"] for i in idx]) for s, idx in (("tune", tu), ("test", te))}
    base_len = np.mean([nwords(recs[i]["generation"]) for i in tu])

    # detector at D (for conditional steering)
    det = fit_probe(mean[tr, D], y[tr])
    p_h = {"tune": det.predict_proba(mean[tu, D])[:, 1], "test": det.predict_proba(mean[te, D])[:, 1]}

    # ---- steering vectors ----
    vec_path = os.path.join(args.out, "vectors.npz")
    if os.path.exists(vec_path):
        V = dict(np.load(vec_path))
    else:
        t = time.time()
        V = {}
        V["caa_mean"] = mean[tr][y[tr] == 0].mean(0) - mean[tr][y[tr] == 1].mean(0)
        V["caa_last"] = last[tr][y[tr] == 0].mean(0) - last[tr][y[tr] == 1].mean(0)
        clfs = Parallel(n_jobs=args.jobs)(delayed(fit_probe)(mean[tr, l], y[tr]) for l in range(mean.shape[1]))
        V["probe_dir"] = np.stack([probe_direction(c) for c in clfs])
        from datasets import load_dataset
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").select(range(len(recs)))
        wrong = [i for i in tr if y[i] == 1 and recs[i]["generation"].strip()]
        P = [recs[i]["prompt"] for i in wrong]
        Hg, Og = teacher_forced(model, tok, P, [ds[int(i)]["answer"]["value"] for i in wrong], 32)
        Hw, Ow = teacher_forced(model, tok, P, [recs[i]["generation"] for i in wrong], 32)
        V["contrastive"] = (Hg.astype(np.float32) - Hw.astype(np.float32)).mean(0)
        V["iti_gold"], V["iti_wrong"] = Og, Ow
        np.savez(vec_path, **V)
        print(f"[vectors] built in {time.time() - t:.0f}s, {len(wrong)} gold/wrong pairs", flush=True)

    def layer_vec(method, l, a):
        v = V[method][l].astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8) * tok_norm[l] * a
        return torch.tensor(v, dtype=torch.bfloat16, device=dev)

    # ---- ITI head selection ----
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    hd = model.config.hidden_size // n_heads
    Og, Ow = V["iti_gold"].astype(np.float32), V["iti_wrong"].astype(np.float32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Og))
    ntr = int(0.8 * len(Og))
    a_tr, a_va = perm[:ntr], perm[ntr:]

    def head_acc(l, h):
        sl = slice(h * hd, (h + 1) * hd)
        X = np.concatenate([Og[:, l, sl], Ow[:, l, sl]])
        lab = np.r_[np.ones(len(Og)), np.zeros(len(Ow))]
        itr = np.r_[a_tr, a_tr + len(Og)]
        iva = np.r_[a_va, a_va + len(Og)]
        clf = LogisticRegression(max_iter=1000).fit(X[itr], lab[itr])
        return clf.score(X[iva], lab[iva])

    acc_path = os.path.join(args.out, "iti_head_acc.npy")
    if os.path.exists(acc_path):
        head_accs = np.load(acc_path)
    else:
        head_accs = np.array(Parallel(n_jobs=args.jobs)(
            delayed(head_acc)(l, h) for l in range(n_layers) for h in range(n_heads))).reshape(n_layers, n_heads)
        np.save(acc_path, head_accs)
    order = np.argsort(head_accs.ravel())[::-1]

    def iti_deltas(K, a):
        deltas = {}
        for flat in order[:K]:
            l, h = divmod(int(flat), n_heads)
            sl = slice(h * hd, (h + 1) * hd)
            d = Og[:, l, sl].mean(0) - Ow[:, l, sl].mean(0)
            d /= np.linalg.norm(d) + 1e-8
            sigma = np.concatenate([Og[:, l, sl], Ow[:, l, sl]]).dot(d).std()
            deltas.setdefault(l, np.zeros(n_heads * hd, dtype=np.float32))[sl] += a * sigma * d
        return {l: torch.tensor(v, dtype=torch.bfloat16, device=dev) for l, v in deltas.items()}

    # ---- run configs on a split, cached ----
    def run(split_name, idx, configs):
        path = os.path.join(args.out, f"steer_{split_name}.json")
        res = json.load(open(path)) if os.path.exists(path) else {}
        prompts = [recs[i]["prompt"] for i in idx]
        for key in configs:
            if key in res:
                continue
            t = time.time()
            m, p1, p2 = key.split(":")
            if m == "iti":
                gens = run_iti(model, tok, prompts, iti_deltas(int(p1), float(p2)), args)
            else:
                gens = run_block_steer(model, tok, prompts, int(p1), layer_vec(m, int(p1), float(p2)), args)
            res[key] = {"c": [int(is_correct(g, recs[i]["gold"])) for g, i in zip(gens, idx)],
                        "len": [nwords(g) for g in gens], "ex": gens[:5]}
            json.dump(res, open(path, "w"))
            print(f"[{split_name}] {key:<22} acc {np.mean(res[key]['c']):.3f} "
                  f"len {np.mean(res[key]['len']):.1f} ({time.time() - t:.0f}s)", flush=True)
        return res

    tune_keys = [f"{m}:{l}:{a}" for m in LAYER_METHODS for l in layers for a in alphas] + \
                [f"iti:{k}:{a}" for k in itik for a in itia]
    tune = run("tune", tu, tune_keys)

    # ---- selection on tune: config x flagged fraction, with the length guard ----
    def mix(split_name, r, tau):
        flag = p_h[split_name] > tau if tau is not None else np.ones(len(r["c"]), bool)
        base = base_c[split_name]
        blen = np.array([nwords(recs[i]["generation"]) for i in (tu if split_name == "tune" else te)])
        return np.where(flag, r["c"], base), np.where(flag, r["len"], blen)

    taus = {f: (None if f == 1.0 else float(np.quantile(p_h["tune"], 1 - f))) for f in FRACS}
    table = []
    for key in tune_keys:
        for f in FRACS:
            c, ln = mix("tune", tune[key], taus[f])
            table.append({"key": key, "frac": f, "gain": float(c.mean() - base_c["tune"].mean()),
                          "len_ratio": float(ln.mean() / base_len), "ok": bool(ln.mean() <= 2 * base_len)})
    ok = [r for r in table if r["ok"]]
    best = max(ok, key=lambda r: r["gain"])
    best_per_method = {m: max([r for r in ok if r["key"].startswith(m + ":")], key=lambda r: r["gain"])
                       for m in LAYER_METHODS + ["iti"]}
    print(f"[select] best on tune: {best}", flush=True)

    # ---- test ----
    test_keys = sorted({best["key"]} | {r["key"] for r in best_per_method.values()})
    method = best["key"].split(":")[0]
    per_layer = {}
    if method != "iti":
        for l in layers:
            rows = [r for r in ok if r["key"].startswith(f"{method}:{l}:")]
            per_layer[l] = max(rows, key=lambda r: r["gain"])
        test_keys = sorted(set(test_keys) | {r["key"] for r in per_layer.values()}
                           | ({f"{method}:{D}:{a}" for a in alphas} if D in layers else set()))
    test = run("test", te, test_keys)

    B = np.random.default_rng(0).integers(0, len(te), size=(2000, len(te)))
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def test_diff(row):
        c, _ = mix("test", test[row["key"]], taus[row["frac"]])
        return c - base_c["test"]

    d_best = test_diff(best)
    q1 = ci(d_best[B].mean(1))
    out = {"D": D, "baseline_tune_acc": float(base_c["tune"].mean()), "baseline_test_acc": float(base_c["test"].mean()),
           "best_config": best, "best_test_gain": float(d_best.mean()), "best_test_gain_ci": q1,
           "Q1_steering_works": q1[0] > 0,
           "secondary_best_per_method": {m: {**r, "test_gain": float(test_diff(r).mean()),
                                             "test_ci": ci(test_diff(r)[B].mean(1))}
                                         for m, r in best_per_method.items()}}
    if per_layer and D in per_layer:
        S = max(per_layer, key=lambda l: per_layer[l]["gain"])
        dS, dD = test_diff(per_layer[S]), test_diff(per_layer[D])
        out.update({"S": S, "gain_S_test": float(dS.mean()), "gain_D_test": float(dD.mean()),
                    "gain_S_minus_D_ci": ci((dS - dD)[B].mean(1)),
                    "per_layer_test_gain": {l: float(test_diff(r).mean()) for l, r in per_layer.items()},
                    "per_layer_tune_gain": {l: r["gain"] for l, r in per_layer.items()}})
        out["Q2_detect_ne_steer"] = bool(out["Q1_steering_works"] and abs(S - D) >= 4
                                         and out["gain_S_minus_D_ci"][0] > 0)
    json.dump(out, open(os.path.join(args.out, "verdict_v2.json"), "w"), indent=1)
    json.dump(table, open(os.path.join(args.out, "tune_table.json"), "w"))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("per_layer")}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in LAYER_METHODS:
        g = [max(r["gain"] for r in ok if r["key"].startswith(f"{m}:{l}:")) * 100 for l in layers]
        ax.plot(layers, g, label=m)
    it = best_per_method["iti"]["gain"] * 100
    ax.axhline(it, ls="--", color="gray", label=f"ITI best ({it:+.1f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(D, color="k", ls=":", lw=0.8)
    ax.set(xlabel="steer layer", ylabel="best tune gain (pts)", title="v2: best tune gain per layer (max over alpha, frac)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "v2_tune_gain.png"), dpi=150)


if __name__ == "__main__":
    main()
