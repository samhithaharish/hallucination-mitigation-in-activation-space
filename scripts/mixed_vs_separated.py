"""Mixed vs separated sweep (see docs/prereg_mixed_vs_separated.md).

    python scripts/mixed_vs_separated.py --key llama3.2-3b          # one model
    python scripts/mixed_vs_separated.py --aggregate                # after all models
    python scripts/mixed_vs_separated.py --key smollm2-1.7b --smoke # quick plumbing test
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
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from go_nogo import build_prompt, generate, is_correct, load_model, normalize  # noqa: E402
from correctability_pilot import batch_nll_inputs, generate_with, optimize_delta  # noqa: E402

MODELS = {
    "llama3.2-3b": ("meta-llama/Llama-3.2-3B-Instruct", "small"),
    "qwen2.5-3b": ("Qwen/Qwen2.5-3B-Instruct", "small"),
    "olmo2-1b": ("allenai/OLMo-2-0425-1B-Instruct", "small"),
    "granite3.1-2b": ("ibm-granite/granite-3.1-2b-instruct", "small"),
    "smollm2-1.7b": ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "small"),
    "llama3.1-8b": ("meta-llama/Llama-3.1-8B-Instruct", "large"),
    "qwen2.5-7b": ("Qwen/Qwen2.5-7B-Instruct", "large"),
    "olmo2-7b": ("allenai/OLMo-2-1124-7B-Instruct", "large"),
    "granite3.1-8b": ("ibm-granite/granite-3.1-8b-instruct", "large"),
    "mistral-7b": ("mistralai/Mistral-7B-Instruct-v0.3", "large"),
}
DEPTHS = [0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))


def _cv_auc(X, y, tr, te):
    return roc_auc_score(y[te], probe().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])


@torch.no_grad()
def teacher_forced_mean(model, tok, prompts, answers, layers, bs):
    """Returns (mean-over-answer-token states at `layers` (N, len(layers), H), mean answer-token log-prob (N,))."""
    out, lps = [], []
    for s in range(0, len(prompts), bs):
        P, A = prompts[s:s + bs], answers[s:s + bs]
        n_ans = [max(1, len(tok(p + a, add_special_tokens=False)["input_ids"]) -
                     len(tok(p, add_special_tokens=False)["input_ids"])) for p, a in zip(P, A)]
        enc = tok([p + a for p, a in zip(P, A)], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        o = model(**enc, output_hidden_states=True)
        hs = o.hidden_states
        T = enc["input_ids"].shape[1]
        m = torch.zeros(len(P), T, device=model.device)
        for j, n in enumerate(n_ans):
            m[j, T - n:] = 1
        feats = [((hs[l].float() * m[..., None]).sum(1) / m.sum(1, keepdim=True)) for l in layers]
        out.append(torch.stack(feats, 1).half().cpu().numpy())
        for j, n in enumerate(n_ans):                     # logits at t-1 predict token t
            lg = o.logits[j, T - n - 1:T - 1].float().log_softmax(-1)
            lps.append(float(lg.gather(1, enc["input_ids"][j, T - n:, None]).mean()))
        del o, hs
    return np.concatenate(out), np.array(lps)


def run_model(key, args):
    hf_id, size = MODELS[key]
    out_dir = os.path.join(args.out, key)
    os.makedirs(out_dir, exist_ok=True)
    summ_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summ_path) and os.path.exists(os.path.join(out_dir, "scores.npz")):
        print(f"[{key}] already done")
        return
    t0 = time.time()
    model, tok = load_model(hf_id)
    for p in model.parameters():
        p.requires_grad_(False)
    dev, L = model.device, model.config.num_hidden_layers
    large = size == "large"
    bs, ebs = (32, 6) if large else (64, 12)

    from datasets import load_dataset
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").select(range(args.n))

    # ---- A. greedy labels + 10 samples ----
    rp = os.path.join(out_dir, "records.json")
    if os.path.exists(rp):
        recs = json.load(open(rp))
    else:
        prompts = [build_prompt(tok, ex["question"]) for ex in ds]
        gens = generate(model, tok, prompts, args.max_new_tokens, bs)
        samples = []
        with torch.no_grad():
            for s in range(0, len(prompts), 16):
                enc = tok(prompts[s:s + 16], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
                g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, num_return_sequences=10,
                                   max_new_tokens=args.max_new_tokens, pad_token_id=tok.pad_token_id)
                txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                samples += [[x.strip() for x in txt[j * 10:(j + 1) * 10]] for j in range(len(txt) // 10)]
        recs = []
        for ex, p, g, sm in zip(ds, prompts, gens, samples):
            golds = list(dict.fromkeys(ex["answer"]["aliases"] + [ex["answer"]["value"]]))
            recs.append({"prompt": p, "gold": golds, "value": ex["answer"]["value"], "generation": g,
                         "samples": sm, "label": 0 if is_correct(g, golds) else 1,
                         "n_ok": sum(is_correct(x, golds) for x in sm)})
        json.dump(recs, open(rp, "w"))
    y = np.array([r["label"] for r in recs])
    n_f, n_h = int((y == 0).sum()), int((y == 1).sum())
    print(f"[{key}] factual {n_f} / hallucinated {n_h} ({time.time() - t0:.0f}s)", flush=True)
    if min(n_f, n_h) < (20 if args.smoke else 150):
        json.dump({"key": key, "size": size, "excluded": f"class sizes {n_f}/{n_h}"}, open(summ_path, "w"))
        return
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.3, random_state=0, stratify=y)
    edit_layer = int(round(0.45 * L))

    # ---- B. question-level features ----
    qp = os.path.join(out_dir, "qfeat.npz")
    if os.path.exists(qp):
        z = np.load(qp)
        Q, tok_norm_e = z["Q"], float(z["tok_norm_e"])
    else:
        Q, norms = [], []
        with torch.no_grad():
            for s in range(0, len(recs), bs):
                enc = tok([r["prompt"] for r in recs[s:s + bs]], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(dev)
                hs = model(**enc, output_hidden_states=True).hidden_states
                mask = enc["attention_mask"].clone()
                mask[torch.arange(mask.shape[0]), mask.float().argmax(1)] = 0
                m = mask[..., None].float()
                Q.append(torch.stack([(h.float() * m).sum(1) / m.sum(1) for h in hs], 1).half().cpu().numpy())
                norms.append(hs[edit_layer].float().norm(dim=-1)[mask.bool()].cpu().numpy())
        Q, tok_norm_e = np.concatenate(Q), float(np.median(np.concatenate(norms)))
        np.savez(qp, Q=Q, tok_norm_e=tok_norm_e)

    # ---- C. M1 question-level detectability ----
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y[tr]))
    Qtr, ytr = Q[tr].astype(np.float32), y[tr]
    aucs = Parallel(n_jobs=args.jobs)(delayed(_cv_auc)(Qtr[:, l], ytr, a, b)
                                      for l in range(1, L + 1) for a, b in folds)
    m1_curve = np.array(aucs).reshape(L, 5).mean(1)
    M1, M1_layer = float(m1_curve.max()), int(np.argmax(m1_curve)) + 1
    # detector scores for Path B: out-of-fold on train, fit-on-train for test
    p_h = np.zeros(len(y))
    for a, b in folds:
        p_h[tr[b]] = probe().fit(Qtr[a, M1_layer], ytr[a]).predict_proba(Qtr[b, M1_layer])[:, 1]
    p_h[te] = probe().fit(Qtr[:, M1_layer], ytr).predict_proba(Q[te, M1_layer].astype(np.float32))[:, 1]
    del Q, Qtr
    print(f"[{key}] M1 {M1:.3f} at layer {M1_layer}/{L}", flush=True)

    # ---- D. M2 knows fraction ----
    wrong = y == 1
    M2 = float(np.mean([recs[i]["n_ok"] >= 2 for i in np.where(wrong)[0]]))

    # ---- E. M3 correction-direction structure ----
    ep = os.path.join(out_dir, "edits.json")
    if os.path.exists(ep):
        E = json.load(open(ep))
    else:
        rng = np.random.default_rng(1)
        eidx = rng.choice([i for i in tr if y[i] == 1], size=args.n_edit, replace=False)
        ctrl = np.roll(eidx, 1)
        prompts = [recs[i]["prompt"] for i in eidx]
        E = {}
        for target in ["gold", "control"]:
            tsrc = eidx if target == "gold" else ctrl
            ttxt = [recs[i]["value"] for i in tsrc]
            tgold = [recs[i]["gold"] for i in tsrc]
            for e in [0.05, 0.1]:
                radius = e * tok_norm_e
                fixed, deltas = [], []
                for s in range(0, len(eidx), ebs):
                    x, yy, m = batch_nll_inputs(tok, prompts[s:s + ebs], ttxt[s:s + ebs], dev)
                    d = optimize_delta(model, x, yy, m, edit_layer, radius, 30, lr=radius / 10)
                    gens = generate_with(model, tok, prompts[s:s + ebs], edit_layer, d, args)
                    fixed += [int(is_correct(g, gl)) for g, gl in zip(gens, tgold[s:s + ebs])]
                    deltas.append(d.cpu())
                E[f"{target}:{e}"] = fixed
                if target == "gold" and e == 0.1:
                    D = torch.cat(deltas).float()
                    Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    E["cos"] = float((Dn @ Dn.T).numpy()[np.triu_indices(len(D), 1)].mean())
                    half, tf = len(D) // 2, {}
                    for a, b in ((slice(0, half), slice(half, None)), (slice(half, None), slice(0, half))):
                        mu = Dn[a].mean(0)
                        mu = (mu / mu.norm() * radius).to(dev)
                        Pb, Gb = prompts[b], [recs[i]["gold"] for i in eidx[b]]
                        for s in range(0, len(Pb), ebs):
                            gens = generate_with(model, tok, Pb[s:s + ebs], edit_layer,
                                                 mu.expand(len(Pb[s:s + ebs]), -1), args)
                            for g, gl in zip(gens, Gb[s:s + ebs]):
                                tf.setdefault(str(b), []).append(int(is_correct(g, gl)))
                    E["transfer"] = sum(tf.values(), [])
                print(f"[{key}] edits {target} eps {e}: fix {np.mean(fixed):.3f}", flush=True)
        json.dump(E, open(ep, "w"))
    per_q = np.mean(E["gold:0.1"])
    M3a, M3b = E["cos"], float(np.mean(E["transfer"]) / max(per_q, 1e-8))
    M3c = float(np.mean(E["gold:0.05"]) - np.mean(E["control:0.05"]))

    # ---- F. M4 answer-level separability + generate-then-select ----
    cands, cq, cl, cfreq, cgreedy = [], [], [], [], []
    for i, r in enumerate(recs):
        seen = {}
        answers = [r["generation"].strip()] + r["samples"]
        counts = collections.Counter(normalize(a) for a in answers if a and normalize(a))
        for a in answers:
            if a and normalize(a) and normalize(a) not in seen:
                seen[normalize(a)] = a
        g = normalize(r["generation"])
        for na, a in seen.items():
            cands.append(a)
            cq.append(i)
            cl.append(int(is_correct(a, r["gold"])))
            cfreq.append(counts[na] / len(answers))
            cgreedy.append(int(na == g))
    cq, cl, cfreq, cgreedy = np.array(cq), np.array(cl), np.array(cfreq), np.array(cgreedy)
    ap = os.path.join(out_dir, "afeat.npz")
    layers = [max(1, int(round(d * L))) for d in DEPTHS]
    if os.path.exists(ap):
        z = np.load(ap)
        AF, clogp = z["AF"], z["LP"]
    else:
        AF, clogp = teacher_forced_mean(model, tok, [recs[i]["prompt"] for i in cq], cands, layers, bs)
        np.savez(ap, AF=AF, LP=clogp)
    del model
    torch.cuda.empty_cache()

    trs, tes = set(tr.tolist()), set(te.tolist())
    ctr = np.array([q in trs for q in cq])
    cte = np.array([q in tes for q in cq])
    gkf = list(GroupKFold(5).split(np.where(ctr)[0], groups=cq[ctr]))
    Xtr_all, ytr_c = AF[ctr].astype(np.float32), cl[ctr]
    depth_auc = []
    for d in range(len(layers)):
        depth_auc.append(np.mean([_cv_auc(Xtr_all[:, d], ytr_c, a, b) for a, b in gkf
                                  if len(set(ytr_c[b])) == 2]))
    bd = int(np.argmax(depth_auc))
    clf = probe().fit(Xtr_all[:, bd], ytr_c)
    p_te = clf.predict_proba(AF[cte][:, bd].astype(np.float32))[:, 1]
    M4 = float(roc_auc_score(cl[cte], p_te))
    # candidate probe scores for Path B: out-of-fold (question-grouped) on train, fit-on-train for test
    cprobe = np.zeros(len(cq))
    tr_pos = np.where(ctr)[0]
    for a, b in gkf:
        cprobe[tr_pos[b]] = probe().fit(Xtr_all[a, bd], ytr_c[a]).predict_proba(Xtr_all[b, bd])[:, 1]
    cprobe[cte] = p_te
    np.savez(os.path.join(out_dir, "scores.npz"), y=y, tr=tr, te=te, p_h=p_h, cq=cq, cl=cl, cfreq=cfreq,
             cgreedy=cgreedy, clogp=clogp, cprobe=cprobe)
    del AF, Xtr_all

    by_q = collections.defaultdict(list)
    for q, lab, p in zip(cq[cte], cl[cte], p_te):
        by_q[q].append((p, lab))
    sel, greedy, maj, orc = [], [], [], []
    for q in te:
        r = recs[q]
        greedy.append(1 - r["label"])
        c = by_q.get(q, [])
        sel.append(max(c)[1] if c else 0)
        orc.append(int(any(lab for _, lab in c)))
        votes = collections.Counter(normalize(a) for a in [r["generation"]] + r["samples"] if normalize(a))
        top = votes.most_common(1)[0][0] if votes else ""
        maj.append(int(bool(top) and is_correct(top, r["gold"])))
    sel, greedy, maj, orc = map(np.array, (sel, greedy, maj, orc))
    B = np.random.default_rng(0).integers(0, len(te), size=(2000, len(te)))
    ci = lambda d: [float(np.percentile(d[B].mean(1), 2.5)), float(np.percentile(d[B].mean(1), 97.5))]

    summ = {"key": key, "hf_id": hf_id, "size": size, "n_layers": L, "n_factual": n_f, "n_halluc": n_h,
            "greedy_acc_all": float(1 - y.mean()),
            "M1": M1, "M1_layer": M1_layer, "M1_depth": M1_layer / L, "M1_curve": m1_curve.tolist(),
            "M2_knows_frac": M2,
            "M3a_cos": M3a, "M3b_transfer_ratio": M3b, "M3c_gold_minus_control": M3c,
            "M3_perq_fix_eps0.1": float(per_q), "M3_edit_layer": edit_layer,
            "M4": M4, "M4_depth": DEPTHS[bd], "M4_depth_cv_auc": [float(a) for a in depth_auc],
            "test_acc": {"greedy": float(greedy.mean()), "majority": float(maj.mean()),
                         "probe_select": float(sel.mean()), "oracle": float(orc.mean())},
            "select_minus_greedy_ci": ci(sel - greedy), "select_minus_majority_ci": ci(sel - maj),
            "call": "separated" if (M1 >= 0.80 and M3b >= 1 / 3) else "mixed",
            "minutes": (time.time() - t0) / 60}
    json.dump(summ, open(summ_path, "w"), indent=1)
    for f in ("qfeat.npz", "afeat.npz"):
        os.remove(os.path.join(out_dir, f))
    print(json.dumps({k: v for k, v in summ.items() if k not in ("M1_curve", "M4_depth_cv_auc")}, indent=1))


def aggregate(args):
    from scipy.stats import mannwhitneyu
    S = [json.load(open(os.path.join(args.out, k, "summary.json"))) for k in MODELS
         if os.path.exists(os.path.join(args.out, k, "summary.json"))]
    S = [s for s in S if "excluded" not in s]
    sm, lg = [s for s in S if s["size"] == "small"], [s for s in S if s["size"] == "large"]
    out = {"n_small": len(sm), "n_large": len(lg), "per_model": {}}
    for s in S:
        out["per_model"][s["key"]] = {k: s[k] for k in ("size", "M1", "M1_depth", "M2_knows_frac", "M3a_cos",
                                                       "M3b_transfer_ratio", "M3c_gold_minus_control", "M4",
                                                       "test_acc", "select_minus_greedy_ci",
                                                       "select_minus_majority_ci", "call")}
    for name, k in (("S1", "M1"), ("S2", "M2_knows_frac"), ("S3", "M3b_transfer_ratio")):
        if sm and lg:
            p = mannwhitneyu([s[k] for s in lg], [s[k] for s in sm], alternative="greater").pvalue
            out[name] = {"p": float(p), "holds": bool(p < 0.05),
                         "small_mean": float(np.mean([s[k] for s in sm])),
                         "large_mean": float(np.mean([s[k] for s in lg]))}
    out["A1"] = {"count": sum(s["M4"] > s["M1"] for s in S), "of": len(S)}
    out["A1"]["holds"] = out["A1"]["count"] >= 0.8 * len(S)
    a2 = sum(s["select_minus_greedy_ci"][0] > 0 and s["select_minus_majority_ci"][0] > 0 for s in S)
    out["A2"] = {"count": a2, "of": len(S), "holds": a2 >= 0.7 * len(S)}
    json.dump(out, open(os.path.join(args.out, "aggregate.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--out", default="results/mixed_vs_separated")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-edit", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=24)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.out, args.n, args.n_edit = args.out + "_smoke", 400, 12
    if args.aggregate:
        aggregate(args)
    else:
        run_model(args.key, args)


if __name__ == "__main__":
    main()
