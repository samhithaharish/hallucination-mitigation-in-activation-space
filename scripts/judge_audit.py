"""LLM-judge audit of the containment answer check (see docs/prereg_judge_audit.md).

    python scripts/judge_audit.py --judge qwen      # writes results/judge_audit/verdicts_qwen.json
    python scripts/judge_audit.py --judge phi
    python scripts/judge_audit.py --analyse
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from go_nogo import is_correct, load_model  # noqa: E402
from mixed_vs_separated import MODELS  # noqa: E402

JUDGES = {"qwen": "Qwen/Qwen2.5-7B-Instruct", "phi": "microsoft/Phi-3.5-mini-instruct"}
DATA = {"triviaqa": ("results/mixed_vs_separated", "results/type_control"),
        "nq_open": ("results/nq/mixed_vs_separated", "results/nq/type_control")}
OUT = "results/judge_audit"


def build_items():
    """Audit A: 50 greedy answers per model per dataset. Audit B: all sampled-wrong targets + 200 gold."""
    items = []
    rng = np.random.default_rng(3)
    for ds, (mvs, tc) in DATA.items():
        gold_pool = []
        for k in MODELS:
            recs = json.load(open(os.path.join(mvs, k, "records.json")))
            if "question" not in recs[0]:          # older TriviaQA records lack the raw question
                from datasets import load_dataset
                qs = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation").select(range(len(recs)))
                for rr, ex in zip(recs, qs):
                    rr["question"] = ex["question"]
            for i in rng.choice(len(recs), size=50, replace=False):
                r = recs[int(i)]
                items.append({"audit": "A", "ds": ds, "model": k, "qi": int(i), "question": r["question"],
                              "gold": r["gold"], "answer": r["generation"].strip(),
                              "contain": int(is_correct(r["generation"], r["gold"]))})
            t = json.load(open(os.path.join(tc, k, "type_control.json")))
            rng_k = np.random.default_rng(2)       # the same question sample as type_control.py
            y = np.array([r["label"] for r in recs])
            from sklearn.model_selection import train_test_split
            tr, _ = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
            idx = rng_k.choice([i for i in tr if y[i] == 1], size=t["n"], replace=False)
            for j, i in enumerate(idx):
                r = recs[int(i)]
                sw = t["targets"]["sampled_wrong"][j]
                if sw:
                    items.append({"audit": "B", "ds": ds, "model": k, "j": j, "question": r["question"],
                                  "gold": r["gold"], "answer": sw, "contain": 0})
                gold_pool.append({"audit": "Bgold", "ds": ds, "model": k, "j": j, "question": r["question"],
                                  "gold": r["gold"], "answer": t["targets"]["gold"][j], "contain": 1})
        for g in np.random.default_rng(4).choice(len(gold_pool), size=100, replace=False):
            items.append(gold_pool[int(g)])
    return items


@torch.no_grad()
def judge(items, key, bs=32):
    model, tok = load_model(JUDGES[key])
    yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")]
    no = [tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")]
    prompts = []
    for it in items:
        msg = (f"Question: {it['question']}\nReference answers: {'; '.join(it['gold'][:10])}\n"
               f"Proposed answer: {it['answer']}\n\nIs the proposed answer correct, i.e. does it refer to the same "
               f"thing as one of the reference answers? Reply with only yes or no.")
        prompts.append(tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                               add_generation_prompt=True))
    out = []
    for s in range(0, len(prompts), bs):
        enc = tok(prompts[s:s + bs], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        lg = model(**enc).logits[:, -1].float()
        py, pn = lg[:, yes].logsumexp(-1), lg[:, no].logsumexp(-1)
        out += (py > pn).int().tolist()
    return out


def analyse():
    items = json.load(open(os.path.join(OUT, "items.json")))
    V = {k: json.load(open(os.path.join(OUT, f"verdicts_{k}.json"))) for k in JUDGES
         if os.path.exists(os.path.join(OUT, f"verdicts_{k}.json"))}
    res = {}

    def kappa(a, b):
        a, b = np.array(a), np.array(b)
        po = (a == b).mean()
        pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        return float((po - pe) / (1 - pe)) if pe < 1 else 1.0

    for ds in DATA:
        A = [n for n, it in enumerate(items) if it["audit"] == "A" and it["ds"] == ds]
        c = np.array([items[n]["contain"] for n in A])
        r = {"n": len(A)}
        for k, v in V.items():
            j = np.array([v[n] for n in A])
            r[k] = {"agreement": float((c == j).mean()), "kappa": kappa(c, j),
                    "false_pos": float(((c == 1) & (j == 0)).sum() / max(1, (c == 1).sum())),
                    "false_neg": float(((c == 0) & (j == 1)).sum() / max(1, (c == 0).sum())),
                    "acc_contain": float(c.mean()), "acc_judge": float(j.mean())}
        if len(V) == 2:
            r["judge_judge_kappa"] = kappa([V["qwen"][n] for n in A], [V["phi"][n] for n in A])
        Bn = [n for n, it in enumerate(items) if it["audit"] == "B" and it["ds"] == ds]
        Gn = [n for n, it in enumerate(items) if it["audit"] == "Bgold" and it["ds"] == ds]
        for k, v in V.items():
            r[k]["sampled_wrong_called_correct"] = float(np.mean([v[n] for n in Bn]))
            r[k]["gold_called_correct"] = float(np.mean([v[n] for n in Gn]))
        # headline robustness: keep only sampled-wrong targets both judges call wrong
        rob, n_ok = {}, 0
        for m in MODELS:
            t = json.load(open(os.path.join(DATA[ds][1], m, "type_control.json")))
            bad = {items[n]["j"] for n in Bn if items[n]["model"] == m and any(V[k][n] == 1 for k in V)}
            g = np.array([np.nan if v is None else v for v in t["gold:0.05"]], float)
            w = np.array([np.nan if v is None else v for v in t["sampled_wrong:0.05"]], float)
            keep = ~np.isnan(g) & ~np.isnan(w) & np.array([j not in bad for j in range(len(g))])
            d = g[keep] - w[keep]
            bs = np.random.default_rng(0).integers(0, len(d), size=(2000, len(d)))
            ci = [float(np.percentile(d[bs].mean(1), q)) for q in (2.5, 97.5)]
            ok = not (ci[0] > 0)                   # gold not significantly easier
            n_ok += ok
            rob[m] = {"n_kept": int(keep.sum()), "n_dropped": len(bad), "gold": float(g[keep].mean()),
                      "sampled_wrong": float(w[keep].mean()), "gold_minus_wrong_ci": ci,
                      "wrong_significantly_easier": ci[1] < 0}
        r["headline_robust"] = {"per_model": rob, "count_gold_not_easier": n_ok, "survives": n_ok >= 7}
        res[ds] = r
    json.dump(res, open(os.path.join(OUT, "audit.json"), "w"), indent=1)
    for ds, r in res.items():
        print(f"== {ds}")
        for k in V:
            x = r[k]
            print(f"  {k}: agreement {x['agreement']:.3f} kappa {x['kappa']:.2f} FP {x['false_pos']:.3f} "
                  f"FN {x['false_neg']:.3f} | sampled-wrong called correct {x['sampled_wrong_called_correct']:.3f} "
                  f"| gold called correct {x['gold_called_correct']:.3f}")
        if "judge_judge_kappa" in r:
            print(f"  judge-judge kappa {r['judge_judge_kappa']:.2f}")
        h = r["headline_robust"]
        for m, x in h["per_model"].items():
            print(f"  {m:14s} kept {x['n_kept']:3d} dropped {x['n_dropped']:3d} gold {x['gold']:.2f} wrong "
                  f"{x['sampled_wrong']:.2f} ci {np.round(x['gold_minus_wrong_ci'], 2)}")
        print(f"  headline: gold not easier in {h['count_gold_not_easier']}/10 -> "
              f"{'survives' if h['survives'] else 'FAILS'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", choices=list(JUDGES))
    ap.add_argument("--analyse", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.analyse:
        return analyse()
    ip = os.path.join(OUT, "items.json")
    if not os.path.exists(ip):
        json.dump(build_items(), open(ip, "w"))
    items = json.load(open(ip))
    print(f"{len(items)} items", flush=True)
    json.dump(judge(items, args.judge), open(os.path.join(OUT, f"verdicts_{args.judge}.json"), "w"))


if __name__ == "__main__":
    main()
