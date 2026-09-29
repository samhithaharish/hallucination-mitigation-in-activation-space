"""One end-to-end run of the detect-vs-steer experiment for a single model.

    python scripts/run_experiment.py --model llama3.1-8b
    python scripts/run_experiment.py --model qwen2.5-7b --alpha 4.0

Steps (each is disk-cached under results/cache/<model>/, so re-runs resume):
    1. generate + label TriviaQA answers          -> records.json
    2. extract per-layer hidden states            -> hidden_states.npy, labels.npy
    3. per-layer probe AUROC sweep                -> detect layer
    4. CAA steering vectors + steer-layer sweep   -> steer layer
    5. layer-swap ablation grid                   -> swap_grid_results.json
    6. append this model's row to results/cross_model_comparison.csv

Run on a GPU box (8B needs ~40GB, e.g. one A100). Set HF_TOKEN in the environment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from facr.ablation import run_swap_grid, summarize_swap
from facr.config import MODEL_CONFIGS, RunConfig, get_hf_token
from facr.data import build_records
from facr.features import extract_hidden_states
from facr.probes import probe_sweep, select_boundary_cases
from facr.steering import baseline_accuracy, build_steering_vectors, steer_layer_sweep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.1-8b", choices=MODEL_CONFIGS.keys())
    ap.add_argument("--n-samples", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--normalize-vectors", action="store_true",
                    help="unit-norm steering vectors so alpha is comparable across layers")
    ap.add_argument("--skip-swap", action="store_true")
    args = ap.parse_args()

    cfg = RunConfig(model_key=args.model, n_samples=args.n_samples, alpha=args.alpha)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = get_hf_token()
    print(f"Model: {cfg.model.hf_id} | device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.hf_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_id, torch_dtype=torch.bfloat16, device_map="auto", token=token)
    model.eval()

    records = build_records(model, tokenizer, cfg, device)
    hidden, labels = extract_hidden_states(model, tokenizer, records, cfg, device)

    layer_auroc, probes, _, detect_layer = probe_sweep(hidden, labels, cfg)
    n_hs = hidden.shape[1]
    detect_depth = detect_layer / (n_hs - 1)
    print(f"Detect-layer peak: {detect_layer} ({100 * detect_depth:.0f}% depth), "
          f"AUROC {layer_auroc[detect_layer]:.4f}")

    vectors = build_steering_vectors(hidden, labels, device, normalize=args.normalize_vectors)
    boundary_idx = select_boundary_cases(hidden, probes, detect_layer, cfg.n_boundary)
    eval_idx = boundary_idx[: cfg.n_steer_eval]
    print(f"Baseline (unsteered) accuracy on eval set: {baseline_accuracy(records, eval_idx, cfg):.3f}")

    steer_accs, steer_layer = steer_layer_sweep(model, tokenizer, records, vectors, eval_idx, cfg, device)
    steer_depth = steer_layer / (n_hs - 1)
    print(f"Steer-layer peak: {steer_layer} ({100 * steer_depth:.0f}% depth)")

    row = {
        "detect_layer": detect_layer, "detect_depth": round(detect_depth, 2),
        "detect_auroc": round(float(layer_auroc[detect_layer]), 4),
        "steer_layer": steer_layer, "steer_depth": round(steer_depth, 2),
        "steer_acc": round(float(steer_accs[steer_layer - 1]), 4),
    }

    if not args.skip_swap:
        df = run_swap_grid(model, tokenizer, records, hidden, probes, vectors,
                           detect_layer, steer_layer, cfg, device)
        row.update({k: round(v, 4) for k, v in summarize_swap(df, detect_layer, steer_layer).items()})

    os.makedirs("results", exist_ok=True)
    out = "results/cross_model_comparison.csv"
    new = pd.DataFrame([row], index=[cfg.model_key])
    if os.path.exists(out):
        old = pd.read_csv(out, index_col=0)
        new = pd.concat([old[old.index != cfg.model_key], new])
    new.to_csv(out)
    np.save(os.path.join(cfg.cache_dir, "layer_auroc.npy"), layer_auroc)
    np.save(os.path.join(cfg.cache_dir, "steer_accs.npy"), steer_accs)
    print(new)


if __name__ == "__main__":
    main()
