"""Layer-swap ablation: decouple the layer that selects cases (detect) from the
layer that steers, over a small grid around the two peaks.

Claim under test: the best detect layer and the best steer layer differ, and using the
matched pair (detect_peak, steer_peak) beats using one layer for both jobs.
"""
from __future__ import annotations

import json
import os

import pandas as pd

from .probes import select_boundary_cases
from .steering import eval_steering_accuracy


def swap_grid_layers(detect_layer: int, steer_layer: int, n_total: int) -> list[int]:
    layers = {max(1, detect_layer - 2), detect_layer, detect_layer + 2,
              max(1, steer_layer - 2), steer_layer, steer_layer + 2}
    return sorted(l for l in layers if 1 <= l < n_total)


def run_swap_grid(model, tokenizer, records, hidden, probes, steering_vectors,
                  detect_layer, steer_layer, cfg, device):
    grid = swap_grid_layers(detect_layer, steer_layer, hidden.shape[1])
    results = []
    for d in grid:
        b_idx = select_boundary_cases(hidden, probes, d, cfg.n_boundary)[: cfg.n_swap_eval]
        for s in grid:
            acc = eval_steering_accuracy(model, tokenizer, records, steering_vectors,
                                         s, b_idx, cfg, device)
            results.append({"detect_layer": d, "steer_layer": s, "accuracy": acc})
            print(f"detect={d:2d}  steer={s:2d}  acc={acc:.3f}")

    os.makedirs(cfg.cache_dir, exist_ok=True)
    with open(os.path.join(cfg.cache_dir, "swap_grid_results.json"), "w") as f:
        json.dump(results, f)
    return pd.DataFrame(results)


def summarize_swap(df: pd.DataFrame, detect_layer: int, steer_layer: int) -> dict:
    same = df[df["detect_layer"] == df["steer_layer"]]["accuracy"].mean()
    matched = df[(df["detect_layer"] == detect_layer) & (df["steer_layer"] == steer_layer)]["accuracy"].mean()
    return {"same_layer_mean_acc": float(same), "matched_peak_acc": float(matched)}
