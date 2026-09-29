"""Per-layer probes: which layer *detects* hallucination best (AUROC), plus
rank-based selection of boundary cases (probe is least certain)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def probe_sweep(hidden: np.ndarray, labels: np.ndarray, cfg):
    """Fit one logistic-regression probe per layer.

    Returns (layer_auroc, probes, (idx_train, idx_test), detect_layer).
    """
    n_layers_plus_embed = hidden.shape[1]
    idx_train, idx_test = train_test_split(
        np.arange(len(labels)), test_size=cfg.test_size,
        random_state=cfg.seed, stratify=labels,
    )
    layer_auroc, probes = [], {}
    for layer in range(n_layers_plus_embed):
        X = hidden[:, layer, :]
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X[idx_train], labels[idx_train])
        p = clf.predict_proba(X[idx_test])[:, 1]
        layer_auroc.append(roc_auc_score(labels[idx_test], p))
        probes[layer] = clf
    layer_auroc = np.array(layer_auroc)
    detect_layer = int(np.argmax(layer_auroc))
    return layer_auroc, probes, (idx_train, idx_test), detect_layer


def select_boundary_cases(hidden, probes, layer: int, n: int) -> np.ndarray:
    """Rank-based: the n examples whose probe probability is closest to 0.5.

    (A fixed probability window like [0.35, 0.65] returned empty sets whenever
    a layer's probe was confident, which broke the sweep. Ranking always
    returns n examples.)
    """
    probs = probes[layer].predict_proba(hidden[:, layer, :])[:, 1]
    return np.argsort(np.abs(probs - 0.5))[:n]
