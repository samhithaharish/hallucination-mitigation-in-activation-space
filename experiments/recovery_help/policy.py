"""Prediction-only resource allocation, with explicit batch and online modes."""
import numpy as np


def select(gains, costs, multiplier):
    return np.argmax(np.asarray(gains)-multiplier*np.asarray(costs), axis=1)


def allocate(gains, costs, budget):
    gains, costs = np.asarray(gains, dtype=float), np.asarray(costs, dtype=float)
    if gains.shape != costs.shape or gains.ndim != 2 or not np.isfinite(gains).all():
        raise ValueError('Finite matching matrices are required')
    if not np.isfinite(costs).all() or (costs[:, 1:] <= 0).any() or (costs[:, 0] != 0).any():
        raise ValueError('Keep must be free and recovery costs strictly positive')
    if (gains[:, 0] != 0).any() or budget < 0:
        raise ValueError('Keep gains must be zero and budget nonnegative')
    rows = np.arange(len(gains))
    choice = select(gains, costs, 0.)
    if costs[rows, choice].sum() <= budget:
        return choice, 0.
    lo, hi = 0., max(0., float(np.max(gains[:, 1:]/costs[:, 1:]))) + 1.
    best = np.zeros(len(gains), dtype=int)
    for _ in range(64):
        mid = (lo+hi)/2
        choice = select(gains, costs, mid)
        if costs[rows, choice].sum() <= budget:
            best, hi = choice, mid
        else:
            lo = mid
    return best, hi


def dual_upper_bound(gains, costs, budget, multiplier):
    """Upper bound on total achievable gain for these labels and this budget."""
    return float(multiplier*budget + np.max(gains-multiplier*costs, axis=1).sum())


def metrics(outcomes, exact, costs, choices):
    rows = np.arange(len(choices))
    initial = exact[:, 0] == 1
    final = exact[rows, choices] == 1
    return {'f1': float(outcomes[rows, choices].mean()), 'em': float(final.mean()),
        'f1_gain': float((outcomes[rows, choices]-outcomes[:, 0]).mean()),
        'rescue_rate': float(final[~initial].mean()) if (~initial).any() else None,
        'harm_rate': float((~final[initial]).mean()) if initial.any() else None,
        'reserved_tokens_per_question': float(costs[rows, choices].mean()),
        'action_counts': np.bincount(choices, minlength=outcomes.shape[1]).tolist()}


def paired_interval(values, seed=20261008, repeats=4000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(len(values), size=len(values))].mean() for _ in range(repeats)])
    return {'mean': float(values.mean()), 'low': float(np.quantile(means, .025)),
            'high': float(np.quantile(means, .975)), 'unit': 'question', 'replicates': repeats}
