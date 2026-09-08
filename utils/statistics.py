import math
from collections import defaultdict

import numpy as np


def pass_at_k(n, c, k):
    if n < k:
        raise ValueError('n must be at least k')
    if c == 0:
        return 0.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_at_k_metrics(n, c):
    """Return pass@k for powers of two up to n."""
    result = {}
    k = 1
    while k <= n:
        result[k] = pass_at_k(n, c, k)
        k *= 2
    return result


def bootstrap_mean_ci(values, num_bootstrap=10000, alpha=0.05, seed=0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float('nan'), float('nan')
    if len(values) == 1:
        value = float(values[0])
        return value, value

    rng = np.random.default_rng(seed)
    means = np.empty(num_bootstrap, dtype=np.float64)
    for index in range(num_bootstrap):
        sample = rng.integers(0, len(values), size=len(values))
        means[index] = values[sample].mean()
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def pass_at_k_samples(returns, group_ids, prefix, labels=None):
    returns = np.asarray(returns)
    group_ids = np.asarray(group_ids)
    assert len(returns) == len(group_ids)
    labels = None if labels is None else np.asarray(labels)
    assert labels is None or len(labels) == len(returns)
    samples = defaultdict(list)
    for group_id in np.unique(group_ids):
        mask = group_ids == group_id
        if labels is not None:
            assert len(np.unique(labels[mask])) == 1, 'labels must be constant within each pass@k group'
        group_returns = returns[mask]
        successes = int((group_returns >= 1.0).sum())
        for k, value in pass_at_k_metrics(len(group_returns), successes).items():
            samples[f'{prefix}pass@{k}'].append(value)
    return samples


def append_pass_at_k(history, returns, group_ids, prefix, labels=None):
    for key, values in pass_at_k_samples(returns, group_ids, prefix, labels).items():
        history[key].extend(values)


def pass_at_k_confidence_intervals(history, prefix='env/'):
    result = {}
    for name, values in history.items():
        if not (name.startswith('env_return_pass@') or ('/pass@' in name and name.rsplit('/pass@', 1)[1].isdigit())):
            continue
        low, high = bootstrap_mean_ci(values)
        result[f'{prefix}{name}_ci_low'] = low
        result[f'{prefix}{name}_ci_high'] = high
    return result
