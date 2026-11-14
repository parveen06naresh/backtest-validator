"""Probability of Backtest Overfitting via CSCV.

Bailey, Borwein, Lopez de Prado and Zhu (2017). Split the sample into S
equal blocks. For every way of choosing S/2 blocks as in-sample (IS), pick
the configuration with the best IS Sharpe and find its rank among all
configurations on the other half (OOS). PBO is the share of splits where
the IS winner lands in the bottom half OOS. With no real edge PBO is near
0.5; a value near 0 means the IS winner tends to stay good.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import rankdata


def pbo(R: np.ndarray, blocks: int = 16, max_splits: int | None = None,
        seed: int = 0) -> dict:
    R = np.asarray(R, float)
    T, N = R.shape
    if blocks % 2:
        raise ValueError("blocks must be even")
    size = T // blocks
    R = R[: size * blocks]
    B = R.reshape(blocks, size, N)
    s1, s2 = B.sum(axis=1), (B ** 2).sum(axis=1)       # per-block moments
    splits = list(combinations(range(blocks), blocks // 2))
    if max_splits and len(splits) > max_splits:
        rng = np.random.default_rng(seed)
        splits = [splits[i] for i in rng.choice(len(splits), max_splits, replace=False)]
    allb = np.arange(blocks)
    n_half = size * blocks // 2

    def sr(sel):
        m = s1[list(sel)].sum(0) / n_half
        v = s2[list(sel)].sum(0) / n_half - m ** 2
        return m / np.sqrt(np.maximum(v, 1e-30))

    logits, is_best, oos_best = [], [], []
    for sel in splits:
        oos = np.setdiff1d(allb, sel)
        a, b = sr(sel), sr(oos)
        k = int(np.argmax(a))
        w = rankdata(b)[k] / (N + 1)
        logits.append(np.log(w / (1 - w)))
        is_best.append(a[k])
        oos_best.append(b[k])
    logits = np.array(logits)
    is_best, oos_best = np.array(is_best) * np.sqrt(252), np.array(oos_best) * np.sqrt(252)
    slope = float(np.polyfit(is_best, oos_best, 1)[0]) if len(set(is_best)) > 1 else float("nan")
    return {"pbo": float(np.mean(logits <= 0)), "logits": logits,
            "is_sharpe": is_best, "oos_sharpe": oos_best,
            "degradation_slope": slope, "p_oos_loss": float(np.mean(oos_best < 0)),
            "splits": len(splits)}
