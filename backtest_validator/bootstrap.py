"""Block bootstraps that keep the time-series dependence of returns.

* Stationary bootstrap (Politis and Romano 1994): blocks of random,
  geometrically distributed length, wrapped around the sample.
* Automatic mean block length (Politis and White 2004, with the correction
  of Patton, Politis and White 2009).

Resampling whole rows keeps the cross-sectional dependence between
strategies, which the Reality Check, SPA and Romano-Wolf tests need.
"""
from __future__ import annotations

import numpy as np


def optimal_block_length(x: np.ndarray) -> float:
    """Politis-White (2004) mean block length for the stationary bootstrap."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 20:
        return 1.0
    x = x - x.mean()
    kn = max(5, int(np.ceil(np.log10(n))))
    mmax = int(np.ceil(np.sqrt(n))) + kn
    bmax = int(np.ceil(min(3 * np.sqrt(n), n / 3)))
    c = 2.0
    var = x @ x / n
    acv = np.array([x[: n - k] @ x[k:] / n for k in range(mmax + 1)])
    rho = acv[1:] / var if var > 0 else np.zeros(mmax)
    thresh = c * np.sqrt(np.log10(n) / n)
    mhat = None
    for m in range(mmax - kn + 1):
        if np.all(np.abs(rho[m: m + kn]) < thresh):
            mhat = m
            break
    if mhat is None:
        mhat = mmax - kn
    M = min(2 * max(mhat, 1), mmax)
    k = np.arange(-M, M + 1)
    lam = np.clip(2 * (1 - np.abs(k / M)), 0, 1)  # flat-top kernel
    R = acv[np.abs(k)]
    G = np.sum(lam * np.abs(k) * R)
    g0 = np.sum(lam * R)
    D = 2 * g0 ** 2
    if D <= 0:
        return 1.0
    b = (2 * G ** 2 / D) ** (1 / 3) * n ** (1 / 3)
    return float(np.clip(b, 1.0, bmax))


def stationary_indices(n: int, block: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """``reps`` x ``n`` matrix of stationary-bootstrap row indices."""
    p = 1.0 / max(block, 1.0)
    idx = np.empty((reps, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, reps)
    jump = rng.random((reps, n)) < p
    fresh = rng.integers(0, n, (reps, n))
    for t in range(1, n):
        idx[:, t] = np.where(jump[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n)
    return idx


def counts(n: int, block: float, reps: int, rng: np.random.Generator) -> np.ndarray:
    """``reps`` x ``n`` matrix: how often each row appears in each resample.

    Means of every resample for every column are then ``counts @ X / n``,
    one matrix product instead of a Python loop.
    """
    idx = stationary_indices(n, block, reps, rng)
    flat = (idx + (np.arange(reps)[:, None] * n)).ravel()
    return np.bincount(flat, minlength=reps * n).reshape(reps, n).astype(np.float32)


def bootstrap_means(X: np.ndarray, block: float | None = None, reps: int = 1000,
                    seed: int = 0) -> tuple[np.ndarray, float]:
    """Bootstrap distribution of column means. Returns (reps x k, block)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if block is None:
        block = float(np.median([optimal_block_length(X[:, j])
                                 for j in range(0, X.shape[1], max(1, X.shape[1] // 25))]))
    C = counts(n, block, reps, np.random.default_rng(seed))
    return (C.astype(float) @ X) / n, block


def mean_pvalues(X: np.ndarray, block: float | None = None, reps: int = 1000,
                 seed: int = 0) -> np.ndarray:
    """One-sided p-value per column for H0: mean <= 0 (recentred bootstrap)."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    M, _ = bootstrap_means(X, block, reps, seed)
    m = X.mean(axis=0)
    return (1 + np.sum(M - m >= m, axis=0)) / (reps + 1)
