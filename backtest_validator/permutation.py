"""Permutation tests for timing skill that keep autocorrelation.

Question: does *when* the rule is invested matter, or would any series of
positions with the same shape have done as well?

* ``circular_shift_test``: slide the whole position series against the
  returns by s days (wrapping around). The positions keep their exact
  autocorrelation, holding periods and average exposure; the returns keep
  theirs; only the alignment is broken. Shifts are limited to
  [lookback + 1, n - lookback - 1] so a shifted position never uses the
  return it is paired with (no look-ahead). All shifts are evaluated at
  once with FFTs, so the test is exact over its permutation set.
* ``random_trader_test``: the Crash Replay skill test. Random traders who
  switch in and out the same number of times, with the same time in the
  market, at random dates.

Both test gross timing value; costs are checked separately.
"""
from __future__ import annotations

import numpy as np


def _circ_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """out[s] = sum_t a[(t - s) mod n] * b[t] for every shift s."""
    fa, fb = np.fft.rfft(a), np.fft.rfft(b)
    return np.fft.irfft(np.conj(fa) * fb, n=len(a))


def circular_shift_test(pos: np.ndarray, excess: np.ndarray, lookback: int,
                        lag: int = 1) -> dict:
    """p-value for the Sharpe of ``pos`` (lagged) times ``excess``.

    The statistic is the annualised Sharpe of w_{t-lag} * x_t. The null set
    is every admissible circular shift of the lagged positions.
    """
    p = np.nan_to_num(np.asarray(pos, float))
    x = np.asarray(excess, float)
    w = np.r_[np.zeros(lag), p[:-lag]] if lag else p
    n = len(x)
    s1 = _circ_corr(w, x)
    s2 = _circ_corr(w ** 2, x ** 2)
    mean = s1 / n
    var = np.maximum(s2 / n - mean ** 2, 1e-30) * n / (n - 1)
    srs = mean / np.sqrt(var) * np.sqrt(252)
    obs = srs[0]
    lo, hi = lookback + lag + 1, n - lookback - lag - 1
    null = srs[lo:hi] if hi > lo else srs[1:]
    pval = (1 + np.sum(null >= obs)) / (len(null) + 1)
    return {"sharpe": float(obs), "p_value": float(pval),
            "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "n_shifts": int(len(null))}


def random_trader_test(pos: np.ndarray, excess: np.ndarray, n_traders: int = 2000,
                       lag: int = 1, seed: int = 0) -> dict:
    """Crash Replay's test: beat how many random traders with equal activity?

    Each random trader is in or out (0/1). It switches at the same number of
    random dates as the rule and is invested for about the same share of
    days. The statistic is total excess return, as in the game.
    """
    rng = np.random.default_rng(seed)
    p = np.nan_to_num(np.asarray(pos, float))
    x = np.asarray(excess, float)
    n = len(x)
    w = np.r_[np.zeros(lag), p[:-lag]] if lag else p
    obs = float(np.sum(w * x))
    exposure = float(np.mean(w > 0))
    switches = int(max(np.sum(np.abs(np.diff((w > 0).astype(int)))), 1))
    size = float(np.mean(np.abs(w[w != 0]))) if np.any(w != 0) else 0.0
    out = np.empty(n_traders)
    for i in range(n_traders):
        flips = np.zeros(n, dtype=np.int64)
        flips[rng.choice(np.arange(1, n), size=min(switches, n - 1), replace=False)] = 1
        start_in = int(rng.random() < exposure)
        state = (np.cumsum(flips) + start_in) % 2
        out[i] = size * np.dot(state, x)
    beat = float(np.mean(out < obs))
    return {"total_excess": obs, "beat_share": beat,
            "p_value": float((1 + np.sum(out >= obs)) / (n_traders + 1)),
            "switches": switches, "exposure": exposure}
