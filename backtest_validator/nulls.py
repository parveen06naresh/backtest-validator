"""Synthetic markets for checking the battery itself.

``simulate`` draws daily returns from a GARCH(1,1) with Student-t shocks:
volatility clusters and tails are fat, like real index returns, but there
is no predictability in the mean unless ``signal`` is set. With
``drift=0`` and ``signal=0`` no rule can beat buy-and-hold on Sharpe, so
every "discovery" is a false positive.

``signal`` > 0 plants a real, known effect for power checks: the next
day's expected return is ``signal * sigma_t * s_t`` where s_t = +1 when
the price is above its 100-day average and -1 otherwise, so trend rules
truly work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Market


def simulate(n: int = 5040, seed: int = 0, drift: float = 0.0, signal: float = 0.0,
             omega: float = 1e-6, a: float = 0.08, b: float = 0.90, nu: float = 5.0,
             start: str = "1990-01-01", trend_n: int = 100) -> Market:
    rng = np.random.default_rng(seed)
    burn = 300
    N = n + burn
    z = rng.standard_t(nu, N) / np.sqrt(nu / (nu - 2))
    r = np.empty(N)
    h = omega / (1 - a - b)
    logp = np.zeros(N + 1)
    csum = np.r_[0.0, np.cumsum(np.zeros(N))]
    for t in range(N):
        sig = np.sqrt(h)
        mu = drift / 252
        if signal and t >= trend_n:
            ma = (csum[t] - csum[t - trend_n]) / trend_n
            mu += signal * sig * (1.0 if logp[t] > ma else -1.0)
        r[t] = mu + sig * z[t]
        logp[t + 1] = logp[t] + np.log1p(r[t])
        csum[t + 1] = csum[t] + logp[t + 1]
        h = omega + a * (r[t] - mu) ** 2 + b * h
    r = r[burn:]
    days = pd.bdate_range(start, periods=n)
    ret = pd.Series(r, index=days)               # simple returns, mean exactly drift
    close = pd.Series(100 * np.cumprod(1 + r), index=days)
    zero = pd.Series(0.0, index=days)
    return Market("synthetic", close, ret, zero, zero)
