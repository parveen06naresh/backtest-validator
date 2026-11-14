"""Regime and non-stationarity checks.

A real effect should not live in one decade or one kind of market. These
checks split the active return (strategy minus benchmark, both scaled to
the same volatility) by regime and over time.

* ``by_regime``: Sharpe of the active return within each label.
* ``rolling_edge``: rolling Sharpe edge and the share of windows above 0.
* ``cusum_break``: OLS-CUSUM test for a shift in the mean active return
  (Ploberger and Kramer 1992), with a Newey-West long-run variance so
  autocorrelation does not inflate it. 5% critical value 1.358.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sharpe


def decade_labels(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series((index.year // 10 * 10).astype(str) + "s", index=index)


def half_labels(index: pd.DatetimeIndex) -> pd.Series:
    mid = index[len(index) // 2]
    return pd.Series(np.where(index < mid, "first half", "second half"), index=index)


def trend_labels(close: pd.Series, n: int = 200) -> pd.Series:
    up = close.shift(1) > close.shift(1).rolling(n).mean()
    return pd.Series(np.where(up, "bull (above 200d)", "bear (below 200d)"), index=close.index)


def vol_labels(excess: pd.Series, n: int = 63) -> pd.Series:
    v = excess.rolling(n).std().shift(1)
    q = v.expanding(252).quantile(0.67)
    return pd.Series(np.where(v > q, "high vol", "normal vol"), index=excess.index)


def by_regime(active: pd.Series, labels: pd.Series, min_days: int = 120) -> pd.DataFrame:
    df = pd.DataFrame({"a": active, "g": labels.reindex(active.index)}).dropna()
    rows = []
    for g, part in df.groupby("g", sort=True):
        if len(part) >= min_days:
            rows.append({"regime": g, "days": len(part), "edge_sharpe": sharpe(part["a"])})
    return pd.DataFrame(rows)


def rolling_edge(active: pd.Series, window: int = 756) -> dict:
    m = active.rolling(window).mean()
    s = active.rolling(window).std()
    roll = (m / s * np.sqrt(252)).dropna()
    # non-overlapping windows for an honest count
    blocks = [sharpe(active.iloc[i:i + window]) for i in range(0, len(active) - window + 1, window)]
    return {"series": roll, "share_positive": float(np.mean(roll > 0)) if len(roll) else float("nan"),
            "blocks": blocks, "share_blocks_positive": float(np.mean(np.array(blocks) > 0))
            if blocks else float("nan")}


def newey_west_var(x: np.ndarray, lags: int | None = None) -> float:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    L = lags if lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = x @ x / n
    for k in range(1, L + 1):
        v += 2 * (1 - k / (L + 1)) * (x[k:] @ x[:-k]) / n
    return float(max(v, 1e-30))


def cusum_break(active: pd.Series) -> dict:
    x = np.asarray(active.dropna(), float)
    n = len(x)
    s = np.cumsum(x - x.mean()) / np.sqrt(n * newey_west_var(x))
    k = int(np.argmax(np.abs(s)))
    stat = float(np.abs(s[k]))
    # Kolmogorov distribution tail for sup |Brownian bridge|
    j = np.arange(1, 101)
    p = float(np.clip(2 * np.sum((-1) ** (j - 1) * np.exp(-2 * j ** 2 * stat ** 2)), 0, 1))
    return {"stat": stat, "p_value": p, "break": stat > 1.358,
            "break_date": str(active.dropna().index[k].date())}
