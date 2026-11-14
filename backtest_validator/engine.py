"""Turn positions into daily returns, with explicit costs.

Timing: a position chosen at the close of day t earns the return from the
close of t to the close of t + ``lag``. Cash earns the T-bill rate. A short
position pays the index return and still earns cash on the proceeds.

Cost model (per unit of turnover |w_t - w_{t-1}|, as a fraction of capital):

    cost = half_spread + fee + impact_coef * sigma_daily * sqrt(trade / ADV)

The last term is the square-root market-impact law (Almgren et al. 2005,
Toth et al. 2011): impact grows with the square root of the order's share
of daily volume, scaled by daily volatility. ``trade`` is |dw| * AUM in
dollars and ADV is the dollar volume of the instrument used to trade the
index (an ETF or future), held fixed in today's dollars.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    half_spread: float = 1e-4     # 1 bp
    fee: float = 0.5e-4           # 0.5 bp
    impact_coef: float = 1.0      # Y in the square-root law
    adv: float = 20e9             # $ daily volume of the traded vehicle
    aum: float = 10e6             # $ capital

    def scaled(self, k: float) -> "Costs":
        """Same model with spread and fee multiplied by ``k`` (stress test)."""
        return Costs(self.half_spread * k, self.fee * k, self.impact_coef * k, self.adv, self.aum)

    def with_aum(self, aum: float) -> "Costs":
        return Costs(self.half_spread, self.fee, self.impact_coef, self.adv, aum)


ZERO = Costs(0.0, 0.0, 0.0)
HARSH = Costs(5e-4, 2.5e-4, 2.0)  # 7.5 bp a side plus 2x impact


def daily_vol(excess: pd.Series, n: int = 20) -> pd.Series:
    return excess.rolling(n, min_periods=5).std().bfill()


def run(pos: pd.Series | pd.DataFrame, excess: pd.Series, rf: pd.Series,
        costs: Costs = Costs(), lag: int = 1) -> pd.DataFrame | pd.Series:
    """Net excess returns (over cash) of one or many position series.

    ``pos`` is indexed like ``excess``; NaN positions (warm-up) count as cash.
    Returns the same shape as ``pos``. The first ``lag`` rows are zero.
    """
    one = isinstance(pos, pd.Series)
    P = pos.to_frame() if one else pos
    w = P.fillna(0.0).shift(lag).fillna(0.0)
    gross = w.mul(excess, axis=0)
    turn = w.diff().abs().fillna(w.abs())
    sig = daily_vol(excess).reindex(w.index).values[:, None]
    unit = costs.half_spread + costs.fee + costs.impact_coef * sig * np.sqrt(
        np.maximum(turn.values, 0) * costs.aum / costs.adv)
    net = gross - turn * unit
    return net.iloc[:, 0] if one else net


def turnover(pos: pd.Series) -> float:
    """Average one-way turnover per year (1.0 = full in-and-out once a year)."""
    w = pos.fillna(0.0)
    return float(w.diff().abs().sum() / max(len(w), 1) * 252)


def capacity(pos: pd.Series, bench_pos: pd.Series, excess: pd.Series, rf: pd.Series,
             costs: Costs = Costs(), grid=None) -> dict:
    """Largest AUM at which the rule still beats the benchmark on Sharpe.

    Scans AUM on a log grid; returns the last AUM with a positive Sharpe
    edge, or None if it never has one (even at zero size).
    """
    from .metrics import sharpe
    grid = grid if grid is not None else np.logspace(5, 12, 29)
    b = sharpe(run(bench_pos, excess, rf, costs.with_aum(1e5)))
    rows, last_ok, broken = [], None, False
    for a in grid:
        s = sharpe(run(pos, excess, rf, costs.with_aum(a)))
        rows.append((float(a), float(s - b)))
        if s - b > 0 and not broken:
            last_ok = float(a)
        else:
            broken = True
    return {"capacity_aum": last_ok, "curve": rows}
