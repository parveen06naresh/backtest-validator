"""Signal families with parameter grids.

Every rule maps information available at the close of day t to a target
position for day t+1. The engine applies the one-day lag, so a rule can
never trade on a price it has not seen. Positions are in [-1, 1]
(1 = fully long the index, 0 = cash, -1 = fully short).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Rule:
    name: str
    family: str
    params: dict
    fn: Callable[[pd.Series], pd.Series]
    lookback: int  # longest window the rule reads, in trading days
    tags: set = field(default_factory=set)

    def positions(self, close: pd.Series) -> pd.Series:
        pos = self.fn(close).astype(float)
        pos.iloc[: self.lookback] = np.nan  # warm-up: no position yet
        return pos.clip(-1, 1)


def _sma(c, n):
    return c.rolling(n, min_periods=n).mean()


def _vol(c, n):
    return np.log(c).diff().rolling(n, min_periods=n).std() * np.sqrt(252)


def _ls(sig, long_short):
    return sig * 2 - 1 if long_short else sig


# --- rule constructors ------------------------------------------------------
def sma_trend(n, long_short=False):
    return Rule(f"sma{n}{'_ls' if long_short else ''}", "trend", {"n": n, "ls": long_short},
                lambda c: _ls((c > _sma(c, n)).astype(float), long_short), n)


def ma_cross(fast, slow, long_short=False):
    return Rule(f"cross{fast}x{slow}{'_ls' if long_short else ''}", "crossover",
                {"fast": fast, "slow": slow, "ls": long_short},
                lambda c: _ls((_sma(c, fast) > _sma(c, slow)).astype(float), long_short), slow)


def tsmom(n, long_short=False):
    return Rule(f"tsmom{n}{'_ls' if long_short else ''}", "momentum", {"n": n, "ls": long_short},
                lambda c: _ls((c / c.shift(n) > 1).astype(float), long_short), n)


def reversal(n):
    return Rule(f"rev{n}", "reversal", {"n": n},
                lambda c: (c / c.shift(n) < 1).astype(float), n)


def rsi(n, thr):
    def f(c):
        d = c.diff()
        up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
        r = 100 - 100 / (1 + up / dn.replace(0, np.nan))
        return (r < thr).astype(float)
    return Rule(f"rsi{n}<{thr}", "rsi", {"n": n, "thr": thr}, f, 3 * n)


def donchian(n):
    def f(c):
        hi = c.rolling(n).max()
        lo = c.rolling(max(n // 2, 2)).min()
        enter, exit_ = (c >= hi), (c <= lo)
        state = pd.Series(np.where(enter, 1.0, np.where(exit_, 0.0, np.nan)), index=c.index)
        return state.ffill().fillna(0.0)
    return Rule(f"donchian{n}", "breakout", {"n": n}, f, n)


def bollinger(n, k):
    return Rule(f"boll{n}k{k}", "meanrev", {"n": n, "k": k},
                lambda c: (c < _sma(c, n) - k * c.rolling(n).std()).astype(float), n)


def trend_voltarget(n=200, target=0.15, vol_n=20, band=0.10):
    """Crash Replay's quant: 200-day trend filter x 15% volatility target.

    Size = min(1, target / realised vol) while the close is above its n-day
    average, else 0. Only re-sizes when the size moves by ``band`` or more.
    """
    def f(c):
        raw = np.where(c > _sma(c, n), np.minimum(1.0, target / _vol(c, vol_n)), 0.0)
        raw = np.nan_to_num(raw)
        out = np.empty_like(raw)
        cur = 0.0
        for i, x in enumerate(raw):
            if abs(x - cur) >= band or (x == 0.0 and cur != 0.0):
                cur = x
            out[i] = cur
        return pd.Series(out, index=c.index)
    return Rule(f"trend{n}_vt{int(target * 100)}_{vol_n}", "trend_vol",
                {"n": n, "target": target, "vol_n": vol_n, "band": band}, f, max(n, vol_n + 1))


def turn_of_month(before=1, after=3):
    def f(c):
        idx = c.index
        ym = idx.year * 12 + idx.month
        s = pd.Series(ym, index=idx)
        pos_in = s.groupby(s).cumcount()                # 0 = first day of month
        left = s.groupby(s).transform("size") - pos_in  # 1 = last day of month
        in_window = (pos_in < after) | (left <= before)
        # decided at close t for day t+1 (the trading calendar is known ahead)
        return in_window.shift(-1, fill_value=False).astype(float)
    return Rule(f"tom{before}+{after}", "calendar", {"before": before, "after": after}, f, 25,
                tags={"calendar"})


def halloween():
    """Hold Nov-Apr, cash May-Oct ("sell in May")."""
    def f(c):
        nxt = c.index.shift(1, freq="B")
        return pd.Series(((nxt.month >= 11) | (nxt.month <= 4)).astype(float), index=c.index)
    return Rule("halloween", "calendar", {}, f, 1, tags={"calendar"})


def buy_and_hold():
    return Rule("buy_hold", "benchmark", {}, lambda c: pd.Series(1.0, index=c.index), 1)


# --- families ---------------------------------------------------------------
def standard_family() -> list[Rule]:
    """About 100 textbook rules: the kind of search a careful researcher runs."""
    R = []
    for n in (20, 50, 100, 150, 200, 250):
        R += [sma_trend(n), sma_trend(n, True)]
    for f in (5, 10, 20, 50):
        for s in (50, 100, 150, 200, 250):
            if f < s:
                R += [ma_cross(f, s), ma_cross(f, s, True)]
    for n in (21, 63, 126, 252):
        R += [tsmom(n), tsmom(n, True)]
    R += [reversal(n) for n in (1, 2, 3, 5, 10)]
    R += [rsi(n, t) for n in (2, 5, 14) for t in (10, 20, 30, 50)]
    R += [donchian(n) for n in (20, 55, 100)]
    R += [bollinger(n, k) for n in (10, 20, 50) for k in (1, 1.5, 2)]
    R += [trend_voltarget(n, t, v) for n in (100, 200) for t in (0.10, 0.15, 0.20) for v in (20, 60)]
    return R


def dense_family() -> list[Rule]:
    """A data-mining search: ~1,000 rules over fine parameter grids."""
    R = []
    for n in range(10, 301, 5):
        R += [sma_trend(n), sma_trend(n, True)]
    for f in (2, 3, 5, 8, 10, 15, 20, 30, 40, 50):
        for s in range(20, 301, 10):
            if f * 2 <= s:
                R += [ma_cross(f, s), ma_cross(f, s, True)]
    for n in range(5, 253, 7):
        R += [tsmom(n), tsmom(n, True)]
    R += [reversal(n) for n in range(1, 21)]
    R += [rsi(n, t) for n in (2, 3, 4, 5, 7, 10, 14, 21) for t in (5, 10, 15, 20, 25, 30, 40, 50)]
    R += [donchian(n) for n in range(10, 201, 10)]
    R += [bollinger(n, k) for n in (5, 10, 15, 20, 30, 50) for k in (0.5, 1, 1.5, 2, 2.5)]
    R += [trend_voltarget(n, t, v) for n in (50, 100, 150, 200, 250)
          for t in (0.10, 0.15, 0.20) for v in (10, 20, 60)]
    R += [turn_of_month(b, a) for b in (0, 1, 2) for a in (1, 2, 3, 4)] + [halloween()]
    return R


def positions_matrix(rules: list[Rule], close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({r.name: r.positions(close) for r in rules}, index=close.index)


def vol_target(target=0.15, vol_n=20, cap=1.0):
    """Always invested, sized to a volatility target: min(cap, target / vol)."""
    return Rule(f"vt{int(target * 100)}_{vol_n}_cap{cap:g}", "vol_target",
                {"target": target, "vol_n": vol_n, "cap": cap},
                lambda c: np.minimum(cap, target / _vol(c, vol_n)), vol_n + 1)


def vol_family() -> list[Rule]:
    return [vol_target(t, n, cap) for t in (0.10, 0.15, 0.20) for n in (10, 20, 60, 126)
            for cap in (1.0, 1.5)]
