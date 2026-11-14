"""Load the cached public data snapshot.

Everything the case studies use ships in ``data/raw`` so results reproduce
offline. Sources (all public): S&P 500, Nasdaq Composite and Dow daily
OHLCV and the VIX from Yahoo Finance; 3-month T-bill (TB3MS), 10-year yield
(DGS10) and NBER recession dates (USREC) from FRED; monthly S&P 500
dividends from Robert Shiller's data. ``scripts/fetch_data.py`` refreshes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
INDEXES = {"spx": "SPX.csv", "ndq": "NDQ.csv", "dji": "DJI.csv"}


@dataclass
class Market:
    """Daily data for one index, aligned on trading days.

    ``ret`` is the total return earned from the previous close to this close
    (price change plus an accrued dividend yield where known). ``rf`` is the
    cash return over the same interval. ``excess`` = ``ret - rf``.
    """

    name: str
    close: pd.Series
    ret: pd.Series
    rf: pd.Series
    volume: pd.Series

    @property
    def excess(self) -> pd.Series:
        return self.ret - self.rf

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.close.index

    def slice(self, start=None, end=None) -> "Market":
        s = slice(start, end)
        return Market(self.name, self.close.loc[s], self.ret.loc[s],
                      self.rf.loc[s], self.volume.loc[s])


def _fred(name: str) -> pd.Series:
    df = pd.read_csv(RAW / f"{name}.csv", na_values=".")
    s = pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]),
                  name=name).dropna()
    return s.astype(float)


def _monthly_lagged(series: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Value of the previous month on each day, so nothing is used early."""
    s = series.copy()
    s.index = s.index + pd.offsets.MonthBegin(1)
    return s.reindex(days, method="ffill")


def dividend_yield(days: pd.DatetimeIndex) -> pd.Series:
    """Annual S&P 500 dividend yield (fraction), last month's Shiller data."""
    sh = pd.read_csv(RAW / "SHILLER.csv")
    sh.index = pd.to_datetime(sh["Date"])
    div = sh["Dividend"].where(sh["Dividend"] > 0).ffill()
    dy = (div / sh["SP500"]).dropna()
    return _monthly_lagged(dy, days).fillna(0.0)


def recessions(days: pd.DatetimeIndex) -> pd.Series:
    rec = _fred("USREC")
    rec.index = rec.index.to_period("M").to_timestamp()
    return rec.reindex(days.to_period("M").to_timestamp()).fillna(0).set_axis(days)


def vix(days: pd.DatetimeIndex) -> pd.Series:
    v = pd.read_csv(RAW / "VIX.csv", index_col=0, parse_dates=True)["close"]
    return v.reindex(days)


def load(name: str = "spx", start: str = "1950-01-03", end: str | None = None) -> Market:
    """Load one index. ``name`` is ``spx``, ``ndq`` or ``dji``."""
    df = pd.read_csv(RAW / INDEXES[name], index_col=0, parse_dates=True)
    df = df[df["close"] > 0].sort_index()
    df = df.loc[start:end]
    days = df.index
    close = df["close"].astype(float)
    gap = np.diff(days.values).astype("timedelta64[D]").astype(float)
    gap = pd.Series(np.r_[np.nan, gap], index=days)
    price_ret = close.pct_change()
    if name == "spx":
        dy = dividend_yield(days)
        div_ret = dy.shift(1) * gap / 365.0
    else:  # no clean dividend history for the others: price return only
        div_ret = pd.Series(0.0, index=days)
    tb = _monthly_lagged(_fred("TB3MS") / 100.0, days)
    rf = tb.shift(1) * gap / 365.0
    ret = (price_ret + div_ret).iloc[1:]
    rf = rf.iloc[1:].fillna(0.0)
    return Market(name, close.iloc[1:], ret, rf, df["volume"].astype(float).iloc[1:])
