import numpy as np
import pandas as pd
import pytest

from backtest_validator import engine, nulls
from backtest_validator import strategies as S
from backtest_validator.conformal import _q
from backtest_validator.permutation import circular_shift_test


def _market(n=1500, seed=0):
    return nulls.simulate(n, seed=seed)


def test_one_day_lag_and_costs():
    idx = pd.bdate_range("2020-01-01", periods=5)
    x = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01], index=idx)
    rf = pd.Series(0.0, index=idx)
    pos = pd.Series([0, 1, 1, 0, 0], index=idx, dtype=float)
    gross = engine.run(pos, x, rf, engine.ZERO)
    # bought at the close of day 1, so earns days 2 and 3 only
    assert list(gross.round(6)) == [0, 0, -0.01, 0.03, 0]
    c = engine.Costs(half_spread=1e-3, fee=0, impact_coef=0)
    net = engine.run(pos, x, rf, c)
    assert (gross - net).round(6).tolist() == [0, 0, 0.001, 0, 0.001]


def test_impact_grows_with_size():
    m = _market()
    pos = S.sma_trend(50).positions(m.close)
    small = engine.run(pos, m.excess, m.rf, engine.Costs(aum=1e6)).sum()
    big = engine.run(pos, m.excess, m.rf, engine.Costs(aum=1e10)).sum()
    assert big < small


@pytest.mark.parametrize("rule", [S.sma_trend(50), S.ma_cross(10, 50), S.tsmom(63), S.rsi(5, 30),
                                  S.donchian(20), S.bollinger(20, 1), S.trend_voltarget(),
                                  S.turn_of_month(1, 3), S.reversal(3)])
def test_rules_never_look_ahead(rule):
    m = _market()
    c = m.close.copy()
    base = rule.positions(c)
    t = 900
    c2 = c.copy()
    c2.iloc[t + 1:] *= np.exp(np.random.default_rng(1).normal(0, 0.05, len(c) - t - 1)).cumprod()
    alt = rule.positions(c2)
    pd.testing.assert_series_equal(base.iloc[: t + 1], alt.iloc[: t + 1])


def test_turn_of_month_window():
    idx = pd.bdate_range("2021-01-01", "2021-03-31")
    c = pd.Series(100.0, index=idx)
    pos = S.turn_of_month(1, 3).fn(c)
    held = pos.shift(1).fillna(0)  # what is held on each day
    feb = held.loc["2021-02"]
    assert feb.iloc[:3].tolist() == [1, 1, 1] and feb.iloc[3] == 0
    assert held.loc["2021-01-29"] == 1 and held.loc["2021-01-28"] == 0


def test_circular_shift_catches_cheating_and_passes_noise():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.01, 3000)
    cheat = np.r_[np.sign(x[1:]), 0]  # knows tomorrow's return
    assert circular_shift_test(cheat, x, 1)["p_value"] < 0.01
    ps = [circular_shift_test(np.sign(rng.normal(size=3000)).cumsum() > 0, rng.normal(0, 0.01, 3000),
                              1)["p_value"] for _ in range(200)]
    assert 0.01 < np.mean(np.array(ps) < 0.05) < 0.11


def test_conformal_quantile_coverage_iid():
    rng = np.random.default_rng(3)
    miss = []
    for _ in range(2000):
        s = rng.normal(size=40)
        miss.append(rng.normal() > _q(s, 0.8))
    assert np.mean(miss) == pytest.approx(0.2, abs=0.03)
