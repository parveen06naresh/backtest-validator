import numpy as np

from backtest_validator import Config, load, nulls, validate
from backtest_validator import strategies as S
from backtest_validator.cli import main


def test_data_snapshot_loads():
    m = load("spx")
    assert m.index.is_monotonic_increasing
    assert not m.ret.isna().any() and not m.rf.isna().any()
    assert str(m.index[0].date()) == "1950-01-04"
    assert 0.08 < m.ret.mean() * 252 < 0.16  # total return about 12%/yr
    assert 0.02 < m.rf.mean() * 252 < 0.06


def test_validate_null_market_fails():
    m = nulls.simulate(2520, seed=11)
    fam = [S.sma_trend(n) for n in (20, 50, 100)] + [S.tsmom(63), S.reversal(2)]
    r = validate(m, fam, "best", cfg=Config(reps=200, capacity=False))
    assert not r.survives
    assert set(r.gates) == {"costs", "timing", "selection", "overfit", "stability", "delay",
                            "holdout", "conformal"}
    assert "Verdict: FAILS" in r.text()


def test_validate_planted_signal_is_found():
    m = nulls.simulate(7560, seed=13, signal=0.15)
    fam = [S.sma_trend(n, True) for n in (50, 100, 200)] + [S.reversal(2), S.rsi(5, 30)]
    r = validate(m, fam, "best", cfg=Config(reps=300, capacity=False))
    assert r.gates["selection"]["pass"], r.text()
    assert r.details["selection"]["edge_sharpe"] > 0.5


def test_cli_runs(capsys):
    assert main(["rules", "--family", "standard"]) == 0
    assert "trend200_vt15_20" in capsys.readouterr().out
    assert main(["validate", "--market", "synthetic", "--family", "vol", "--reps", "100"]) == 0
    assert "Verdict" in capsys.readouterr().out


def test_families_have_unique_names():
    for fam in (S.standard_family(), S.dense_family(), S.vol_family()):
        names = [r.name for r in fam]
        assert len(names) == len(set(names))
    assert len(S.dense_family()) > 800
    assert np.isfinite(len(S.standard_family()))
