import numpy as np
import pytest

from backtest_validator import metrics
from backtest_validator.bootstrap import mean_pvalues, optimal_block_length, stationary_indices
from backtest_validator.multiple_testing import SnoopingBootstrap, benjamini_hochberg, bonferroni, holm
from backtest_validator.pbo import pbo


def test_adjustments_known_values():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    assert np.allclose(bonferroni(p), [0.04, 0.16, 0.12, 0.02])
    # Holm: sorted 0.005,0.01,0.03,0.04 -> 0.02,0.03,0.06,0.06
    assert np.allclose(holm(p), [0.03, 0.06, 0.06, 0.02])
    # BH: sorted p*m/rank = 0.02,0.02,0.04,0.04 -> monotone from the top
    assert np.allclose(benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02])


def test_adjustment_ordering():
    rng = np.random.default_rng(0)
    p = rng.random(50) ** 3
    assert np.all(bonferroni(p) >= holm(p) - 1e-12)
    assert np.all(holm(p) >= benjamini_hochberg(p) - 1e-12)


def test_psr_and_dsr_behave():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0005, 0.01, 5000)
    assert 0.5 < metrics.psr(x) < 1
    assert metrics.psr(-x) < 0.5
    srs = rng.normal(0, 0.014, 100)
    a = metrics.deflated_sharpe(x, srs, n_trials=10)["dsr"]
    b = metrics.deflated_sharpe(x, srs, n_trials=1000)["dsr"]
    assert b < a <= metrics.psr(x)


def test_expected_max_sharpe_matches_simulation():
    rng = np.random.default_rng(2)
    n, sd = 200, 0.02
    sim = rng.normal(0, sd, (4000, n)).max(axis=1).mean()
    assert metrics.expected_max_sr(n, sd ** 2) == pytest.approx(sim, rel=0.03)


def test_block_length_reacts_to_dependence():
    rng = np.random.default_rng(3)
    iid = rng.normal(size=3000)
    ar = np.zeros(3000)
    for t in range(1, 3000):
        ar[t] = 0.8 * ar[t - 1] + rng.normal()
    assert optimal_block_length(iid) < 5
    assert optimal_block_length(ar) > 3 * optimal_block_length(iid)


def test_stationary_bootstrap_mean_block():
    idx = stationary_indices(5000, 10.0, 20, np.random.default_rng(4))
    breaks = np.mean(np.diff(idx, axis=1) != 1)
    assert 1 / breaks == pytest.approx(10.0, rel=0.1)


def test_bootstrap_pvalues_are_uniformish_under_null():
    rng = np.random.default_rng(5)
    X = rng.standard_t(5, size=(1000, 200)) * 0.01
    p = mean_pvalues(X, block=2.0, reps=400)
    assert 0.02 < np.mean(p < 0.05) < 0.10


def test_reality_check_and_spa():
    rng = np.random.default_rng(6)
    null = rng.normal(0, 0.01, (2500, 40))
    sb = SnoopingBootstrap(null, reps=500, block=2.0)
    assert sb.reality_check() > 0.05 and sb.spa()["consistent"] > 0.05
    good = null.copy()
    good[:, 0] += 0.002  # a strong edge (t ~ 10)
    sb = SnoopingBootstrap(good, reps=500, block=2.0)
    assert sb.reality_check() < 0.01 and sb.spa()["consistent"] < 0.01
    assert 0 in sb.stepm(0.05)["rejected"]


def test_pbo_extremes():
    rng = np.random.default_rng(7)
    avg = np.mean([pbo(rng.normal(0, 0.01, (3200, 30)), 8)["pbo"] for _ in range(8)])
    assert 0.35 < avg < 0.65  # no edge: the IS winner is a coin flip OOS
    noise = rng.normal(0, 0.01, (3200, 30))
    edge = noise.copy()
    edge[:, 3] += 0.003
    assert pbo(edge, 8)["pbo"] < 0.05
