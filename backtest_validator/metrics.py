"""Sharpe ratio statistics: PSR, Deflated Sharpe, minimum track record.

References: Bailey and Lopez de Prado, "The Sharpe Ratio Efficient
Frontier" (2012) and "The Deflated Sharpe Ratio" (2014). Inside these
formulas Sharpe ratios are per period (daily), not annualised.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER = 0.5772156649015329
ANN = 252


def _arr(x):
    a = np.asarray(x, dtype=float)
    return a[~np.isnan(a)]


def sharpe(x, annualise: bool = True) -> float:
    a = _arr(x)
    if len(a) < 2:
        return 0.0
    sd = a.std(ddof=1)
    if sd == 0:
        return 0.0
    sr = a.mean() / sd
    return float(sr * np.sqrt(ANN)) if annualise else float(sr)


def sharpe_cols(X: np.ndarray, annualise: bool = True) -> np.ndarray:
    sd = X.std(axis=0, ddof=1)
    sr = np.divide(X.mean(axis=0), sd, out=np.zeros(X.shape[1]), where=sd > 0)
    return sr * np.sqrt(ANN) if annualise else sr


def _sr_se(sr: float, skew: float, kurt: float, n: int) -> float:
    """Std. error of a per-period Sharpe estimate under non-normal returns."""
    v = (1 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)
    return float(np.sqrt(max(v, 1e-18)))


def psr(x, sr_star: float = 0.0) -> float:
    """Probabilistic Sharpe Ratio: P(true SR > sr_star) given the sample.

    ``sr_star`` is per period (daily).
    """
    a = _arr(x)
    sr = sharpe(a, annualise=False)
    se = _sr_se(sr, stats.skew(a), stats.kurtosis(a, fisher=False), len(a))
    return float(stats.norm.cdf((sr - sr_star) / se))


def expected_max_sr(n_trials: float, var_sr: float) -> float:
    """E[max of n_trials Sharpe estimates] when every true Sharpe is zero."""
    n = max(float(n_trials), 1.0)
    if n <= 1:
        return 0.0
    z1 = stats.norm.ppf(1 - 1 / n)
    z2 = stats.norm.ppf(1 - 1 / (n * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2))


def effective_trials(R: np.ndarray) -> float:
    """Effective number of independent trials from the trials' correlation.

    Uses N_eff = rho + (1 - rho) * N with rho the mean pairwise correlation
    (clipped at 0), a common approximation; correlated variants of one rule
    count for less than independent ideas.
    """
    n = R.shape[1]
    if n < 2:
        return 1.0
    C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C)
    rho = (C.sum() - np.trace(C)) / (n * (n - 1))
    rho = float(np.clip(rho, 0, 1))
    return rho + (1 - rho) * n


def deflated_sharpe(x, trial_srs, n_trials: float | None = None,
                    var_method: str = "sampling") -> dict:
    """Deflated Sharpe Ratio of the selected strategy ``x``.

    ``trial_srs`` are the per-period Sharpe ratios of every configuration
    tried (including the selected one). Returns the DSR (a probability) and
    the hurdle SR0 it was tested against.

    ``var_method`` sets V[SR] in the hurdle. "trials" is Bailey and Lopez de
    Prado's choice, the cross-sectional variance of the trial Sharpes; it
    grows when the family contains rules that are truly bad, which raises
    the bar for everyone. "sampling" uses the variance of a Sharpe estimate
    of this length when the true Sharpe is zero, i.e. the bar that pure luck
    sets. We gate on "sampling" (the less strict one) and report both.
    """
    srs = _arr(trial_srs)
    n = n_trials if n_trials is not None else len(srs)
    if var_method == "trials":
        var = float(np.var(srs, ddof=1)) if len(srs) > 1 else 0.0
    else:
        # Var of a Sharpe estimate when the true SR is 0 (the skew and
        # kurtosis terms vanish at SR = 0): 1 / (T - 1).
        var = 1.0 / (len(_arr(x)) - 1)
    sr0 = expected_max_sr(n, var)
    return {"dsr": psr(x, sr0), "sr0_daily": sr0, "sr0_annual": sr0 * np.sqrt(ANN),
            "n_trials": float(n), "var_sr": var}


def min_track_record(x, sr_star: float = 0.0, conf: float = 0.95) -> float:
    """Days of data needed for PSR(sr_star) to reach ``conf`` (inf if SR<=sr_star)."""
    a = _arr(x)
    sr = sharpe(a, annualise=False)
    if sr <= sr_star:
        return float("inf")
    g3, g4 = stats.skew(a), stats.kurtosis(a, fisher=False)
    z = stats.norm.ppf(conf)
    return float(1 + (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) * (z / (sr - sr_star)) ** 2)


def summary(x: pd.Series) -> dict:
    a = _arr(x)
    eq = np.cumprod(1 + a)
    dd = 1 - eq / np.maximum.accumulate(eq)
    return {"ann_return": float(a.mean() * ANN), "ann_vol": float(a.std(ddof=1) * np.sqrt(ANN)),
            "sharpe": sharpe(a), "max_drawdown": float(dd.max()) if len(dd) else 0.0,
            "days": int(len(a))}
