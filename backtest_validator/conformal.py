"""Conformal bounds on out-of-sample performance (the Pacer idea).

A backtest Sharpe is an in-sample (IS) number. What we want is a bound on
the next out-of-sample (OOS) Sharpe that has a stated coverage rate.

Walk forward through history. At each origin, use the trailing ``is_days``
to choose a rule (the best of the family, or a fixed rule), then measure
the chosen rule on the next ``oos_days``. The *haircut* IS - OOS is the
non-conformity score. A split-conformal bound for a new origin is

    OOS lower bound = IS Sharpe - Q_{1 - alpha}(past haircuts)

which covers with probability >= 1 - alpha when scores are exchangeable.
Markets drift, so exchangeability is doubtful. As in Pacer we therefore
run Adaptive Conformal Inference (Gibbs and Candes 2021): after each
outcome, alpha_t moves by gamma * (alpha - miss_t), which keeps the
long-run miss rate near alpha even under drift. We always report the
realised miss rate so the claim can be checked rather than assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import sharpe_cols


def walk_forward(A: pd.DataFrame, is_days: int = 1260, oos_days: int = 756,
                 step: int = 63, fixed: str | None = None) -> pd.DataFrame:
    """IS and OOS annualised Sharpe of the rule chosen at each origin.

    ``A`` holds one active-return column per rule. With ``fixed`` the same
    rule is used at every origin (no selection); otherwise the IS best.
    """
    X = A.values
    rows = []
    for t in range(is_days, len(A) - oos_days + 1, step):
        ins = X[t - is_days: t]
        oos = X[t: t + oos_days]
        s_is = sharpe_cols(ins)
        k = A.columns.get_loc(fixed) if fixed else int(np.argmax(s_is))
        s_oos = sharpe_cols(oos[:, [k]])[0]
        rows.append({"origin": A.index[t], "rule": A.columns[k],
                     "is_sharpe": float(s_is[k]), "oos_sharpe": float(s_oos)})
    df = pd.DataFrame(rows)
    df["haircut"] = df["is_sharpe"] - df["oos_sharpe"]
    return df


def _q(scores, level):
    n = len(scores)
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return np.inf
    return float(np.sort(scores)[max(k, 1) - 1])


def aci_bounds(wf: pd.DataFrame, alpha: float = 0.2, gamma: float = 0.02,
               oos_days: int = 756, step: int = 63, min_cal: int = 8) -> pd.DataFrame:
    """Online conformal lower bounds with ACI and delayed feedback.

    The outcome of origin i is only known ``oos_days`` later, so its score
    joins the calibration set (and updates alpha) ``h = oos_days / step``
    origins later. No future information is ever used.
    """
    h = int(np.ceil(oos_days / step))
    a_t = alpha
    out = []
    for i in range(len(wf)):
        # outcomes that have finished by origin i
        if i - h >= 0 and i - h < len(out) and out[i - h]["bound"] is not None:
            done = out[i - h]
            a_t = float(np.clip(a_t + gamma * (alpha - done["miss"]), 1e-3, 0.999))
        cal = wf["haircut"].values[: max(i - h + 1, 0)]
        row = {"origin": wf["origin"].iloc[i], "is_sharpe": wf["is_sharpe"].iloc[i],
               "oos_sharpe": wf["oos_sharpe"].iloc[i], "alpha_t": a_t, "bound": None,
               "split_bound": None, "miss": None}
        if len(cal) >= min_cal:
            q = _q(cal, 1 - a_t)
            qs = _q(cal, 1 - alpha)
            row["bound"] = float(wf["is_sharpe"].iloc[i] - q)
            row["split_bound"] = float(wf["is_sharpe"].iloc[i] - qs)
            row["miss"] = float(wf["oos_sharpe"].iloc[i] < row["bound"])
        out.append(row)
    return pd.DataFrame(out)


def summary(b: pd.DataFrame, alpha: float) -> dict:
    ok = b.dropna(subset=["bound"])
    ok = ok[np.isfinite(ok["bound"])]
    if ok.empty:
        return {"n": 0}
    miss = ok["oos_sharpe"] < ok["bound"]
    smiss = ok["oos_sharpe"] < ok["split_bound"]
    return {"n": int(len(ok)), "target_miss": alpha,
            "aci_miss_rate": float(miss.mean()), "split_miss_rate": float(smiss.mean()),
            "share_bound_above_0": float((ok["bound"] > 0).mean()),
            "median_haircut": float((ok["is_sharpe"] - ok["oos_sharpe"]).median()),
            "last_origin": str(pd.Timestamp(ok["origin"].iloc[-1]).date()),
            "last_is_sharpe": float(ok["is_sharpe"].iloc[-1]),
            "last_bound": float(ok["bound"].iloc[-1])}


def current_bound(A: pd.DataFrame, wf: pd.DataFrame, b: pd.DataFrame, alpha: float,
                  gamma: float = 0.02, is_days: int = 1260, fixed: str | None = None) -> dict:
    """Bound for the period starting at the end of the data (today)."""
    ins = A.values[-is_days:]
    s = sharpe_cols(ins)
    k = A.columns.get_loc(fixed) if fixed else int(np.argmax(s))
    done = b.dropna(subset=["miss"])
    a_t = float(done["alpha_t"].iloc[-1]) if len(done) else alpha
    if len(done):
        a_t = float(np.clip(a_t + gamma * (alpha - done["miss"].iloc[-1]), 1e-3, 0.999))
    q = _q(wf["haircut"].values, 1 - a_t)
    return {"rule": str(A.columns[k]), "is_sharpe": float(s[k]), "alpha_t": a_t,
            "lower_bound": float(s[k] - q), "median_forecast": float(s[k] - np.median(wf["haircut"]))}
