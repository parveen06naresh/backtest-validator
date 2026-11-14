"""Run the three case studies on the cached S&P 500 / Nasdaq / Dow data.

1. Crash Replay's trend rule (200-day trend filter x 15% vol target):
   what the game shows inside five crashes, then the full battery on
   1950-2026 with a post-publication holdout (2007+), and replications on
   the Nasdaq and the Dow.
2. Data mining: the best of ~900 technical rules on 1950-1999, with
   2000-2026 held out.
3. Turn-of-the-month: a published calendar effect, searched over 13
   variants on 1950-1989 with 1990-2026 held out.

Writes results/case_studies.json and the figures in paper/figures/.
Usage: python scripts/case_studies.py [--reps 1000]
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from backtest_validator import Config, engine, load, metrics, validate  # noqa: E402  # noqa: E402
from backtest_validator import strategies as S  # noqa: E402
from backtest_validator.permutation import random_trader_test  # noqa: E402

CRASHES = [  # Crash Replay's five scenarios
    ("2008 Financial Crisis", "spx", "2007-06-01", "2009-12-31"),
    ("Black Monday 1987", "spx", "1987-06-01", "1988-03-31"),
    ("Dot-com bust", "ndq", "1999-10-01", "2002-12-31"),
    ("COVID crash", "spx", "2019-12-02", "2020-12-31"),
    ("2022 rate shock", "ndq", "2021-09-01", "2023-06-30"),
]
CR = "trend200_vt15_20"


def clean(o):
    """Make a details dict JSON-safe and drop large private arrays."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def crash_windows():
    rule = S.trend_voltarget()
    rows = []
    for name, ix, a, b in CRASHES:
        m = load(ix, start="1950-01-03" if ix == "spx" else "1972-01-03")
        pos = rule.positions(m.close)
        net = engine.run(pos, m.excess, m.rf).loc[a:b]
        bh = engine.run(S.buy_and_hold().positions(m.close), m.excess, m.rf).loc[a:b]
        tot = lambda r, rf: float(np.prod(1 + r + rf) - 1)  # noqa: E731
        rf = m.rf.loc[a:b]
        rt = random_trader_test(pos.loc[a:b].values, m.excess.loc[a:b].values,
                                n_traders=2000, seed=1)
        rows.append({"crash": name, "index": ix, "from": a, "to": b,
                     "rule_return": tot(net, rf), "buy_hold_return": tot(bh, rf),
                     "rule_maxdd": metrics.summary(net)["max_drawdown"],
                     "buy_hold_maxdd": metrics.summary(bh)["max_drawdown"],
                     "beats_random_traders": rt["beat_share"]})
    return rows


def run_case(market, family, candidate, cfg):
    r = validate(market, family, candidate, cfg=cfg)
    print(r.text(), "\n", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    spx = load("spx")
    out, keep = {}, {}

    out["crash_windows"] = crash_windows()
    fam = S.standard_family()
    r = run_case(spx, fam, CR, Config(reps=args.reps, holdout_start="2007-01-01"))
    out["crash_replay_spx"], keep["cr"] = r.to_dict(), r
    r_full = run_case(spx, fam, CR, Config(reps=args.reps, capacity=False))
    out["crash_replay_spx_full"], keep["cr_full"] = r_full.to_dict(), r_full
    for ix, start in (("ndq", "1972-01-03"), ("dji", "1992-01-02")):
        r2 = run_case(load(ix, start=start), fam, CR, Config(reps=args.reps, capacity=False))
        out[f"crash_replay_{ix}"] = r2.to_dict()

    r = run_case(spx, S.dense_family(), "best", Config(reps=args.reps, holdout_start="2000-01-01"))
    out["data_mined"], keep["dm"] = r.to_dict(), r

    cal = [S.turn_of_month(b, a) for b in (0, 1, 2) for a in (1, 2, 3, 4)] + [S.halloween()]
    r = run_case(spx, cal, "best", Config(reps=args.reps, holdout_start="1990-01-01"))
    out["turn_of_month"], keep["tom"] = r.to_dict(), r

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "case_studies.json").write_text(json.dumps(clean(out), indent=1))
    figures(spx, keep)


# --- figures -----------------------------------------------------------------
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#c9c8c2")
    ax.spines["bottom"].set_color("#c9c8c2")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color="#ecebe7", lw=0.6)
    ax.set_axisbelow(True)


def figures(spx, keep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    fd = ROOT / "paper" / "figures"
    fd.mkdir(parents=True, exist_ok=True)

    # 1. Crash Replay rule vs buy-and-hold, growth of $1 (log), full sample
    pos = S.trend_voltarget().positions(spx.close)
    net = engine.run(pos, spx.excess, spx.rf).iloc[202:] + spx.rf.iloc[202:]
    bh = engine.run(S.buy_and_hold().positions(spx.close), spx.excess, spx.rf).iloc[202:] + spx.rf.iloc[202:]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    ax.plot((1 + bh).cumprod(), color=GRAY, lw=1.2, label="Buy and hold")
    ax.plot((1 + net).cumprod(), color=BLUE, lw=1.2, label="Crash Replay rule")
    ax.set_yscale("log")
    ax.set_title("Growth of $1, S&P 500 total return", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5)
    ax.axvline(pd.Timestamp("2007-01-01"), color=MUTED, lw=0.8, ls="--")
    ax.text(pd.Timestamp("2007-06-01"), 1.3, "holdout", color=MUTED, fontsize=7)
    _style(ax)
    ax = axes[1]
    dec = pd.DataFrame(keep["cr_full"].details["stability"]["decade"])
    ax.bar(dec["regime"], dec["edge_sharpe"], color=[BLUE if v > 0 else ORANGE for v in dec["edge_sharpe"]],
           width=0.6)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_title("Sharpe edge over buy and hold, by decade", fontsize=9, loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fd / "crash_replay.png", dpi=220)
    plt.close(fig)

    # 2. Conformal bounds over time for the Crash Replay rule
    b = keep["cr_full"].details["conformal"]["_bounds"].dropna(subset=["bound"])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.plot(b["origin"], b["oos_sharpe"], color=GRAY, lw=1.0, label="Realised next-3-year edge")
    ax.plot(b["origin"], b["bound"], color=BLUE, lw=1.2, label="80% conformal lower bound (ACI)")
    miss = b[b["oos_sharpe"] < b["bound"]]
    ax.scatter(miss["origin"], miss["oos_sharpe"], s=9, color=ORANGE, zorder=3, label="Miss")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Crash Replay rule: bound on future Sharpe edge vs what happened", fontsize=9,
                 loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fd / "conformal.png", dpi=220)
    plt.close(fig)

    # 3. Data mining: CSCV scatter and holdout equity of the winner
    dm = keep["dm"]
    is_s, oos_s = dm.details["pbo"]["_scatter"]
    cand = dm.candidate
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5))
    ax = axes[0]
    ax.scatter(is_s, oos_s, s=3, color=BLUE, alpha=0.35, linewidths=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("In-sample Sharpe edge of the winner", fontsize=8, color=MUTED)
    ax.set_ylabel("Out-of-sample edge", fontsize=8, color=MUTED)
    ax.set_title("CSCV splits, 1950-1999 (best of 912)", fontsize=9, loc="left", color=INK)
    _style(ax)
    ax = axes[1]
    rule = {r.name: r for r in S.dense_family()}[cand]
    p = rule.positions(spx.close)
    bx = engine.run(S.buy_and_hold().positions(spx.close), spx.excess, spx.rf).loc["1952":]
    for lag, col, lab in ((1, BLUE, "trade next close"), (2, ORANGE, "trade one day later")):
        nr = engine.run(p, spx.excess, spx.rf, lag=lag).loc["1952":]
        ax.plot((1 + nr).cumprod() / (1 + bx).cumprod(), color=col, lw=1.1, label=lab)
    ax.axvline(pd.Timestamp("2000-01-01"), color=MUTED, lw=0.8, ls="--")
    ax.set_yscale("log")
    ax.set_title(f"{cand} relative to buy and hold", fontsize=9, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fd / "data_mined.png", dpi=220)
    plt.close(fig)

    # 4. Turn of the month: rolling edge with publication date
    tom = keep["tom"]
    rule = S.turn_of_month(*[int(x) for x in tom.candidate[3:].split("+")])
    p = rule.positions(spx.close)
    nr = engine.run(p, spx.excess, spx.rf)
    bnet = engine.run(S.buy_and_hold().positions(spx.close), spx.excess, spx.rf)
    from backtest_validator.pipeline import active_returns
    a = active_returns(nr.to_frame("x").iloc[30:], bnet.iloc[30:])["x"]
    roll = (a.rolling(1260).mean() / a.rolling(1260).std() * np.sqrt(252)).dropna()
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    ax.plot(roll, color=BLUE, lw=1.1)
    ax.axhline(0, color=MUTED, lw=0.8)
    for d, t in (("1988-01-01", "published (Lakonishok-Smidt 1988)"), ("1990-01-01", "")):
        ax.axvline(pd.Timestamp(d), color=MUTED, lw=0.8, ls="--")
    ax.text(pd.Timestamp("1990-09-01"), roll.min() * 0.95, "published 1987-88; holdout from 1990",
            fontsize=7, color=MUTED)
    ax.set_title(f"Turn of the month ({tom.candidate}): rolling 5-year Sharpe edge", fontsize=9,
                 loc="left", color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(fd / "turn_of_month.png", dpi=220)
    plt.close(fig)
    print("figures written to", fd)


if __name__ == "__main__":
    main()
