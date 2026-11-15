"""The validation battery: run every test and give a verdict.

Inputs are a market, the *family* of rules that were tried (the honest
record of the search), the candidate to judge (a rule name, or "best" for
whichever rule looks best in the design sample), and a benchmark.

The comparison is always on a like-for-like basis: each rule's net excess
return and the benchmark's are scaled to the same volatility (10% a year)
before subtracting, so the *active return* d_t measures a Sharpe-ratio
edge, not extra leverage.

A candidate SURVIVES only if it passes every gate. Each gate maps to one
way a backtest fools people:

  costs       the edge is eaten by trading costs or market impact
  timing      a random alignment of the same positions does as well
  selection   the edge is what you expect from the best of N tries
  overfit     the in-sample winner does not stay good out of sample
  stability   the edge lives in one era or flips sign over time
  delay       the edge needs perfect same-close execution
  holdout     the edge disappears on data the search never saw
  conformal   after the usual in-sample to out-of-sample decay, the
              forecast edge is not above zero (the calibrated bound is
              reported alongside)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import conformal, engine, metrics, regimes
from .data import Market, recessions
from .multiple_testing import ADJUST, SnoopingBootstrap
from .pbo import pbo
from .permutation import circular_shift_test, random_trader_test
from .strategies import Rule, buy_and_hold, positions_matrix

TARGET_VOL = 0.10 / np.sqrt(252)


@dataclass
class Config:
    alpha: float = 0.05
    costs: engine.Costs = field(default_factory=engine.Costs)
    harsh: engine.Costs = field(default_factory=lambda: engine.HARSH)
    reps: int = 1000
    seed: int = 0
    pbo_blocks: int = 16
    conformal_alpha: float = 0.2
    is_days: int = 1260
    oos_days: int = 756
    holdout_start: str | None = None
    capacity: bool = True
    # "stepspa": Hsu-Hsu-Kuan Step-SPA p <= alpha (exact bootstrap FWER control).
    # "strict": also require Deflated Sharpe >= 1 - alpha (v0.1 rule; ~half the power).
    selection: str = "stepspa"
    # "median": forecast edge after the typical haircut must be > 0.
    # "bound": the (1 - conformal_alpha) lower bound must be > 0 (v0.1 rule).
    conformal_gate: str = "median"


def active_returns(net: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """Vol-matched active returns: z_k - z_bench, each scaled to 10%/yr."""
    z = net / net.std(ddof=1).replace(0, np.nan) * TARGET_VOL
    zb = bench / bench.std(ddof=1) * TARGET_VOL
    return z.sub(zb, axis=0).fillna(0.0)


def _edge(net: pd.Series, bench: pd.Series) -> float:
    return metrics.sharpe(net) - metrics.sharpe(bench)


@dataclass
class Report:
    candidate: str
    market: str
    period: tuple
    gates: dict
    details: dict

    @property
    def survives(self) -> bool:
        return all(g["pass"] for g in self.gates.values() if g["pass"] is not None)

    def to_dict(self) -> dict:
        return {"candidate": self.candidate, "market": self.market,
                "period": list(self.period), "survives": self.survives,
                "gates": self.gates, "details": self.details}

    def text(self) -> str:
        lines = [f"Candidate: {self.candidate}   Market: {self.market}   "
                 f"{self.period[0]} to {self.period[1]}",
                 f"Verdict: {'SURVIVES' if self.survives else 'FAILS'}", ""]
        for name, g in self.gates.items():
            mark = "n/a " if g["pass"] is None else ("PASS" if g["pass"] else "FAIL")
            lines.append(f"  [{mark}] {name:<10} {g['why']}")
        return "\n".join(lines)


def validate(market: Market, family: list[Rule], candidate: str = "best",
             benchmark: Rule | None = None, cfg: Config = Config()) -> Report:
    bench_rule = benchmark or buy_and_hold()
    rules = {r.name: r for r in family}
    warm = max(r.lookback for r in family) + 2
    P = positions_matrix(family, market.close)
    idle = [c for c in P.columns if P[c].iloc[warm:].fillna(0).abs().sum() == 0]
    P = P.drop(columns=idle)  # rules that never take a position
    bpos = bench_rule.positions(market.close)

    # design sample (everything before the holdout, after warm-up)
    idx = market.index[warm:]
    design = idx if cfg.holdout_start is None else idx[idx < pd.Timestamp(cfg.holdout_start)]
    hold = None if cfg.holdout_start is None else idx[idx >= pd.Timestamp(cfg.holdout_start)]

    def nets(costs, lag=1, cols=None):
        Q = P if cols is None else P[cols]
        return engine.run(Q, market.excess, market.rf, costs, lag)

    net_all = nets(cfg.costs)
    bnet_all = engine.run(bpos, market.excess, market.rf, cfg.costs)
    net, bnet = net_all.loc[design], bnet_all.loc[design]
    A = active_returns(net, bnet)

    # choose the candidate on the design sample only
    srs_daily = metrics.sharpe_cols(A.values, annualise=False)
    cand = A.columns[int(np.argmax(srs_daily))] if candidate == "best" else candidate
    rule = rules[cand]
    k = A.columns.get_loc(cand)
    d = A[cand]
    gates, det = {}, {"idle_rules": idle}

    # 1. costs and capacity ------------------------------------------------
    gross = engine.run(P[cand], market.excess, market.rf, engine.ZERO).loc[design]
    bgross = engine.run(bpos, market.excess, market.rf, engine.ZERO).loc[design]
    harsh = engine.run(P[cand], market.excess, market.rf, cfg.harsh).loc[design]
    bharsh = engine.run(bpos, market.excess, market.rf, cfg.harsh).loc[design]
    e_gross, e_net, e_harsh = _edge(gross, bgross), _edge(net[cand], bnet), _edge(harsh, bharsh)
    cap = engine.capacity(P[cand].loc[design], bpos.loc[design], market.excess.loc[design],
                          market.rf.loc[design], cfg.costs) if cfg.capacity else {"capacity_aum": None}
    det["costs"] = {"edge_gross": e_gross, "edge_net": e_net, "edge_harsh": e_harsh,
                    "turnover_per_year": engine.turnover(P[cand].loc[design]),
                    "capacity_aum": cap["capacity_aum"],
                    "candidate": metrics.summary(net[cand]), "benchmark": metrics.summary(bnet)}
    gates["costs"] = {"pass": bool(e_net > 0 and e_harsh > 0),
                      "why": f"Sharpe edge vs benchmark: gross {e_gross:+.2f}, net {e_net:+.2f}, "
                             f"harsh costs {e_harsh:+.2f}; turnover "
                             f"{det['costs']['turnover_per_year']:.1f}x/yr"}

    # 2. timing (permutation) ---------------------------------------------
    x = market.excess.loc[design].values
    pos_d = P[cand].loc[design].values
    cs = circular_shift_test(pos_d, x, rule.lookback)
    rt = random_trader_test(pos_d, x, n_traders=1000, seed=cfg.seed)
    det["timing"] = {"circular_shift": cs, "random_trader": rt}
    gates["timing"] = {"pass": bool(cs["p_value"] < cfg.alpha),
                       "why": f"circular-shift p = {cs['p_value']:.3f} over {cs['n_shifts']} shifts "
                              f"(Crash Replay random-trader test: beats {rt['beat_share']:.0%})"}

    # 3. selection (data snooping) ----------------------------------------
    sb = SnoopingBootstrap(A.values, reps=cfg.reps, seed=cfg.seed)
    single = sb.single_pvalues()
    adj = {name: float(f(single)[k]) for name, f in ADJUST.items()}
    n_eff = metrics.effective_trials(A.values) if A.shape[1] > 1 else 1.0
    dsr = metrics.deflated_sharpe(d.values, srs_daily, n_trials=n_eff)
    dsr_trials = metrics.deflated_sharpe(d.values, srs_daily, n_trials=n_eff, var_method="trials")
    spa = sb.spa()
    stepm = sb.stepm(cfg.alpha)
    stepspa = sb.stepm(cfg.alpha, recenter=True)
    det["selection"] = {"n_rules": int(A.shape[1]), "n_effective": n_eff, "block": sb.block,
                        "p_single": float(single[k]), "p_adjusted": adj,
                        "dsr": dsr, "dsr_trial_variance": dsr_trials,
                        "reality_check_p": sb.reality_check(), "spa": spa,
                        "stepm_p": float(stepm["p_adjusted"][k]),
                        "stepspa_p": float(stepspa["p_adjusted"][k]),
                        "stepm_survivors": [A.columns[i] for i in stepm["rejected"]],
                        "edge_sharpe": metrics.sharpe(d)}
    if A.shape[1] == 1:
        ok = single[k] < cfg.alpha
        why = f"single rule, bootstrap p = {single[k]:.3f}"
    else:
        p_sel = float(stepspa["p_adjusted"][k])
        ok = p_sel <= cfg.alpha if cfg.selection == "stepspa" else (
            dsr["dsr"] >= 1 - cfg.alpha and stepm["p_adjusted"][k] <= cfg.alpha)
        why = (f"Step-SPA p = {p_sel:.3f} across {A.shape[1]} rules (Romano-Wolf "
               f"{stepm['p_adjusted'][k]:.3f}, SPA {spa['consistent']:.3f}, Holm {adj['holm']:.3f}); "
               f"DSR = {dsr['dsr']:.2f} vs hurdle {dsr['sr0_annual']:.2f} from {n_eff:.0f} effective rules")
    gates["selection"] = {"pass": bool(ok), "why": why}

    # 4. overfitting (PBO) -------------------------------------------------
    if A.shape[1] >= 4:
        pb = pbo(A.values, cfg.pbo_blocks, max_splits=4000, seed=cfg.seed)
        det["pbo"] = {k2: v for k2, v in pb.items() if k2 not in ("logits", "is_sharpe", "oos_sharpe")}
        det["pbo"]["_scatter"] = (pb["is_sharpe"].tolist(), pb["oos_sharpe"].tolist())
        gates["overfit"] = {"pass": bool(pb["pbo"] < 0.5),
                            "why": f"PBO = {pb['pbo']:.2f} over {pb['splits']} CSCV splits; "
                                   f"OOS/IS slope {pb['degradation_slope']:+.2f}"}
    else:
        gates["overfit"] = {"pass": None, "why": "needs 4+ rules"}

    # 5. stability (regimes and drift) ------------------------------------
    rec = recessions(d.index)
    tables = {
        "decade": regimes.by_regime(d, regimes.decade_labels(d.index)),
        "half": regimes.by_regime(d, regimes.half_labels(d.index)),
        "trend": regimes.by_regime(d, regimes.trend_labels(market.close).reindex(d.index)),
        "volatility": regimes.by_regime(d, regimes.vol_labels(market.excess).reindex(d.index)),
        "recession": regimes.by_regime(d, pd.Series(np.where(rec > 0, "recession", "expansion"),
                                                     index=d.index)),
    }
    roll = regimes.rolling_edge(d)
    cus = regimes.cusum_break(d)
    det["stability"] = {name: t.to_dict("records") for name, t in tables.items()}
    det["stability"].update({"share_3y_blocks_positive": roll["share_blocks_positive"],
                             "cusum": cus, "_rolling": roll["series"]})
    halves_ok = bool((tables["half"]["edge_sharpe"] > 0).all())
    blocks_ok = roll["share_blocks_positive"] >= 0.6
    gates["stability"] = {"pass": bool(halves_ok and blocks_ok and not cus["break"]),
                          "why": f"edge > 0 in {roll['share_blocks_positive']:.0%} of 3-year blocks; "
                                 f"halves {', '.join(f'{v:+.2f}' for v in tables['half']['edge_sharpe'])}; "
                                 f"CUSUM p = {cus['p_value']:.3f}"
                                 + (f" (break near {cus['break_date']})" if cus["break"] else "")}

    # 6. execution delay ---------------------------------------------------
    late = engine.run(P[cand], market.excess, market.rf, cfg.costs, lag=2).loc[design]
    blate = engine.run(bpos, market.excess, market.rf, cfg.costs, lag=2).loc[design]
    e_late = _edge(late, blate)
    det["delay"] = {"edge_lag1": e_net, "edge_lag2": e_late}
    gates["delay"] = {"pass": bool(e_late > 0),
                      "why": f"edge {e_net:+.2f} trading next close, {e_late:+.2f} one day later"}

    # 7. holdout -----------------------------------------------------------
    if hold is not None and len(hold) > 252:
        hn, hb = net_all[cand].loc[hold], bnet_all.loc[hold]
        hA = active_returns(net_all[[cand]].loc[hold], hb)[cand]
        from .bootstrap import mean_pvalues
        hp = float(mean_pvalues(hA.values, reps=cfg.reps, seed=cfg.seed)[0])
        e_h = _edge(hn, hb)
        det["holdout"] = {"start": cfg.holdout_start, "edge": e_h, "p_value": hp,
                          "candidate": metrics.summary(hn), "benchmark": metrics.summary(hb)}
        gates["holdout"] = {"pass": bool(e_h > 0),
                            "why": f"edge {e_h:+.2f} after {cfg.holdout_start} "
                                   f"(bootstrap p = {hp:.3f}; the gate asks only for the right sign)"}
    else:
        gates["holdout"] = {"pass": None, "why": "no holdout set"}

    # 8. conformal bound on future edge ------------------------------------
    fixed = None if candidate == "best" else cand
    wf = conformal.walk_forward(A, cfg.is_days, cfg.oos_days, fixed=fixed)
    if len(wf) >= 12 + int(np.ceil(cfg.oos_days / 63)) + 8:
        b = conformal.aci_bounds(wf, cfg.conformal_alpha, oos_days=cfg.oos_days)
        cs_ = conformal.summary(b, cfg.conformal_alpha)
        now = conformal.current_bound(A, wf, b, cfg.conformal_alpha, is_days=cfg.is_days, fixed=fixed)
        det["conformal"] = {**cs_, "now": now, "_bounds": b}
        gates["conformal"] = {
            "pass": bool(now["median_forecast"] > 0 if cfg.conformal_gate == "median"
                         else now["lower_bound"] > 0),
            "why": f"forecast next-3-year Sharpe edge {now['median_forecast']:+.2f} after the typical "
                   f"in-to-out-of-sample haircut; {1 - cfg.conformal_alpha:.0%} lower bound "
                   f"{now['lower_bound']:+.2f} (realised miss rate {cs_['aci_miss_rate']:.0%} vs "
                   f"{cfg.conformal_alpha:.0%} target over {cs_['n']} origins)"}
    else:
        gates["conformal"] = {"pass": None, "why": "sample too short for walk-forward"}

    period = (str(design[0].date()), str(design[-1].date()))
    return Report(cand, market.name, period, gates, det)
