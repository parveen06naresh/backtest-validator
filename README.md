# Backtest Validator

An open-source Python framework that tries to kill false positives in trading-signal research.

You give it a trading idea and the record of everything you tried while looking for it. It runs eight
gates, each aimed at one common way a backtest fools people, and reports which ones the idea passed and
which one stopped it.

**Paper (7 pages):** [paper/backtest-validator.pdf](paper/backtest-validator.pdf)

> Research and education only. Nothing here is investment advice, and the tool never says an idea will
> make money. It can only say when the evidence does not support one.

## Results in one table

On synthetic markets where no rule has an edge, picking the best of 99 textbook rules gives a false
positive most of the time if you skip the corrections. The battery keeps that rate at or below 5%:

| Method (400 null markets, best of 99 rules) | False-positive rate |
|---|---|
| Plain t-test on the best rule | **31.8%** |
| Holm / Benjamini-Hochberg | 0.5% / 1.2% |
| White Reality Check / Hansen SPA | 3.8% / 2.2% |
| Romano-Wolf StepM | 1.8% |
| Deflated Sharpe Ratio ≥ 0.95 | 0.2% |
| Step-SPA (the selection gate) | 2.2% |
| **Full verdict (all eight gates)** | **0.8%** (3 of 400; 95% upper bound 2.2%) |

It also has to find real edges. On markets with a planted edge, v0.2 finds it far more often than the
stricter first version while staying under 5% false positives:

| Planted Sharpe edge | 20 years of data, v0.2 | v0.1 | 40 years, v0.2 |
|---|---|---|---|
| 0.5 | 20% | 5% | 40% |
| 0.8 | **60%** | 28% | **68%** |
| 1.1 | **74%** | 59% | **83%** |
| 1.4 | 82% | 71% | |

Then four real ideas on public index data from 1950 to 2026. **None survives:**

| Idea | What it looked like | What killed it |
|---|---|---|
| Crash Replay's trend rule (200-day trend × 15% vol target) | Lost far less in 2008, 1987, 2000 and 2022; Sharpe edge +0.16 after costs | Selection: not distinguishable from the best of 98 trend rules (Step-SPA p = 0.29). Edge fell from +0.89 in the 1950s to −0.15 in the 2010s |
| Best of 912 data-mined rules, 1951–1999 | Sharpe edge +0.51; **passed Holm, Romano-Wolf, SPA and Step-SPA** | One-day execution delay turns it to −0.32; break near 1980; −0.49 on 2000–2026 |
| Turn of the month, 1950–1989 | Passed **every** statistical gate | Reversed after publication: −0.19 on 1990–2026 |
| Same trend rule on the Dow, 1993–2026 | | Negative edge, fails 6 of 8 gates |

The data-mined rule is the main lesson. Multiple-testing corrections said it was real, and in-sample it
probably was: before the 1980s the S&P 500 index was autocorrelated because many stocks traded rarely.
But that pattern was not tradable and it is gone. Corrections alone are not enough.

The framework also found a flaw in my earlier project, [Crash Replay](https://github.com/parveen06naresh/Crash-Replay).
Its "random trader" skill test rewards time in the market as if it were skill. With a 6% market drift it
calls a no-skill trend rule significant 14 to 19% of the time instead of 5%. The circular-shift test
used here fixes that (3.5 to 7.8%).

## The eight gates

| Gate | The question | Method |
|---|---|---|
| costs | Does the edge survive trading costs and market impact? | spread + fee + square-root impact; harsh-cost stress; capacity in $ |
| timing | Would the same positions at a random alignment do as well? | circular-shift permutation (keeps autocorrelation, exact via FFT, no look-ahead) |
| selection | Is the edge more than the best of N tries? | Step-SPA (Hsu-Hsu-Kuan: Romano-Wolf stepdown with Hansen recentering); also reports Deflated Sharpe, White RC, Hansen SPA, Holm, BH, BY |
| overfit | Does the in-sample winner stay good out of sample? | Probability of Backtest Overfitting (CSCV, 16 blocks) |
| stability | Does the edge live in one era? | both halves, 3-year blocks, CUSUM break test (Newey-West); tables by decade, trend, volatility, recession |
| delay | Does it need perfect same-close execution? | rerun trading one day later |
| holdout | Does it hold on data the search never saw? | candidate chosen without the holdout |
| conformal | After the usual in-sample to out-of-sample decay, is the forecast edge still above zero? | walk-forward haircuts + Adaptive Conformal Inference (the method from [Pacer](https://github.com/parveen06naresh/pacer-serve)); the 80% bound is reported, with a 21.2% realized miss rate vs 20% target on null markets |

All comparisons are like for like: the rule and buy-and-hold are both scaled to 10% volatility before
subtracting, so a rule gets no credit for simply holding less risk. Shared stationary-bootstrap draws
(Politis-Romano, automatic block length) keep the dependence across rules and across time.

## Install and run

```bash
git clone https://github.com/parveen06naresh/backtest-validator
cd backtest-validator
pip install -e ".[dev]"

btv validate --market spx --candidate trend200_vt15_20
btv validate --market spx --family dense --candidate best --holdout 2000-01-01
btv rules --family standard
btv null --reps 20          # quick false-positive check on synthetic markets
```

From Python:

```python
import backtest_validator as bv
from backtest_validator import strategies as S

market = bv.load("spx")                                  # cached data, works offline
family = S.standard_family()                             # everything you tried
report = bv.validate(market, family, "best",
                     cfg=bv.Config(holdout_start="2007-01-01"))
print(report.text())
report.to_dict()                                         # full details
```

Bring your own rule with `bv.Rule(name, family, params, fn, lookback)` where `fn` maps a price series to
positions in [-1, 1] using only data up to each close. The engine adds the one-day lag.

## Reproduce everything

```bash
python scripts/null_harness.py       # ~20 min on 4 cores -> results/null_harness.json
python scripts/case_studies.py       # ~10 min -> results/case_studies.json, paper/figures
python scripts/build_paper.py        # -> paper/backtest-validator.pdf (needs Playwright for the PDF)
pytest -q                            # 29 tests
```

Every number in the paper is read from the results files. The data snapshot in `data/raw` (S&P 500,
Nasdaq, Dow and VIX from Yahoo Finance; T-bills, 10-year yield and recession dates from FRED; S&P 500
dividends from Robert Shiller) ends on 30 Oct 2026. `scripts/fetch_data.py` refreshes it.

## Layout

```
backtest_validator/
  data.py              load the cached snapshot (total return, T-bill cash)
  strategies.py        rule families: standard (99), dense (912), calendar, vol-target
  engine.py            lagged execution, cost model, capacity
  metrics.py           Sharpe, PSR, Deflated Sharpe, minimum track record
  bootstrap.py         stationary bootstrap, Politis-White block length
  multiple_testing.py  Bonferroni, Holm, BH, BY, Reality Check, SPA, Romano-Wolf
  pbo.py               Probability of Backtest Overfitting (CSCV)
  permutation.py       circular-shift test, Crash Replay random-trader test
  regimes.py           regime tables, rolling edge, CUSUM
  conformal.py         walk-forward haircuts, split conformal, ACI
  nulls.py             GARCH-t synthetic markets with optional planted signal
  pipeline.py          the eight gates and the verdict
  cli.py               the `btv` command
scripts/               null harness, case studies, paper builder, data refresh
```

## Limitations

- The search record is self-reported. Rules you tried and forgot are not counted.
- Small edges are hard to confirm: on 20 years of daily data a planted 0.5-Sharpe edge is found 20% of
  the time (40% with 40 years). No valid test can do much better on one index. `Config(selection="strict",
  conformal_gate="bound")` restores the stricter v0.1 rules.
- US index data only, mostly long/flat timing rules. Dividends are monthly approximations for the
  S&P 500 and missing for the Nasdaq and Dow. Costs assume today's ETF liquidity.
- Null models are models. Conformal coverage is checked empirically, not proved, for overlapping windows.
- Volatility scaling uses the full sample (does not change Sharpe ratios, slightly affects test scale).

## References

Bailey & López de Prado (2014) Deflated Sharpe Ratio · Bailey, Borwein, López de Prado & Zhu (2017)
Probability of Backtest Overfitting · White (2000) Reality Check · Hansen (2005) SPA · Romano & Wolf
(2005) StepM · Politis & Romano (1994) stationary bootstrap · Politis & White (2004) block length ·
Holm (1979) · Benjamini & Hochberg (1995) · Gibbs & Candès (2021) Adaptive Conformal Inference ·
Almgren et al. (2005), Tóth et al. (2011) square-root impact · Sullivan, Timmermann & White (1999) ·
Harvey, Liu & Zhu (2016) · McLean & Pontiff (2016). Full list in the paper.

## Author

Parveen Kumar Naresh Kumar · Applied Mathematics and Computer Science, University of South Florida ·
[LinkedIn](https://www.linkedin.com/in/parveenkumarnareshkumar/)

Built with AI-assisted development (Claude Code). MIT license.
