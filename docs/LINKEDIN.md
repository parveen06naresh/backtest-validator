# Posting kit

## Resume entry

**Backtest Validator** · Python, NumPy, SciPy · github.com/parveen06naresh/backtest-validator
- Built an open-source framework that runs 8 statistical gates on a trading idea (Deflated Sharpe, PBO,
  White Reality Check, Hansen SPA, Romano-Wolf, block bootstrap, cost and capacity model, conformal bounds).
- Cut the false-positive rate on the best of 99 rules from 31.8% to 0.8% (400 null markets) while finding a planted
  0.8-Sharpe edge 60% of the time (2x the stricter first version); 29 tests, CI.
- Showed that a 912-rule data-mining winner passed Holm, Romano-Wolf and SPA but failed under a 1-day execution delay
  and after 1999. Wrote a 7-page paper.

## LinkedIn post (draft)

I tested 912 trading rules on nearly 50 years of S&P 500 data. The best one passed three standard
multiple-testing corrections. It still doesn't work.

The rule: buy when the index closes above its 10-day average. From 1951 to 1999 it beat buy-and-hold
by +0.51 Sharpe after costs. Holm, Romano-Wolf and Hansen's SPA test all said it was real.

Then I ran two more checks:
• Trade one day later instead of at the next close: the edge flips to −0.32.
• Run it on 2000–2026, data the search never saw: −0.49.

The pattern was real but couldn't be traded. Before the 1980s, many stocks in the index traded rarely,
so yesterday's index move predicted today's. You couldn't buy the index at those prices, and the
pattern is gone now.

So I built Backtest Validator, an open-source Python tool that runs a trading idea through 8 gates:
costs and market impact, a permutation test, a correction for how many rules you tried, overfitting
(PBO), stability across decades, execution delay, a holdout, and a conformal bound on future performance.

Two results I didn't expect:
1. My own earlier project failed. The trend rule from my Crash Replay game looks great in crashes, but
   its edge shrank every decade, from +0.89 in the 1950s to −0.15 in the 2010s.
2. The game's "skill test" had a flaw. It rewards time in the market as if it were skill, so with a
   rising market it flags no-skill rules 14–19% of the time instead of 5%. The new test fixes that.

On 400 simulated markets with no edge, picking the best of 99 rules gives a false positive 32% of the
time. With all 8 gates: 0.8%. And when I plant a real edge, it finds it 60% of the time, so it's not
a tool that just says no.

Code, data and a 7-page paper: github.com/parveen06naresh/backtest-validator

Research and education only, not investment advice.

## Notes for posting

- Attach page 1 of the paper or Figure 1 (paper/figures/harness.png) as the image.
- Put the GitHub link in the first comment if you want more reach; LinkedIn tends to show posts with
  outside links to fewer people.
- Interview angle: "How do you know your backtest isn't luck?" Walk through the data-mined case: the
  corrections passed, but the delay and holdout gates caught it.
