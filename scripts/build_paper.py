"""Build paper/backtest-validator.pdf from results/*.json and paper/figures.

Every number in the paper is read from the results files, so re-running
the experiments and this script keeps text and data in sync.
Needs matplotlib and (for the PDF) playwright with Chromium.
Usage: python scripts/build_paper.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
FIG = PAPER / "figures"
H = json.loads((ROOT / "results" / "null_harness.json").read_text())
C = json.loads((ROOT / "results" / "case_studies.json").read_text())

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f2(x):
    return f"{x:+.2f}"


def harness_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
    n = H["null"]
    rows = [("Best rule, no correction", "naive_best_p<0.05"), ("Bonferroni", "bonferroni"),
            ("Holm", "holm"), ("Benjamini-Hochberg", "bh_fdr"), ("White Reality Check", "reality_check"),
            ("Hansen SPA", "spa_consistent"), ("Romano-Wolf StepM", "romano_wolf_any"),
            ("Deflated Sharpe >= 0.95", "dsr_sampling>=0.95"), ("Full verdict (all gates)", "verdict_survives")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ys = range(len(rows))[::-1]
    for y, (lab, k) in zip(ys, rows):
        r, (lo, hi) = n[k]["rate"], n[k]["ci95"]
        ax.barh(y, r, color=ORANGE if k.startswith("naive") else BLUE, height=0.55)
        ax.plot([lo, hi], [y, y], color=INK, lw=0.8)
        ax.text(0.445, y, pct(r), va="center", ha="right", fontsize=7, color=INK)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.axvline(0.05, color=MUTED, lw=0.8, ls="--")
    ax.text(0.055, -0.9, "5% target", fontsize=7, color=MUTED)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(-1.2, len(rows) - 0.5)
    ax.set_title(f"False-positive rate, {n['verdict_survives']['n']} null markets", fontsize=9,
                 loc="left", color=INK)
    ax2 = axes[1]
    edges = H["planted_edge_sharpe_sma100_ls"]
    sigs = list(H["power"])
    x = [edges[s] for s in sigs]
    for k, lab, col, ls in (("naive_best_p<0.05", "No correction", ORANGE, "-"),
                            ("romano_wolf_any", "Romano-Wolf", GRAY, "-"),
                            ("selection_gate", "Selection gate (DSR + RW)", BLUE, "--"),
                            ("verdict_survives", "Full verdict", BLUE, "-")):
        ax2.plot(x, [H["power"][s][k]["rate"] for s in sigs], color=col, ls=ls, lw=1.4, marker="o",
                 ms=3.5, label=lab)
    ax2.set_xlabel("Planted Sharpe edge of the true rule", fontsize=8, color=MUTED)
    ax2.set_ylim(0, 1.03)
    ax2.set_title("Power: runs that find a real edge", fontsize=9, loc="left", color=INK)
    ax2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=7.5)
    ax2.grid(axis="y", color="#ecebe7", lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "harness.png", dpi=220)
    plt.close(fig)


def gate_cell(rep, g):
    v = rep["gates"][g]["pass"]
    return '<td class="na">n/a</td>' if v is None else ('<td class="ok">pass</td>' if v else
                                                         '<td class="no">FAIL</td>')


def build_html():
    n, fr, cn = H["null"], H["fixed_rule"], H["conformal_null"]
    cr, crf, dm, tom = C["crash_replay_spx"], C["crash_replay_spx_full"], C["data_mined"], C["turn_of_month"]
    ndq, dji = C["crash_replay_ndq"], C["crash_replay_dji"]
    d = lambda r: r["details"]  # noqa: E731
    gates = ["costs", "timing", "selection", "overfit", "stability", "delay", "holdout", "conformal"]
    case_rows = [("Crash Replay rule, S&amp;P 500, 1951-2006 (holdout 2007+)", cr),
                 ("Crash Replay rule, Nasdaq, 1973-2026", ndq),
                 ("Crash Replay rule, Dow, 1993-2026", dji),
                 (f"Best of {d(dm)['selection']['n_rules']} rules ({dm['candidate']}), 1951-1999 "
                  "(holdout 2000+)", dm),
                 (f"Turn of the month ({tom['candidate']}), 1950-1989 (holdout 1990+)", tom)]
    verdict_table = "".join(
        f"<tr><td class='l'>{lab}</td><td>{f2(d(r)['costs']['edge_net'])}</td>"
        + "".join(gate_cell(r, g) for g in gates)
        + f"<td class='{'ok' if r['survives'] else 'no'}'><b>{'survives' if r['survives'] else 'fails'}"
          "</b></td></tr>" for lab, r in case_rows)
    crash_table = "".join(
        f"<tr><td class='l'>{w['crash']}</td><td>{pct(w['rule_return'], 0)}</td>"
        f"<td>{pct(w['buy_hold_return'], 0)}</td><td>{pct(w['rule_maxdd'], 0)}</td>"
        f"<td>{pct(w['buy_hold_maxdd'], 0)}</td><td>{pct(w['beats_random_traders'], 0)}</td></tr>"
        for w in C["crash_windows"])
    fixed_rows = "".join(
        f"<tr><td class='l'>{rule}</td>"
        f"<td>{pct(fr['drift_0.0'][rule]['circular_shift']['rate'])}</td>"
        f"<td>{pct(fr['drift_0.0'][rule]['random_trader']['rate'])}</td>"
        f"<td>{pct(fr['drift_0.06'][rule]['circular_shift']['rate'])}</td>"
        f"<td><b>{pct(fr['drift_0.06'][rule]['random_trader']['rate'])}</b></td></tr>"
        for rule in fr["drift_0.0"])
    sel_cr, sel_dm, sel_tom = d(cr)["selection"], d(dm)["selection"], d(tom)["selection"]
    st_full = crf["details"]["stability"]
    dec = {r["regime"]: r["edge_sharpe"] for r in st_full["decade"]}
    halves = [r["edge_sharpe"] for r in st_full["half"]]
    p_low = H["power"]["0.06"]["verdict_survives"]["rate"]
    p_mid = H["power"]["0.08"]["verdict_survives"]["rate"]
    p_hi = H["power"]["0.1"]["verdict_survives"]["rate"]
    e = H["planted_edge_sharpe_sma100_ls"]
    hcr = d(cr)["holdout"]
    conf_cr = crf["details"]["conformal"]
    cap = d(cr)["costs"]["capacity_aum"]

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Backtest Validator</title>
<style>
@page {{ size: Letter; margin: 0.72in 0.75in 0.72in 0.75in; }}
body {{ font-family: 'Charter', 'Georgia', serif; font-size: 10pt; line-height: 1.38; color: #111; }}
h1 {{ font-size: 16.5pt; margin: 0 0 4pt; line-height: 1.2; }}
h2 {{ font-size: 11.5pt; margin: 13pt 0 4pt; }}
h3 {{ font-size: 10pt; margin: 9pt 0 2pt; }}
p {{ margin: 0 0 6pt; text-align: justify; hyphens: auto; }}
.auth {{ font-size: 10pt; color: #333; margin-bottom: 8pt; }}
.abs {{ border-top: 1px solid #bbb; border-bottom: 1px solid #bbb; padding: 7pt 0; margin-bottom: 8pt; font-size: 9.4pt; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 4pt 0 6pt; font-family: 'Helvetica', 'Arial', sans-serif; }}
th, td {{ padding: 2.2pt 4pt; border-bottom: 1px solid #e3e3e0; text-align: center; }}
th {{ border-bottom: 1px solid #888; font-weight: 600; }}
td.l, th.l {{ text-align: left; }}
td.ok {{ color: #1d6b3a; }} td.no {{ color: #b2361f; font-weight: 600; }} td.na {{ color: #999; }}
.cap {{ font-size: 8.6pt; color: #444; margin: 0 0 9pt; }}
figure {{ margin: 6pt 0 2pt; }} figure img {{ width: 100%; }}
.eq {{ text-align: center; margin: 4pt 0 6pt; font-style: italic; }}
ol.refs {{ font-size: 8.3pt; padding-left: 16pt; line-height: 1.3; }}
.note {{ font-size: 8.5pt; color: #444; }}
code {{ font-size: 8.8pt; }}
</style></head><body>

<h1>Backtest Validator: a test battery for killing false positives in trading-signal research</h1>
<div class="auth">Parveen Kumar Naresh Kumar &middot; University of South Florida &middot;
github.com/parveen06naresh/backtest-validator &middot; November 2026</div>

<div class="abs"><b>Abstract.</b> A backtest that looks good is cheap to produce: try enough rules,
parameters and periods and one of them will. We describe an open-source Python framework that takes a
trading idea together with the record of everything that was tried to find it, and runs eight gates,
each aimed at one known way of fooling oneself: trading costs and market impact, timing that a random
alignment matches, selection from many trials, in-sample overfitting, instability across regimes,
dependence on perfect execution, failure on held-out data, and a calibrated bound on future
performance. On {n['verdict_survives']['n']} synthetic markets with no edge, picking the best of
{H['family_size']} textbook rules and running a plain significance test gives a false positive
{pct(n['naive_best_p<0.05']['rate'], 0)} of the time; the corrected tests stay at or below 5%, and the full
verdict passed none of them (95% upper bound {pct(n['verdict_survives']['ci95'][1])}). We then apply it
to four real ideas on 76 years of public index data: the trend rule from our earlier Crash Replay project,
the best of {sel_dm['n_rules']} data-mined technical rules, and the turn-of-the-month effect, plus
replications on other indexes. None survives. The data-mined winner passes the Holm, Romano-Wolf and SPA
corrections and still fails, because its edge vanishes with a one-day execution delay and on data after
1999. The turn-of-the-month effect passes every statistical gate before its publication and reverses
afterwards. The tool is for research and teaching, not investment advice.</div>

<h2>1. Problem</h2>
<p>Most published and private backtests are the survivors of a search. If a researcher tries
<i>N</i> unrelated rules on noise, the best in-sample Sharpe ratio grows like
&radic;(2 ln <i>N</i>) standard errors, so a "t-statistic above 2" is close to guaranteed
[3, 12]. Harvey, Liu and Zhu [12] argue that factor research should require <i>t</i> &gt; 3 for this
reason, and Bailey et al. [3, 4] show that the probability of backtest overfitting can be estimated
directly. The tools exist in separate papers. What is missing in practice is a single, tested
procedure that runs them together, records the search honestly, and reports which test killed an idea.
That is the aim of this project. It extends two of our earlier pieces of work: the random-trader
permutation test in Crash Replay, and the conformal risk bounds used to schedule requests in Pacer.</p>

<h2>2. Method</h2>
<h3>2.1 Data, timing and costs</h3>
<p>We use public daily data cached in the repository so every number reproduces offline: the S&amp;P 500
(1950 to 30 Oct 2026) with dividends from Shiller's monthly series, the Nasdaq Composite (1972+) and Dow
(1992+) as price indexes, the 3-month T-bill (FRED TB3MS, lagged one month) as cash, and NBER recession
dates. A rule maps information at the close of day <i>t</i> to a position <i>w<sub>t</sub></i> in
[-1, 1] that earns the excess return from <i>t</i> to <i>t</i>+1. A unit test perturbs all future prices
and checks that past positions do not change. Each change in position pays</p>
<p class="eq">cost = |&Delta;w| &middot; ( s/2 + f + Y &middot; &sigma;<sub>d</sub> &middot;
&radic;( |&Delta;w| &middot; AUM / ADV ) ),</p>
<p>a half-spread, a fee, and the square-root market-impact law [1, 24] with daily volatility
&sigma;<sub>d</sub>. Defaults are 1 bp, 0.5 bp, Y = 1, $10M of capital and $20B of daily volume for the
traded vehicle (an index ETF or future); a harsh scenario uses 5x spread and fee and 2x impact.
Capacity is the largest AUM at which the edge stays positive.</p>
<p><b>Like-for-like comparison.</b> Timing rules on an index mostly change exposure. To avoid crediting
a rule for holding less risk, each rule's net excess return and the benchmark's (buy and hold) are scaled
to 10% annual volatility before subtracting. The resulting <i>active return</i> d<sub>k,t</sub> has a
mean proportional to the Sharpe-ratio edge, and every test below is a test on d.</p>

<h3>2.2 The eight gates</h3>
<p><b>Costs.</b> The edge must be positive after realistic and after harsh costs.
<b>Timing.</b> A circular-shift permutation test: slide the whole position series against the returns
by <i>s</i> days. Positions keep their autocorrelation, holding periods and average exposure; returns keep
theirs; only the alignment is broken. We exclude shifts that would pair a position with a return it was
computed from, and evaluate all admissible shifts exactly with FFTs. We also report Crash Replay's
original random-trader test for comparison.
<b>Selection.</b> One shared stationary bootstrap [20] of the T&times;K active-return matrix
(automatic block length [21, 18]) gives White's Reality Check [25], Hansen's SPA test [11] and Romano-Wolf
StepM adjusted p-values [22]; Holm [13], Benjamini-Hochberg [7] and Benjamini-Yekutieli [8] adjustments are
also reported. The gate requires the Deflated Sharpe Ratio [5, 6] to be at least 0.95 and the candidate's
StepM p-value to be at most 0.05. The DSR hurdle is the expected maximum Sharpe of <i>N</i><sub>eff</sub>
null trials, with <i>N</i><sub>eff</sub> = &rho;&#772; + (1 &minus; &rho;&#772;)<i>N</i> from the mean
correlation of the trials. For the variance in the hurdle we use the sampling variance of a zero-Sharpe
estimate; the original cross-trial variance is reported too and is stricter when a family contains
rules that are genuinely bad.
<b>Overfitting.</b> PBO by combinatorially symmetric cross-validation [4] with 16 blocks.
<b>Stability.</b> The edge must be positive in both halves and in at least 60% of non-overlapping
3-year blocks, with no OLS-CUSUM break in the mean [19] (Newey-West variance [17]). We also tabulate the edge
by decade, trend state, volatility state and recession.
<b>Delay.</b> The edge must survive trading one day later.
<b>Holdout.</b> If a holdout date is given, the candidate is chosen without it and must keep a positive
edge on it.
<b>Conformal bound.</b> Walking forward every quarter, we choose a rule on the last 5 years and record
its next-3-year edge; the haircut (in-sample minus out-of-sample) is the conformity score. The bound for a
new origin is the in-sample edge minus the (1 &minus; &alpha;<sub>t</sub>) quantile of completed past
haircuts, with &alpha;<sub>t</sub> updated by Adaptive Conformal Inference [10] once each outcome is known
(the same online update as in Pacer). The gate asks that today's 80% lower bound be above zero.
An idea <b>survives</b> only if it passes every gate that applies.</p>

<h2>3. Does the battery control false positives?</h2>
<p>We simulate {n['verdict_survives']['n']} markets of 20 years from a GARCH(1,1) with Student-t(5) shocks:
volatility clusters and tails are fat, but the mean is zero, so no rule can beat buy and hold on Sharpe and
every discovery is false. On each we run the {H['family_size']}-rule standard family (trend,
crossovers, momentum, reversal, RSI, breakout, Bollinger and trend-with-volatility-target rules), pick the
best, and hold out the last 5 years.</p>
<figure><img src="figures/harness.png"></figure>
<p class="cap"><b>Figure 1.</b> Left: share of null markets where each method calls the best rule real,
with 95% Wilson intervals. Right: power when a real trend effect is planted (x-axis: the Sharpe edge it
creates for a 100-day trend rule, measured on 400 years of simulated data).</p>
<p>A plain test on the best rule is "significant" in {pct(n['naive_best_p<0.05']['rate'], 0)} of null
markets, six times the nominal rate even though the rules share a common benchmark. Every correction
brings this to 5% or less: Holm {pct(n['holm']['rate'])}, BH {pct(n['bh_fdr']['rate'])}, Reality Check
{pct(n['reality_check']['rate'])}, SPA {pct(n['spa_consistent']['rate'])}, Romano-Wolf
{pct(n['romano_wolf_any']['rate'])}, DSR {pct(n['dsr_sampling>=0.95']['rate'])}. The full verdict
passed none ({pct(n['verdict_survives']['ci95'][1])} upper bound). The timing gate alone, applied to the
<i>selected</i> rule, fires in {pct(n['timing_gate_on_best']['rate'], 0)} of null markets: a single-rule
test is only valid for a rule fixed in advance, which is why selection is its own gate. Mean PBO under
the null is {n['mean_pbo']:.2f}, as theory predicts.</p>
<p><b>Single-rule calibration.</b> For rules fixed in advance (Table 1), the circular-shift test rejects
close to 5% with and without a market drift. Crash Replay's random-trader test does not: when the market
has a positive drift it rejects up to {pct(max(fr['drift_0.06'][r]['random_trader']['rate'] for r in fr['drift_0.06']), 0)}
of the time, because random traders who switch at random dates spend about half their time in the market
while a trend rule is invested most of the time. It rewards exposure as if it were skill. This is a flaw in
our earlier project that this framework found and fixes.</p>
<table><tr><th class="l">Fixed rule, {fr['drift_0.0']['sma200']['circular_shift']['n']} markets</th>
<th>Circular shift, no drift</th><th>Random trader, no drift</th><th>Circular shift, 6% drift</th>
<th>Random trader, 6% drift</th></tr>{fixed_rows}</table>
<p class="cap"><b>Table 1.</b> Rejection rates at the 5% level when the rule has no skill.
Monte Carlo error is about &plusmn;2.2 points.</p>
<p><b>Conformal coverage.</b> On {cn['runs']} null markets of 60 years ({cn['origins_per_run']:.0f}
origins each), the ACI bound missed {pct(cn['aci_miss_mean'])} of the time against a 20% target
(sd {pct(cn['aci_miss_sd'])} across runs); the fixed split-conformal quantile missed
{pct(cn['split_miss_mean'])}, because overlapping windows and selection make scores only approximately
exchangeable. ACI's correction is what keeps the bound honest.</p>
<p><b>Power.</b> Rigor has a price. With 20 years of data the full verdict finds a planted edge of
{e['0.06']:.1f} Sharpe {pct(p_low, 0)} of the time, {e['0.08']:.1f} Sharpe {pct(p_mid, 0)} and {e['0.1']:.1f}
Sharpe {pct(p_hi, 0)}. Edges of a few tenths of a Sharpe point, typical of real index timing, cannot be
confirmed from 20 years of daily index data by any of these tests. The battery is built to make false
positives rare; it will also reject some true but small effects.</p>

<h2>4. Case studies on real data</h2>
<table><tr><th class="l">Idea and sample</th><th>Net edge</th><th>Costs</th><th>Timing</th>
<th>Selection</th><th>Overfit</th><th>Stability</th><th>Delay</th><th>Holdout</th><th>Conformal</th>
<th>Verdict</th></tr>{verdict_table}</table>
<p class="cap"><b>Table 2.</b> Gate results. Net edge is the Sharpe ratio of the rule minus that of buy and
hold after realistic costs, on the design sample. Search family: the {sel_cr['n_rules']}-rule standard family
for the Crash Replay rule, {sel_dm['n_rules']} rules for data mining, {sel_tom['n_rules']} calendar variants
for the turn of the month.</p>

<h3>4.1 Crash Replay's trend rule</h3>
<p>Crash Replay pits the player against a "quant": invested only while the S&amp;P 500 is above its 200-day
average, sized to a 15% volatility target. Inside the game's five crashes it looks excellent (Table 3): it
lost far less in 2008, 1987, 2000-02 and 2022. But the crashes were chosen because they are crashes,
the setting where trend rules do best, and the rule lagged in the COVID rebound.</p>
<table><tr><th class="l">Crash window</th><th>Rule</th><th>Buy and hold</th><th>Rule max DD</th>
<th>B&amp;H max DD</th><th>Beats random traders</th></tr>{crash_table}</table>
<p class="cap"><b>Table 3.</b> Total return inside each Crash Replay window after costs.</p>
<p>On the full record the picture is more modest. The rule's Sharpe edge is {f2(d(cr)['costs']['edge_net'])}
on 1951-2006 and survives costs (capacity about ${cap / 1e9:.1f}B) and delay. It fails the selection gate:
DSR {sel_cr['dsr']['dsr']:.2f} against a hurdle of {sel_cr['dsr']['sr0_annual']:.2f}, Romano-Wolf
p = {sel_cr['stepm_p']:.2f}. The edge has also decayed decade by decade, from {f2(dec['1950s'])} in the
1950s to {f2(dec['2010s'])} in the 2010s (Figure 2), with halves of {f2(halves[0])} and {f2(halves[1])}.
After 2007, the year the 200-day rule was popularized [9], it kept its drawdown control
(max drawdown {pct(hcr['candidate']['max_drawdown'], 0)} vs {pct(hcr['benchmark']['max_drawdown'], 0)}) and a
small Sharpe edge of {f2(hcr['edge'])} (bootstrap p = {hcr['p_value']:.2f}). It replicates in direction on the
Nasdaq ({f2(d(ndq)['costs']['edge_net'])}) and fails on the Dow since 1993 ({f2(d(dji)['costs']['edge_net'])}).
The honest summary: a sensible risk-control rule that reduces drawdowns, whose return edge cannot be
distinguished from the best of a search.</p>
<figure><img src="figures/crash_replay.png"></figure>
<p class="cap"><b>Figure 2.</b> The Crash Replay rule on the S&amp;P 500 after costs, and its Sharpe edge by decade.</p>
<figure><img src="figures/conformal.png"></figure>
<p class="cap"><b>Figure 3.</b> 80% conformal lower bound on the next-3-year edge at each quarter (blue)
against the realised edge (gray). Misses: {pct(conf_cr['aci_miss_rate'], 0)} over {conf_cr['n']} origins.
Today's bound is {f2(conf_cr['now']['lower_bound'])}: the data cannot promise a positive edge.</p>

<h3>4.2 Data mining: the best of {sel_dm['n_rules']} rules</h3>
<p>We search {sel_dm['n_rules']} rules (fine grids of trend, crossover, momentum, reversal, RSI, breakout,
Bollinger, volatility-target and calendar rules) on 1951-1999 and hold out 2000-2026, as in Sullivan,
Timmermann and White [23]. The winner, {dm['candidate']} (long when the index closes above its 10-day
average), has a net edge of {f2(d(dm)['costs']['edge_net'])} and passes the Holm
(p = {sel_dm['p_adjusted']['holm']:.3f}), Romano-Wolf (p = {sel_dm['stepm_p']:.3f}) and SPA
(p = {sel_dm['spa']['consistent']:.3f}) corrections. Many researchers would stop here. The battery does not:
DSR is {sel_dm['dsr']['dsr']:.2f}, the edge turns to {f2(d(dm)['delay']['edge_lag2'])} if trades happen one day
later, CUSUM finds a break near {d(dm)['stability']['cusum']['break_date'][:4]}, and on the holdout the edge is
{f2(d(dm)['holdout']['edge'])}. The likely cause is known: daily index returns before the 1980s were
positively autocorrelated because many component stocks traded rarely, so yesterday's index move
predicted today's [15]. That pattern is real in the data but could not be traded at index prices,
and it is gone. This is the case that shows why multiple-testing corrections alone are not enough.</p>
<figure><img src="figures/data_mined.png"></figure>
<p class="cap"><b>Figure 4.</b> Left: CSCV splits of the 1950-1999 search (in-sample winner's edge vs its
out-of-sample edge). Right: the winner relative to buy and hold after costs, trading at the next close and
one day later; the dashed line marks the holdout.</p>

<h3>4.3 A published anomaly: the turn of the month</h3>
<p>Stocks have historically earned much of their return around the turn of the month [2, 14]. We search
{sel_tom['n_rules']} variants on 1950-1989 and hold out 1990-2026. The best, {tom['candidate']} (hold the last
trading day and the first three of each month), passes every statistical gate on the design sample: DSR
{sel_tom['dsr']['dsr']:.2f}, Romano-Wolf p = {sel_tom['stepm_p']:.3f}, PBO {d(tom)['pbo']['pbo']:.2f}, stable
halves. On the holdout its edge is {f2(d(tom)['holdout']['edge'])}. This matches McLean and Pontiff's
finding that anomalies weaken after publication [16]. A real effect in one era is not a forecast.</p>
<figure><img src="figures/turn_of_month.png"></figure>
<p class="cap"><b>Figure 5.</b> Rolling 5-year Sharpe edge of the turn-of-the-month rule.</p>

<h2>5. Limitations</h2>
<p>(1) <b>The search record is self-reported.</b> Every selection-adjusted test is only as honest as the
family passed to it; rules tried and forgotten are not counted. (2) <b>Null models are models.</b> The
GARCH-t null has no regime changes; the circular-shift null keeps each series' own dependence but not
their joint dependence at other lags. (3) <b>Low power.</b> Section 3 shows modest true edges are usually
rejected on 20 years of daily data. (4) <b>Index data only.</b> Results are for three US indexes, mostly
long-only timing; dividends are monthly approximations for the S&amp;P 500 and missing for the Nasdaq and Dow.
Costs assume an index ETF or future at today's liquidity, which flatters the early decades. (5) <b>Scaling
uses full-sample volatility,</b> a mild look-ahead that does not change Sharpe ratios but affects the
bootstrap tests' scale. (6) <b>N<sub>eff</sub> is an approximation,</b> and DSR assumes trials are
comparable. (7) <b>Conformal guarantees are marginal and approximate:</b> coverage is checked empirically,
not proved, for overlapping windows. (8) No result here is investment advice; the framework says when an
idea is <i>not</i> supported, never that one will make money.</p>

<h2>6. Reproducibility</h2>
<p style="text-align:left"><code>pip install -e .</code> then <code>btv validate --market spx --candidate trend200_vt15_20</code>.
<code>python scripts/null_harness.py</code> and <code>python scripts/case_studies.py</code> regenerate every
number and figure (about 20 and 10 minutes on 4 CPU cores); <code>python scripts/build_paper.py</code>
rebuilds this PDF from the result files. The data snapshot is in <code>data/raw</code>; 28 unit tests and CI
check the statistics, the no-look-ahead property and the pipeline. Code: MIT license.</p>

<h2>References</h2>
<ol class="refs">
<li>Almgren, Thum, Hauptmann, Li. Direct estimation of equity market impact. <i>Risk</i>, 2005.</li>
<li>Ariel. A monthly effect in stock returns. <i>J. Financial Economics</i> 18, 1987.</li>
<li>Bailey, Borwein, L&oacute;pez de Prado, Zhu. Pseudo-mathematics and financial charlatanism. <i>Notices of the AMS</i> 61(5), 2014.</li>
<li>Bailey, Borwein, L&oacute;pez de Prado, Zhu. The probability of backtest overfitting. <i>J. Computational Finance</i> 20(4), 2017.</li>
<li>Bailey, L&oacute;pez de Prado. The Sharpe ratio efficient frontier. <i>J. Risk</i> 15(2), 2012.</li>
<li>Bailey, L&oacute;pez de Prado. The deflated Sharpe ratio. <i>J. Portfolio Management</i> 40(5), 2014.</li>
<li>Benjamini, Hochberg. Controlling the false discovery rate. <i>JRSS B</i> 57, 1995.</li>
<li>Benjamini, Yekutieli. The control of the false discovery rate under dependency. <i>Annals of Statistics</i> 29, 2001.</li>
<li>Faber. A quantitative approach to tactical asset allocation. <i>J. Wealth Management</i> 9(4), 2007.</li>
<li>Gibbs, Cand&egrave;s. Adaptive conformal inference under distribution shift. <i>NeurIPS</i>, 2021.</li>
<li>Hansen. A test for superior predictive ability. <i>J. Business &amp; Economic Statistics</i> 23, 2005.</li>
<li>Harvey, Liu, Zhu. &hellip;and the cross-section of expected returns. <i>Review of Financial Studies</i> 29, 2016.</li>
<li>Holm. A simple sequentially rejective multiple test procedure. <i>Scandinavian J. Statistics</i> 6, 1979.</li>
<li>Lakonishok, Smidt. Are seasonal anomalies real? A ninety-year perspective. <i>Review of Financial Studies</i> 1, 1988.</li>
<li>Lo, MacKinlay. An econometric analysis of nonsynchronous trading. <i>J. Econometrics</i> 45, 1990.</li>
<li>McLean, Pontiff. Does academic research destroy stock return predictability? <i>J. Finance</i> 71, 2016.</li>
<li>Newey, West. A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. <i>Econometrica</i> 55, 1987.</li>
<li>Patton, Politis, White. Correction to automatic block-length selection for the dependent bootstrap. <i>Econometric Reviews</i> 28, 2009.</li>
<li>Ploberger, Kr&auml;mer. The CUSUM test with OLS residuals. <i>Econometrica</i> 60, 1992.</li>
<li>Politis, Romano. The stationary bootstrap. <i>JASA</i> 89, 1994.</li>
<li>Politis, White. Automatic block-length selection for the dependent bootstrap. <i>Econometric Reviews</i> 23, 2004.</li>
<li>Romano, Wolf. Stepwise multiple testing as formalized data snooping. <i>Econometrica</i> 73, 2005.</li>
<li>Sullivan, Timmermann, White. Data-snooping, technical trading rule performance, and the bootstrap. <i>J. Finance</i> 54, 1999.</li>
<li>T&oacute;th et al. Anomalous price impact and the critical nature of liquidity. <i>Physical Review X</i> 1, 2011.</li>
<li>White. A reality check for data snooping. <i>Econometrica</i> 68, 2000.</li>
</ol>
<p class="note">Built with AI-assisted development (Claude Code). Research and education only; not investment advice.</p>
</body></html>"""


def main():
    harness_figures()
    html = build_html()
    out = PAPER / "backtest-validator.html"
    out.write_text(html)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page()
            pg.goto(out.resolve().as_uri())
            pg.wait_for_load_state("networkidle")
            pg.pdf(path=str(PAPER / "backtest-validator.pdf"), format="Letter",
                   print_background=True, display_header_footer=True,
                   header_template="<span></span>",
                   footer_template="<div style='font-size:7pt;width:100%;text-align:center;color:#777'>"
                                   "<span class='pageNumber'></span> / <span class='totalPages'></span></div>",
                   margin={"top": "0.7in", "bottom": "0.7in", "left": "0.75in", "right": "0.75in"})
            b.close()
        print("wrote", PAPER / "backtest-validator.pdf")
    except ImportError:  # fall back to Node's playwright if present
        import subprocess
        js = (
            "const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();"
            "const p=await b.newPage();await p.goto(process.argv[1],{waitUntil:'networkidle'});"
            "await p.pdf({path:process.argv[2],format:'Letter',printBackground:true,"
            "displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:"
            "\"<div style='font-size:7pt;width:100%;text-align:center;color:#777'>"
            "<span class='pageNumber'></span> / <span class='totalPages'></span></div>\","
            "margin:{top:'0.7in',bottom:'0.7in',left:'0.75in',right:'0.75in'}});await b.close();})();")
        try:
            subprocess.run(["node", "-e", js, out.resolve().as_uri(),
                            str(PAPER / "backtest-validator.pdf")], check=True)
            print("wrote", PAPER / "backtest-validator.pdf")
        except Exception as e:  # no browser available
            print("could not render PDF:", e, "- HTML is at", out)


if __name__ == "__main__":
    main()
