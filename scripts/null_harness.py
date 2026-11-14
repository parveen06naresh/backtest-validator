"""Check the battery itself on synthetic markets where the truth is known.

1. Size: on GARCH-t markets with no edge, how often does each test (and the
   full verdict) declare the best of ~100 rules a winner? A naive t-test
   on the best rule should "find" something most of the time; the
   corrected tests should stay near or below 5%.
2. Calibration of single-rule timing tests on a fixed rule, with and
   without a positive equity drift (the drift case exposes a flaw in the
   Crash Replay random-trader test: it rewards time in the market).
3. Power: plant a real trend effect of growing strength and record how
   often the battery finds it.

Usage: python scripts/null_harness.py [--reps 400] [--power-reps 100] [--jobs 4]
Writes results/null_harness.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from backtest_validator import Config, nulls, validate  # noqa: E402
from backtest_validator import strategies as S  # noqa: E402
from backtest_validator.permutation import circular_shift_test, random_trader_test  # noqa: E402

N_DAYS = 5040          # 20 years
HOLDOUT_AT = 3780      # last 5 years held out


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float((c - h) / d), float((c + h) / d)]


def one_null(seed: int, signal: float = 0.0) -> dict:
    m = nulls.simulate(N_DAYS, seed=seed, signal=signal)
    cfg = Config(reps=300, capacity=False, seed=seed,
                 holdout_start=str(m.index[HOLDOUT_AT].date()))
    r = validate(m, S.standard_family(), "best", cfg=cfg)
    sel = r.details["selection"]
    g = {k: v["pass"] for k, v in r.gates.items()}
    a = 0.05
    out = {
        "naive_best_p<0.05": sel["p_single"] < a,
        "bonferroni": sel["p_adjusted"]["bonferroni"] < a,
        "holm": sel["p_adjusted"]["holm"] < a,
        "bh_fdr": sel["p_adjusted"]["bh"] < a,
        "reality_check": sel["reality_check_p"] < a,
        "spa_consistent": sel["spa"]["consistent"] < a,
        "romano_wolf_any": len(sel["stepm_survivors"]) > 0,
        "dsr_sampling>=0.95": sel["dsr"]["dsr"] >= 1 - a,
        "dsr_trials>=0.95": sel["dsr_trial_variance"]["dsr"] >= 1 - a,
        "timing_gate_on_best": bool(g["timing"]),
        "selection_gate": bool(g["selection"]),
        "verdict_survives": r.survives,
        "pbo": r.details.get("pbo", {}).get("pbo"),
        "candidate": r.candidate,
    }
    c = r.details.get("conformal", {})
    if c.get("n"):
        out["conformal_miss"] = c["aci_miss_rate"]
        out["conformal_split_miss"] = c["split_miss_rate"]
    return out


def one_fixed(seed: int, drift: float) -> dict:
    m = nulls.simulate(N_DAYS, seed=10_000 + seed, drift=drift)
    out = {}
    for rule in (S.sma_trend(200), S.ma_cross(50, 200), S.tsmom(126)):
        pos = rule.positions(m.close).values
        cs = circular_shift_test(pos, m.excess.values, rule.lookback)
        rt = random_trader_test(pos, m.excess.values, n_traders=500, seed=seed)
        out[rule.name] = {"circular_shift": cs["p_value"] < 0.05, "random_trader": rt["p_value"] < 0.05}
    return out


def one_conformal(seed: int) -> dict:
    """Coverage of the walk-forward conformal bound on a long null market."""
    m = nulls.simulate(15_120, seed=90_000 + seed)   # 60 years, no holdout
    r = validate(m, S.standard_family(), "best", cfg=Config(reps=200, capacity=False, seed=seed))
    c = r.details.get("conformal", {})
    return {"aci": c.get("aci_miss_rate"), "split": c.get("split_miss_rate"), "n": c.get("n")}


def rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    k = int(np.sum(vals))
    return {"rate": k / len(vals), "ci95": wilson(k, len(vals)), "n": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--power-reps", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    res = {"n_days": N_DAYS, "holdout_days": N_DAYS - HOLDOUT_AT,
           "family_size": len(S.standard_family())}
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(one_null, range(args.reps)))
        keys = [k for k in rows[0] if k not in ("pbo", "candidate", "conformal_miss",
                                                 "conformal_split_miss")]
        res["null"] = {k: rate(rows, k) for k in keys}
        res["null"]["mean_pbo"] = float(np.nanmean([r["pbo"] for r in rows]))
        cm = [r["conformal_miss"] for r in rows if "conformal_miss" in r]
        sm = [r["conformal_split_miss"] for r in rows if "conformal_split_miss" in r]
        res["null"]["conformal_miss_mean"] = float(np.mean(cm)) if cm else None
        res["null"]["conformal_split_miss_mean"] = float(np.mean(sm)) if sm else None
        print("null done", round(time.time() - t0), "s", flush=True)

        cr = list(ex.map(one_conformal, range(max(args.reps // 8, 8))))
        res["conformal_null"] = {"target_miss": 0.2, "runs": len(cr),
                                 "aci_miss_mean": float(np.mean([c["aci"] for c in cr])),
                                 "aci_miss_sd": float(np.std([c["aci"] for c in cr])),
                                 "split_miss_mean": float(np.mean([c["split"] for c in cr])),
                                 "origins_per_run": float(np.mean([c["n"] for c in cr]))}
        print("conformal done", round(time.time() - t0), "s", flush=True)

        res["fixed_rule"] = {}
        for drift in (0.0, 0.06):
            fr = list(ex.map(one_fixed, range(args.reps), [drift] * args.reps))
            res["fixed_rule"][f"drift_{drift}"] = {
                rule: {test: rate([f[rule] for f in fr], test) for test in fr[0][rule]}
                for rule in fr[0]}
        print("fixed done", round(time.time() - t0), "s", flush=True)

        res["power"] = {}
        for sig in (0.02, 0.04, 0.06, 0.08, 0.10):
            pr = list(ex.map(one_null, range(50_000, 50_000 + args.power_reps),
                             [sig] * args.power_reps))
            res["power"][str(sig)] = {k: rate(pr, k) for k in
                                      ("naive_best_p<0.05", "romano_wolf_any", "spa_consistent",
                                       "dsr_sampling>=0.95", "selection_gate", "verdict_survives")}
            res["power"][str(sig)]["trend_family_selected"] = float(np.mean(
                [any(t in r["candidate"] for t in ("sma", "cross", "tsmom", "trend", "donchian"))
                 for r in pr]))
            print("power", sig, round(time.time() - t0), "s", flush=True)
    # what edge does each planted signal create? (measured on long samples)
    from backtest_validator import engine, metrics
    edges = {}
    for sig in (0.02, 0.04, 0.06, 0.08, 0.10):
        m = nulls.simulate(100_000, seed=7, signal=sig)
        pos = S.sma_trend(100, True).positions(m.close)
        net = engine.run(pos, m.excess, m.rf, engine.ZERO)
        bh = engine.run(S.buy_and_hold().positions(m.close), m.excess, m.rf, engine.ZERO)
        edges[str(sig)] = metrics.sharpe(net) - metrics.sharpe(bh)
    res["planted_edge_sharpe_sma100_ls"] = edges
    res["seconds"] = round(time.time() - t0)
    out = ROOT / "results" / "null_harness.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["null"], indent=1))


if __name__ == "__main__":
    main()
