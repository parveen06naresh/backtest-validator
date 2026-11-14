"""Command line interface.

    btv validate --market spx --family standard --candidate trend200_vt15_20
    btv validate --market spx --family dense --candidate best --holdout 2000-01-01
    btv rules --family standard
    btv null --reps 50

Every command prints a plain-text report; ``--json PATH`` also saves the
full details.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

from . import Config, load, nulls, strategies, validate

FAMILIES = {"standard": strategies.standard_family, "dense": strategies.dense_family,
            "vol": strategies.vol_family}


def _clean(o):
    import numpy as np
    import pandas as pd
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, pd.Timestamp):
        return str(o.date())
    return o


def main(argv=None):
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(prog="btv", description="Backtest Validator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="run the full battery on one idea")
    v.add_argument("--market", default="spx", choices=["spx", "ndq", "dji", "synthetic"])
    v.add_argument("--start", default=None)
    v.add_argument("--end", default=None)
    v.add_argument("--family", default="standard", choices=sorted(FAMILIES))
    v.add_argument("--candidate", default="best", help='rule name, or "best"')
    v.add_argument("--holdout", default=None, help="first date of the holdout, YYYY-MM-DD")
    v.add_argument("--reps", type=int, default=1000)
    v.add_argument("--alpha", type=float, default=0.05)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--json", default=None)

    r = sub.add_parser("rules", help="list the rules in a family")
    r.add_argument("--family", default="standard", choices=sorted(FAMILIES))

    n = sub.add_parser("null", help="quick false-positive check on synthetic markets")
    n.add_argument("--reps", type=int, default=50)

    a = ap.parse_args(argv)
    if a.cmd == "rules":
        for rule in FAMILIES[a.family]():
            print(f"{rule.name:<24} {rule.family:<11} {rule.params}")
        return 0
    if a.cmd == "null":
        fam = strategies.standard_family()
        hits = 0
        for s in range(a.reps):
            m = nulls.simulate(seed=s)
            rep = validate(m, fam, "best", cfg=Config(reps=300, capacity=False, seed=s))
            hits += rep.gates["selection"]["pass"]
            print(f"\rnull market {s + 1}/{a.reps}: selection-gate false positives {hits}", end="")
        print(f"\nfalse-positive rate {hits / a.reps:.1%} (target <= 5%)")
        return 0

    if a.market == "synthetic":
        m = nulls.simulate(seed=a.seed)
    else:
        starts = {"spx": "1950-01-03", "ndq": "1972-01-03", "dji": "1992-01-02"}
        m = load(a.market, start=a.start or starts[a.market], end=a.end)
    fam = FAMILIES[a.family]()
    if a.candidate != "best" and a.candidate not in {x.name for x in fam}:
        print(f"unknown rule {a.candidate!r}; see `btv rules --family {a.family}`", file=sys.stderr)
        return 2
    cfg = Config(alpha=a.alpha, reps=a.reps, seed=a.seed, holdout_start=a.holdout)
    rep = validate(m, fam, a.candidate, cfg=cfg)
    print(rep.text())
    print("\nResearch and education only. Not investment advice.")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(_clean(rep.to_dict()), f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
