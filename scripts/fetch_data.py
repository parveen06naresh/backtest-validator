"""Refresh the cached data snapshot in data/raw (needs internet).

Index prices: Yahoo Finance chart endpoint. Rates and recession dates: FRED.
Dividends: Robert Shiller's monthly S&P 500 data (download ie_data.xls from
http://www.econ.yale.edu/~shiller/data.htm and save the columns Date, SP500,
Dividend as data/raw/SHILLER.csv). The committed snapshot ends 2026-10-30;
results in the paper use that snapshot. Standard library only.
"""
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "raw"
FRED = ["TB3MS", "DGS3MO", "DGS10", "USREC"]
YAHOO = {"^GSPC": "SPX", "^IXIC": "NDQ", "^DJI": "DJI", "^VIX": "VIX"}
UA = {"User-Agent": "Mozilla/5.0 (research data fetch)"}


def get(url):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:  # retry with backoff
            print(f"retry {i + 1}: {e}")
            time.sleep(2 ** i)
    raise RuntimeError(url)


def yahoo(sym):
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.request.quote(sym)}"
           f"?period1=-1325635200&period2={int(time.time())}&interval=1d")
    res = json.loads(get(url))["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = ["date,open,high,low,close,volume"]
    for i, ts in enumerate(res["timestamp"]):
        c = q["close"][i]
        if c is None:
            continue
        d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        o, h, lo = (q[k][i] if q[k][i] is not None else c for k in ("open", "high", "low"))
        rows.append(f"{d},{o:.4f},{h:.4f},{lo:.4f},{c:.4f},{q['volume'][i] or 0}")
    return "\n".join(rows) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in FRED:
        (OUT / f"{s}.csv").write_bytes(get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}"))
        print("FRED", s)
    for sym, name in YAHOO.items():
        (OUT / f"{name}.csv").write_text(yahoo(sym))
        print("Yahoo", name)


if __name__ == "__main__":
    main()
