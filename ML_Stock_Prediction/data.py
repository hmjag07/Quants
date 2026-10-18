import argparse
import sys
import numpy as np
import pandas as pd

MARKET = "^NSEI"

UNIVERSE = {
    "Banks":  ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS"],
    "IT":     ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS"],
    "FMCG":   ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS"],
    "Auto":   ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS"],
}
SECTORS = {t: s for s, ts in UNIVERSE.items() for t in ts}


def download_prices(start, end=None, out_csv="prices.csv"):
    import yfinance as yf
    rows, failed = [], []
    for t in list(SECTORS) + [MARKET]:
        df = None
        for _ in range(2):                                  # one retry
            try:
                df = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
                if len(df):
                    break
            except Exception as e:                          # network hiccup
                print(f"  {t}: {e}")
        if df is None or len(df) < 250:
            failed.append(t)
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        rows.append(pd.DataFrame({"date": df.index, "ticker": t,
                                  "close": df["Close"].to_numpy(), "volume": df["Volume"].to_numpy()}))
        print(f"  {t}: {len(df)} days")
    if not rows:
        raise RuntimeError("Nothing downloaded - check your internet connection.")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out)} rows to {out_csv}")
    if failed:
        print("WARNING - these tickers failed or had < 250 days and were skipped:", failed)
    return out


def to_wide(long_df, market=MARKET, max_missing=0.05, ffill_limit=2):
    """Long table -> (close, volume) wide tables, one column per ticker, one row per date.

    * duplicated (date, ticker) rows and non-positive prices are rejected
    * short gaps (<= ffill_limit days) are forward-filled, tickers with > max_missing gaps are dropped
    * dates where any remaining ticker is still missing are dropped, so every ticker shares
      one identical calendar (the sequence code relies on that)
    """
    d = long_df.copy()
    d["date"] = pd.to_datetime(d["date"])
    if d.duplicated(["date", "ticker"]).any():
        raise ValueError("duplicate (date, ticker) rows in the price file")
    if (d["close"] <= 0).any():
        raise ValueError("non-positive close prices in the price file")
    close = d.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = d.pivot(index="date", columns="ticker", values="volume").sort_index()
    keep = [c for c in close.columns if close[c].isna().mean() <= max_missing]
    dropped = sorted(set(close.columns) - set(keep))
    if dropped:
        print("Dropping tickers with too many missing days:", dropped)
    if market not in keep:
        raise ValueError(f"market index {market} missing from the data")
    close, volume = close[keep].ffill(limit=ffill_limit), volume[keep].ffill(limit=ffill_limit)
    ok = close.notna().all(axis=1)
    return close[ok], volume[ok].fillna(0.0)


def load_prices(path):
    close, volume = to_wide(pd.read_csv(path))
    sectors = {t: SECTORS[t] for t in close.columns if t in SECTORS}
    return close, volume, sectors


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-10-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default="prices.csv")
    a = ap.parse_args()
    try:
        download_prices(a.start, a.end, a.out)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(2)