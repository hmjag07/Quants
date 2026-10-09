
import sys
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm


# ---------------- Black-Scholes with continuous dividend yield q ----------------
def bs_price_q(S, K, r, sigma, T, option="call", q=0.0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# ---------------- Implied volatility ----------------
def implied_vol(price, S, K, r, T, option="call", q=0.0, lo=1e-4, hi=5.0):
    """Solve bs_price_q(sigma) = price. Returns nan if the price violates no-arbitrage bounds."""
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    if option == "call":
        intrinsic, upper = max(S * disc_q - K * disc_r, 0.0), S * disc_q
    else:
        intrinsic, upper = max(K * disc_r - S * disc_q, 0.0), K * disc_r
    if not (intrinsic < price < upper):
        return float("nan")
    f = lambda s: bs_price_q(S, K, r, s, T, option, q) - price
    try:
        return brentq(f, lo, hi, xtol=1e-10)
    except ValueError:
        return float("nan")


# ---------------- Historical volatility ----------------
def historical_vol(prices, trading_days=252):
    """Annualised standard deviation of daily log returns."""
    logret = np.diff(np.log(np.asarray(prices, dtype=float)))
    return float(logret.std(ddof=1) * np.sqrt(trading_days))


def download_history(ticker="^NSEI", period="1y"):
    """Close prices from Yahoo Finance (needs internet). ^NSEI = NIFTY 50 index."""
    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    close = df["Close"]
    return close.squeeze().dropna()


def forward_from_parity(df, r, T):
    """
    Infer the forward price from the chain itself: at the strike where |C - P| is smallest,
    F = K + e^{rT} (C - P).  Use implied_q = -ln(F/S)/T + r to get the carry for bs_price_q.
    """
    calls = df[df.option == "call"].set_index("strike").market_price
    puts = df[df.option == "put"].set_index("strike").market_price
    both = calls.index.intersection(puts.index)
    K = (calls[both] - puts[both]).abs().idxmin()
    return float(K + np.exp(r * T) * (calls[K] - puts[K]))


# ---------------- Chain loading (flexible column names) ----------------
_STRIKE = ("strike", "StrkPric", "STRIKE_PR", "Strike Price", "STRIKE")
_TYPE = ("option", "OptnTp", "OPTION_TYP", "Option Type", "type")
_PRICE = ("market_price", "ClsPric", "CLOSE", "LTP", "Close", "close", "last_price")


def load_chain_csv(path):
    """Accepts NSE bhavcopy-style or simple CSVs. Output columns: strike, option, market_price."""
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    pick = lambda names: next((c for c in names if c in raw.columns), None)
    sc, tc, pc = pick(_STRIKE), pick(_TYPE), pick(_PRICE)
    if None in (sc, tc, pc):
        raise ValueError(f"Could not find strike/type/price columns. Columns present: {list(raw.columns)}")
    out = pd.DataFrame({
        "strike": pd.to_numeric(raw[sc], errors="coerce"),
        "option": raw[tc].astype(str).str.strip().str.upper().map(
            {"CE": "call", "CALL": "call", "C": "call", "PE": "put", "PUT": "put", "P": "put"}),
        "market_price": pd.to_numeric(raw[pc], errors="coerce"),
    })
    return out.dropna().reset_index(drop=True)


# ---------------- Comparison ----------------
def validate_chain(df, S, r, T, sigma, q=0.0, min_price=1.0):
    """Adds model_price, abs/pct error and implied vol columns. Rows with market_price < min_price are dropped."""
    d = df[df.market_price >= min_price].copy()
    d["model_price"] = [float(bs_price_q(S, k, r, sigma, T, o, q)) for k, o in zip(d.strike, d.option)]
    d["abs_err"] = d.model_price - d.market_price
    d["pct_err"] = 100.0 * d.abs_err / d.market_price
    d["iv"] = [implied_vol(p, S, k, r, T, o, q) for p, k, o in zip(d.market_price, d.strike, d.option)]
    d["moneyness"] = np.log(d.strike / (S * np.exp((r - q) * T)))      # ln(K/F); 0 = at the money
    return d.reset_index(drop=True)


def calibrate_flat_vol(df, S, r, T, q=0.0, min_price=1.0, atm_band=0.05):
    """Single sigma minimising squared price error over near-the-money options."""
    d = df[df.market_price >= min_price].copy()
    d["m"] = np.log(d.strike / (S * np.exp((r - q) * T)))
    d = d[d.m.abs() <= atm_band]
    obj = lambda s: sum((float(bs_price_q(S, k, r, s, T, o, q)) - p) ** 2
                        for k, o, p in zip(d.strike, d.option, d.market_price))
    return float(minimize_scalar(obj, bounds=(0.01, 2.0), method="bounded").x)


def summarize(d, bands=(0.03, 0.07, 0.15)):
    """Error statistics by |ln(K/F)| bucket. This table is what goes in your write-up."""
    rows, edges = [], (0.0,) + tuple(bands) + (np.inf,)
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = d[(d.moneyness.abs() >= lo) & (d.moneyness.abs() < hi)]
        if len(b):
            rows.append({"|ln(K/F)| bucket": f"{lo:.2f} to {hi:.2f}", "n": len(b),
                         "mean abs % err": b.pct_err.abs().mean(),
                         "median abs % err": b.pct_err.abs().median(),
                         "% within 5%": 100.0 * (b.pct_err.abs() <= 5).mean()})
    return pd.DataFrame(rows)


# ---------------- Offline demo (no internet, no data needed) ----------------
def make_demo_chain(S=24000.0, r=0.065, T=30 / 365, seed=0):
    """Synthetic chain with a volatility smile + 1% quote noise, to prove the pipeline works."""
    rng = np.random.default_rng(seed)
    rows = []
    for K in np.arange(S * 0.90, S * 1.10 + 1, S * 0.01):
        K = round(K / 50) * 50
        smile = 0.14 + 1.2 * (np.log(K / S)) ** 2             # higher IV in the wings
        for opt in ("call", "put"):
            p = float(bs_price_q(S, K, r, smile, T, opt)) * (1 + 0.01 * rng.standard_normal())
            rows.append({"strike": K, "option": opt, "market_price": p})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    S, r, T = 24000.0, 0.065, 30 / 365          # r = PLACEHOLDER; use the current RBI T-bill yield
    if "--live" in sys.argv:
        hist = download_history("^NSEI", "1y")
        print(f"NIFTY historical vol (1y): {historical_vol(hist.values):.2%}")
        sys.exit(0)
    print("DEMO on a synthetic chain with a volatility smile (not real market data)\n")
    chain = make_demo_chain(S, r, T)
    sigma = calibrate_flat_vol(chain, S, r, T)
    print(f"Calibrated flat sigma (near-the-money): {sigma:.2%}\n")
    d = validate_chain(chain, S, r, T, sigma)
    print(summarize(d).round(2).to_string(index=False))
    print("\nNote how a flat sigma fits near the money and degrades in the wings - that is the smile.")