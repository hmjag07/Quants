import argparse
import math
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from Bot import Bot, StateStore
from Broker_paper import PaperBroker
from Config import Settings, load_dotenv
from Trade_logger import TradeLogger


# ---------------------------------------------------------------- data
def make_synthetic_bars(n=3000, seed=0, start="2024-01-01", vol_per_bar=0.0006, freq="h"):
    """Regime-switching random walk (trends up, trends down, ranges) with plausible OHLC."""
    rng = np.random.default_rng(seed)
    drifts = np.empty(n)
    i = 0
    while i < n:
        length = int(rng.integers(80, 400))
        drifts[i:i + length] = rng.choice([-1.0, 0.0, 1.0]) * vol_per_bar * 0.25
        i += length
    rets = drifts + rng.normal(0, vol_per_bar, n)
    close = 1.10 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[1.10], close[:-1]])
    wiggle = np.abs(rng.normal(0, vol_per_bar * 0.5, n)) * close
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    time_ = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"time": time_, "open": open_, "high": high, "low": low,
                         "close": close, "volume": 1.0})


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "volume" not in df:
        df["volume"] = 0.0
    return df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").reset_index(drop=True)


# ---------------------------------------------------------------- statistics
def performance_stats(equity: pd.Series, periods_per_year: float):
    """Total return, annualised vol and Sharpe (rf = 0) from per-bar equity, max drawdown."""
    equity = pd.Series(equity, dtype=float).reset_index(drop=True)
    rets = equity.pct_change().dropna()
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    vol = rets.std(ddof=1) * math.sqrt(periods_per_year) if len(rets) > 1 else float("nan")
    sharpe = (rets.mean() / rets.std(ddof=1)) * math.sqrt(periods_per_year) if len(rets) > 1 and rets.std(ddof=1) > 0 else float("nan")
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())
    return {"total_return": float(total), "annualised_vol": float(vol), "sharpe": float(sharpe),
            "max_drawdown": max_dd}


def trade_stats(fills):
    """
    Win rate / profit factor from the broker's own fill records (every closing fill carries its
    realised P&L - including stop and target exits, which the CSV log does not price).
    NOTE: for the live bot, exits triggered at the broker (stop/target) appear in trades.csv without a
    price or P&L; Week 8 reconciles them from the broker's transaction history.
    """
    pl = pd.Series([f.pl for f in fills if f.pl != 0.0], dtype=float)
    wins, losses = pl[pl > 0], pl[pl < 0]
    return {"n_closed_trades": int(len(pl)),
            "win_rate": float((pl > 0).mean()) if len(pl) else float("nan"),
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("nan")}


# ---------------------------------------------------------------- engine
@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    stats: dict


def run_backtest(bars: pd.DataFrame, settings: Settings = None, periods_per_year=None):
    s = settings or Settings()
    s.state_file = ""                                             # in-memory state, nothing persisted
    broker = PaperBroker(s.paper_start_equity, s.paper_spread)
    logger = TradeLogger(None)
    bot = Bot(s, broker, logger, store=StateStore(None))
    for row in bars.itertuples(index=False):
        broker.on_bar(row.time, row.open, row.high, row.low, row.close, row.volume)
        bot.step()
    eq, tr = logger.equity_frame(), logger.trades_frame()
    ppy = periods_per_year or (365.25 * 24 * 3600 / s.bar_seconds * (5 / 7))      # FX trades ~5 days/week
    stats = performance_stats(eq["equity"], ppy)
    stats.update(trade_stats(broker.fills))
    stats["n_bars"] = len(bars)
    return BacktestResult(eq, tr, stats)


def print_report(res: BacktestResult):
    st = res.stats
    print(f"bars processed      : {st['n_bars']}")
    print(f"closed trades       : {st['n_closed_trades']}   win rate {st['win_rate']:.1%}   profit factor {st['profit_factor']:.2f}")
    print(f"total return        : {st['total_return']:+.2%}")
    print(f"annualised vol      : {st['annualised_vol']:.2%}")
    print(f"Sharpe (rf=0)       : {st['sharpe']:.2f}")
    print(f"max drawdown        : {st['max_drawdown']:.2%}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--csv")
    g.add_argument("--oanda", action="store_true")
    ap.add_argument("--count", type=int, default=5000)
    args = ap.parse_args()
    load_dotenv()
    if args.synthetic:
        bars, s = make_synthetic_bars(), Settings()
    elif args.csv:
        bars, s = load_csv(args.csv), Settings.from_env()
    else:
        from broker_oanda import OandaBroker
        s = Settings.from_env()
        bars = OandaBroker(s.oanda_token, s.oanda_account_id, s.oanda_env).candles(
            s.instrument, s.granularity, min(args.count, 5000))
    if args.synthetic:
        print("SYNTHETIC DATA with trends built in: this only proves the machinery runs.\n"
              "The numbers below say NOTHING about real-market performance.\n")
    print_report(run_backtest(bars, s))


if __name__ == "__main__":
    main()