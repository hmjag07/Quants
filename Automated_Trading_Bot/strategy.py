from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    fast: int = 20
    slow: int = 50
    mom_lookback: int = 10
    atr_period: int = 14


@dataclass(frozen=True)
class Signal:
    target: int            # +1 / 0 / -1
    close: float
    atr: float
    fast_sma: float
    slow_sma: float
    momentum: float
    bar_time: object


def compute_indicators(df, p: StrategyParams):
    """Returns a copy of df with sma_fast, sma_slow, mom, atr columns added."""
    out = df.copy()
    out["sma_fast"] = out["close"].rolling(p.fast).mean()
    out["sma_slow"] = out["close"].rolling(p.slow).mean()
    out["mom"] = out["close"] / out["close"].shift(p.mom_lookback) - 1.0     # rate of change
    prev_close = out["close"].shift(1)
    tr = pd.concat([out["high"] - out["low"],
                    (out["high"] - prev_close).abs(),
                    (out["low"] - prev_close).abs()], axis=1).max(axis=1)     # skips NaN on row 0
    out["atr"] = tr.rolling(p.atr_period).mean()                              # simple-average ATR
    return out


def target_series(ind):
    """Vectorised target position for every bar of an indicator frame."""
    long_ = (ind["sma_fast"] > ind["sma_slow"]) & (ind["mom"] > 0)
    short_ = (ind["sma_fast"] < ind["sma_slow"]) & (ind["mom"] < 0)
    return pd.Series(np.where(long_, 1, np.where(short_, -1, 0)), index=ind.index)


def latest_signal(df, p: StrategyParams):
    """Signal on the last bar of df. Warm-up (any NaN indicator) -> flat."""
    ind = compute_indicators(df, p)
    row = ind.iloc[-1]
    if row[["sma_fast", "sma_slow", "mom", "atr"]].isna().any():
        target = 0
    else:
        target = int(target_series(ind.iloc[[-1]]).iloc[0])
    return Signal(target, float(row["close"]), float(row["atr"]), float(row["sma_fast"]),
                  float(row["sma_slow"]), float(row["mom"]), row["time"])