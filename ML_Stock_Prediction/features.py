from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURES = ["ret_lag0", "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag4",
            "mom_5", "mom_20", "mom_60", "vol_5", "vol_20",
            "rel_sector_mom_20", "rel_mkt_mom_20", "mkt_ret", "mkt_vol_20", "volume_z"]


@dataclass
class FeatureSet:
    dates: pd.DatetimeIndex      # feature dates t (D of them)
    tickers: list                # the stocks (T of them), index excluded
    F: np.ndarray                # (D, T, K) features at day t
    y: np.ndarray                # (D, T) target = log return of day t+1
    features: list               # K feature names
    returns: pd.DataFrame        # full daily log returns of the stocks (for ARIMA), all dates
    sectors: dict


def build_features(close, volume, sectors, market="^NSEI"):
    stocks = [c for c in close.columns if c != market]
    if market not in close.columns or not stocks:
        raise ValueError("need the market index column plus at least one stock")
    r = np.log(close).diff()
    rs, mkt = r[stocks], r[market]
    W = {}
    for k in range(5):
        W[f"ret_lag{k}"] = rs.shift(k)
    for n in (5, 20, 60):
        W[f"mom_{n}"] = rs.rolling(n).sum()
    for n in (5, 20):
        W[f"vol_{n}"] = rs.rolling(n).std()
    mom20 = W["mom_20"]
    sec_mean = pd.DataFrame(np.nan, index=rs.index, columns=stocks)
    for sec in sorted({sectors[s] for s in stocks}):
        cols = [s for s in stocks if sectors[s] == sec]
        m = mom20[cols].mean(axis=1).to_numpy()
        sec_mean[cols] = np.repeat(m[:, None], len(cols), axis=1)
    W["rel_sector_mom_20"] = mom20 - sec_mean
    W["rel_mkt_mom_20"] = mom20.sub(mkt.rolling(20).sum(), axis=0)
    ones = np.ones((1, len(stocks)))
    W["mkt_ret"] = pd.DataFrame(mkt.to_numpy()[:, None] * ones, index=rs.index, columns=stocks)
    W["mkt_vol_20"] = pd.DataFrame(mkt.rolling(20).std().to_numpy()[:, None] * ones, index=rs.index, columns=stocks)
    lv = np.log(volume[stocks].replace(0, np.nan).ffill())
    W["volume_z"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()

    F = np.stack([W[f].to_numpy() for f in FEATURES], axis=2).astype(float)
    F[~np.isfinite(F)] = np.nan
    y = rs.shift(-1).to_numpy().astype(float)
    valid = ~np.isnan(F).any(axis=(1, 2)) & ~np.isnan(y).any(axis=1)
    pos = np.flatnonzero(valid)
    if len(pos) < 100:
        raise ValueError(f"only {len(pos)} usable days after the warm-up - need more history")
    if not np.all(np.diff(pos) == 1):
        raise ValueError("usable days are not one contiguous block (gaps in the data)")
    sl = slice(pos[0], pos[-1] + 1)
    return FeatureSet(dates=rs.index[sl], tickers=stocks, F=F[sl], y=y[sl], features=list(FEATURES),
                      returns=rs, sectors={s: sectors[s] for s in stocks})