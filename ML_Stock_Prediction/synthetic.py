import numpy as np
import pandas as pd


def make_synthetic(n_days=1500, n_sectors=5, per_sector=4, signal=0.0, seed=0, market="^NSEI"):
    rng = np.random.default_rng(seed)
    names, sectors = [], {}
    for s in range(n_sectors):
        for k in range(per_sector):
            t = f"SYN{s}_{k}"
            names.append(t)
            sectors[t] = f"Sector{s}"
    T = len(names)
    sec_id = np.array([int(n[3]) for n in names])
    beta = rng.uniform(0.7, 1.3, T)
    base_vol = rng.uniform(0.008, 0.014, T)
    R = np.zeros((n_days, T))
    sig2 = base_vol**2
    mvol2 = 0.007**2
    for t in range(n_days):
        mvol2 = 0.1 * 0.007**2 + 0.08 * (R[t - 1].mean() ** 2 if t else 0) + 0.85 * mvol2   # volatility clustering
        m = np.sqrt(mvol2) * rng.standard_normal()
        f = 0.005 * rng.standard_normal(n_sectors)
        eps = rng.standard_normal(T)
        idio = np.sqrt(sig2) * eps
        r = beta * m + f[sec_id] + idio
        if signal and t >= 21:
            mom = R[t - 20:t].sum(axis=0)
            rel = mom - np.array([mom[sec_id == s].mean() for s in range(n_sectors)])[sec_id]
            r = r - signal * 0.003 * np.tanh(rel / 0.05)
        R[t] = r
        sig2 = 0.15 * base_vol**2 + 0.07 * idio**2 + 0.78 * sig2
    dates = pd.bdate_range("2021-10-01", periods=n_days)
    close = pd.DataFrame(100 * np.exp(np.cumsum(R, axis=0)), index=dates, columns=names)
    close[market] = 18000 * np.exp(np.cumsum(R.mean(axis=1)))
    vol = pd.DataFrame(np.exp(rng.normal(13, 0.3, (n_days, T)) + 6 * np.abs(R)), index=dates, columns=names)
    vol[market] = 1e8
    return close, vol, sectors