import warnings
import numpy as np


def zero_forecast(shape):
    return np.zeros(shape)


def expanding_mean_forecast(returns_df, tickers, dates):
    """returns_df: full daily returns; for each date t, the mean of the stock's returns up to and including t."""
    csum = returns_df[tickers].expanding(min_periods=20).mean()
    return csum.reindex(dates).to_numpy()


def _fit_arima(y, order):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(y, order=order, trend="c").fit()


def select_arima_order(y_train, max_p=2, max_q=2):
    """Lowest-AIC (p, 0, q) on the training returns (scaled by 100 for numerical stability)."""
    best, best_aic = (0, 0, 0), np.inf
    y = np.asarray(y_train) * 100.0
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                aic = _fit_arima(y, (p, 0, q)).aic
            except Exception:
                continue
            if np.isfinite(aic) and aic < best_aic:
                best, best_aic = (p, 0, q), aic
    return best


def arima_walk_forward(r_series, test_dates, order, refit_every=21):
    """One-step-ahead forecasts of r(t+1) for each feature date t in test_dates.

    r_series: pandas Series of daily log returns indexed by date (full history).
    Forecast for date t uses r_series up to and including t - nothing later.
    """
    y = r_series.to_numpy() * 100.0
    idx = r_series.index.get_indexer(test_dates)
    if (idx < 0).any():
        raise ValueError("test date missing from the return series")
    out = np.empty(len(idx))
    res, last_fit, prev = None, 0, None
    for k, i in enumerate(idx):
        if res is None or k - last_fit >= refit_every:
            res = _fit_arima(y[: i + 1], order)
            last_fit = k
        else:
            res = res.append(y[prev + 1: i + 1], refit=False)
        prev = i
        out[k] = float(np.asarray(res.forecast(1))[0]) / 100.0
    return out


def arima_forecasts(fs, n_train, test_idx, refit_every=21, verbose=True):
    """ARIMA for every ticker. Returns (preds[len(test_idx), T], dict of chosen orders)."""
    test_dates = fs.dates[test_idx]
    preds = np.empty((len(test_idx), len(fs.tickers)))
    orders = {}
    for j, t in enumerate(fs.tickers):
        r = fs.returns[t].dropna()
        train_end = fs.dates[n_train - 1]
        order = select_arima_order(r.loc[:train_end].to_numpy())
        orders[t] = order
        preds[:, j] = arima_walk_forward(r, test_dates, order, refit_every)
        if verbose:
            print(f"  ARIMA {t}: order {order}")
    return preds, orders


def ridge_forecasts(F, y, train_idx, val_idx, test_idx, alphas=(1.0, 10.0, 100.0, 1000.0, 10000.0)):
    """Ridge on the day-t feature vector. Standardisation uses TRAINING rows only."""
    from sklearn.linear_model import Ridge
    K = F.shape[2]
    Xtr, ytr = F[train_idx].reshape(-1, K), y[train_idx].reshape(-1)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-12
    z = lambda A: ((A.reshape(-1, K) - mu) / sd)
    best_a, best_mse = None, np.inf
    for a in alphas:
        m = Ridge(alpha=a).fit(z(Xtr), ytr)
        mse = np.mean((m.predict(z(F[val_idx])) - y[val_idx].reshape(-1)) ** 2)
        if mse < best_mse:
            best_a, best_mse = a, mse
    m = Ridge(alpha=best_a).fit(z(Xtr), ytr)
    return m.predict(z(F[test_idx])).reshape(len(test_idx), F.shape[1]), best_a