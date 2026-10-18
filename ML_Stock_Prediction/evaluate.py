import numpy as np


def rmse(pred, y):
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def mae(pred, y):
    return float(np.mean(np.abs(pred - y)))


def directional_accuracy(pred, y):
    """Share of (stock, day) pairs where the predicted sign equals the actual sign (days with y == 0 skipped)."""
    m = y != 0
    return float(np.mean(np.sign(pred[m]) == np.sign(y[m])))


def daily_loss(pred, y):
    """Mean squared error across stocks for each day -> vector of length n_days."""
    return np.mean((pred - y) ** 2, axis=1)


def improvement(loss_model, loss_ref):
    """% reduction in RMSE of the model relative to the reference (positive = model better)."""
    return 100.0 * (1.0 - np.sqrt(np.mean(loss_model)) / np.sqrt(np.mean(loss_ref)))


def diebold_mariano(loss_a, loss_b, lag=5):
    """Two-sided test of equal predictive accuracy (squared-error loss, one-step forecasts).
    d_t = loss_a - loss_b. Negative mean -> model A is more accurate. Newey-West variance (Bartlett weights)."""
    from math import erf, sqrt
    d = np.asarray(loss_a) - np.asarray(loss_b)
    T = len(d)
    dc = d - d.mean()
    lrv = np.sum(dc * dc) / T
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * np.sum(dc[k:] * dc[:-k]) / T
    if lrv <= 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(lrv / T)
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / sqrt(2.0))))
    return float(stat), float(p)


def block_bootstrap_improvement(loss_model, loss_ref, block=10, n_boot=2000, seed=0):
    """95% interval for the % RMSE improvement, resampling blocks of consecutive days
    (keeps volatility clustering and autocorrelation)."""
    rng = np.random.default_rng(seed)
    T = len(loss_model)
    n_blocks = int(np.ceil(T / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:T] % T       # circular blocks
        out[b] = improvement(loss_model[idx], loss_ref[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))