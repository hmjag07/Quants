import sys
import numpy as np
import pandas as pd

import evaluate as E
import baselines as B
from features import build_features, FEATURES
from synthetic import make_synthetic
from lstm_model import make_sequences, Scaler, prepare
from run_experiment import split_indices

n_pass = n_fail = 0


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
    else:
        n_fail += 1
        print(f"FAIL: {name} {detail}")


close, volume, sectors = make_synthetic(700, signal=0.0, seed=1)
fs = build_features(close, volume, sectors)
D, T = fs.y.shape

# ---------------- features: shape, hand-calculated values -----------------
check("feature tensor shape", fs.F.shape == (D, T, len(FEATURES)) and fs.y.shape == (D, T))
check("no NaN/inf in features or target", np.isfinite(fs.F).all() and np.isfinite(fs.y).all())
check("market index excluded from stocks", "^NSEI" not in fs.tickers and len(fs.tickers) == 20)
j, d = 3, 100                                                   # stock 3, feature-date index 100
t0 = fs.dates[d]
lp = np.log(close[fs.tickers[j]])
r = lp.diff()
pos = close.index.get_loc(t0)
fi = {f: k for k, f in enumerate(FEATURES)}
check("ret_lag0 = today's log return", abs(fs.F[d, j, fi["ret_lag0"]] - r.iloc[pos]) < 1e-12)
check("ret_lag3 = return 3 days ago", abs(fs.F[d, j, fi["ret_lag3"]] - r.iloc[pos - 3]) < 1e-12)
check("mom_20 = sum of last 20 returns", abs(fs.F[d, j, fi["mom_20"]] - r.iloc[pos - 19:pos + 1].sum()) < 1e-12)
check("vol_5 = std of last 5 returns", abs(fs.F[d, j, fi["vol_5"]] - r.iloc[pos - 4:pos + 1].std()) < 1e-12)
check("target = NEXT day's return", abs(fs.y[d, j] - r.iloc[pos + 1]) < 1e-12)
sec = sectors[fs.tickers[j]]
peers = [s for s in fs.tickers if sectors[s] == sec]
m20 = lambda s: np.log(close[s]).diff().iloc[pos - 19:pos + 1].sum()
check("rel_sector_mom_20 hand-calculated",
      abs(fs.F[d, j, fi["rel_sector_mom_20"]] - (m20(fs.tickers[j]) - np.mean([m20(s) for s in peers]))) < 1e-12)
mk = np.log(close["^NSEI"]).diff().iloc[pos - 19:pos + 1].sum()
check("rel_mkt_mom_20 hand-calculated", abs(fs.F[d, j, fi["rel_mkt_mom_20"]] - (m20(fs.tickers[j]) - mk)) < 1e-12)

# ---------------- NO LOOK-AHEAD -----------------
cut = close.index[500]
fs_cut = build_features(close.loc[:cut], volume.loc[:cut], sectors)
n = len(fs_cut.dates)
check("features at past dates unchanged when the future is deleted",
      np.array_equal(fs_cut.F, fs.F[:n], equal_nan=True) and (fs_cut.dates == fs.dates[:n]).all())
check("targets at past dates unchanged when the future is deleted", np.allclose(fs_cut.y, fs.y[:n]))
close2, vol2 = close.copy(), volume.copy()
close2.iloc[505:] = close2.iloc[505:] * 3.0                       # corrupt prices after day 505
vol2.iloc[505:] = vol2.iloc[505:] * 7.0
fs2 = build_features(close2, vol2, sectors)
upto = fs.dates.get_loc(close.index[504])                         # feature dates through day 504 must not move
check("features through day 504 unchanged when later prices are corrupted",
      np.allclose(fs.F[:upto + 1], fs2.F[:upto + 1], equal_nan=True))
check("corrupting later prices DOES change later features (test has teeth)",
      not np.allclose(fs.F[upto + 5:], fs2.F[upto + 5:]))

# ---------------- splits and sequences -----------------
tr, va, te, a = split_indices(D, 20)
check("splits are ordered and disjoint", tr[-1] < va[0] and va[-1] < te[0] and va[-1] + 1 == te[0])
check("splits cover the end", te[-1] == D - 1)
L = 20
X, yy = make_sequences(fs.F, fs.y, L, [30, 31])
check("sequence shape", X.shape == (2 * T, L, len(FEATURES)) and yy.shape == (2 * T,))
check("sequence window is the L days ENDING at t (stock 2, date 31)",
      np.array_equal(X[T + 2], fs.F[31 - L + 1:32, 2, :]))
check("sequence label is the target of date 31 for stock 2", yy[T + 2] == fs.y[31, 2])
try:
    make_sequences(fs.F, fs.y, L, [5])
    check("too-early index rejected", False)
except ValueError:
    check("too-early index rejected", True)
sc = Scaler(fs.F[tr], fs.y[tr])
Z = sc.x(fs.F[tr]).reshape(-1, len(FEATURES))
check("scaler: training features have mean 0 / std 1", np.allclose(Z.mean(0), 0, atol=1e-9) and np.allclose(Z.std(0), 1, atol=1e-6))
fs_mod_F = fs.F.copy()
fs_mod_F[te] = fs_mod_F[te] * 100 + 5                              # wreck the test period
sc2 = Scaler(fs_mod_F[tr], fs.y[tr])
check("scaler ignores the test period", np.array_equal(sc.mu, sc2.mu) and np.array_equal(sc.sd, sc2.sd))

pre_a = prepare(fs.F, fs.y, tr, va, te, 20)
F_wrecked = fs.F.copy()
F_wrecked[te] = F_wrecked[te] * 50 + 3
y_wrecked = fs.y.copy()
y_wrecked[te] = y_wrecked[te] * 50
pre_b = prepare(F_wrecked, y_wrecked, tr, va, te, 20)
check("LSTM training/validation tensors and scaler do not depend on the test period",
      all(np.array_equal(pre_a[i], pre_b[i]) for i in range(4)) and np.array_equal(pre_a[5].mu, pre_b[5].mu)
      and pre_a[5].ysd == pre_b[5].ysd)
check("LSTM test tensor does depend on test data (test has teeth)", not np.allclose(pre_a[4], pre_b[4]))

# ---------------- baselines -----------------
check("zero forecast is all zeros", not B.zero_forecast((3, 4)).any())
em = B.expanding_mean_forecast(fs.returns, fs.tickers, fs.dates[te])
tk = fs.tickers[0]
chk_date = fs.dates[te[10]]
check("expanding mean = mean of returns up to that date",
      abs(em[10, 0] - fs.returns[tk].loc[:chk_date].dropna().mean()) < 1e-12)

# ARIMA must not peek: forecasts for the first 8 test dates must be identical whether or not the
# later data exists, and must not change when later returns are corrupted.
rs = fs.returns[tk].dropna()
dates8 = fs.dates[te[:8]]
f_full = B.arima_walk_forward(rs, dates8, (1, 0, 1), refit_every=4)
f_trunc = B.arima_walk_forward(rs.loc[:dates8[-1]], dates8, (1, 0, 1), refit_every=4)
bad = rs.copy()
bad.loc[bad.index > dates8[-1]] *= 25
f_bad = B.arima_walk_forward(bad, dates8, (1, 0, 1), refit_every=4)
check("ARIMA forecasts identical when future data is deleted", np.allclose(f_full, f_trunc, atol=1e-10))
check("ARIMA forecasts identical when future data is corrupted", np.allclose(f_full, f_bad, atol=1e-10))
y_bad = rs.copy()
y_bad.loc[y_bad.index == dates8[3]] += 0.2                         # a shock ON a test date must move later forecasts
f_shock = B.arima_walk_forward(y_bad, dates8, (1, 0, 1), refit_every=4)
for k in (0, 4):                                                   # refit steps: fit uses exactly y[:i+1]
    bad_k = rs.copy()
    bad_k.loc[bad_k.index > dates8[k]] *= 25
    one = B.arima_walk_forward(bad_k, dates8[k:k + 1], (1, 0, 1), refit_every=4)[0]
    check(f"ARIMA refit at step {k} ignores everything after that date", abs(one - f_full[k]) < 1e-10)
check("ARIMA does use data up to date t (test has teeth)", not np.allclose(f_full[3:], f_shock[3:], atol=1e-9))
# order selection on a clean AR(1): should pick p>=1
rng = np.random.default_rng(0)
ar = np.zeros(1500)
for i in range(1, 1500):
    ar[i] = 0.5 * ar[i - 1] + 0.01 * rng.standard_normal()
o = B.select_arima_order(ar)
check("AIC order selection finds the AR term in an AR(1) series", o[0] >= 1, str(o))
fc = B.arima_walk_forward(pd.Series(ar, index=pd.bdate_range("2020-01-01", periods=1500)),
                          pd.bdate_range("2020-01-01", periods=1500)[1000:1100], (1, 0, 0), 21)
check("ARIMA beats zero on a genuinely predictable series",
      np.mean((fc - ar[1001:1101]) ** 2) < 0.8 * np.mean(ar[1001:1101] ** 2))

# ---------------- evaluation statistics -----------------
p_ = np.array([[1.0, 2.0], [0.0, -1.0]]); y_ = np.array([[0.0, 4.0], [1.0, -1.0]])
check("RMSE by hand", abs(E.rmse(p_, y_) - np.sqrt((1 + 4 + 1 + 0) / 4)) < 1e-12)
check("MAE by hand", abs(E.mae(p_, y_) - (1 + 2 + 1 + 0) / 4) < 1e-12)
check("direction accuracy by hand", abs(E.directional_accuracy(np.array([1., -1, 2]), np.array([1., 1, -3])) - 1 / 3) < 1e-12)
check("direction accuracy skips y == 0", E.directional_accuracy(np.array([1., 1]), np.array([0., 1])) == 1.0)
check("daily loss shape/values", np.allclose(E.daily_loss(p_, y_), [(1 + 4) / 2, (1 + 0) / 2]))
check("improvement 0 when equal", E.improvement(np.array([1., 2.]), np.array([1., 2.])) == 0)
check("improvement: half the loss -> 29.29%", abs(E.improvement(np.array([0.5]), np.array([1.0])) - 100 * (1 - np.sqrt(.5))) < 1e-9)
rng = np.random.default_rng(1)
a_ = rng.normal(1.0, 0.2, 400) ** 2
b_ = a_ * 1.3
st, p = E.diebold_mariano(a_, b_)
check("DM: clearly better model -> tiny p and negative statistic", p < 1e-6 and st < 0)
x1, x2 = rng.normal(1, 0.3, 400) ** 2, rng.normal(1, 0.3, 400) ** 2
ps = [E.diebold_mariano(rng.normal(1, 0.3, 300) ** 2, rng.normal(1, 0.3, 300) ** 2)[1] for _ in range(200)]
check("DM: equal models -> p-values roughly uniform (false-positive rate near 5%)", 0.01 <= np.mean(np.array(ps) < 0.05) <= 0.11,
      f"rate={np.mean(np.array(ps) < 0.05):.3f}")
def ar1(n, phi):
    e = rng.standard_normal(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x
ps_ac = [E.diebold_mariano(1 + ar1(300, 0.6), np.ones(300))[1] for _ in range(300)]
check("DM: autocorrelated loss gaps are corrected for (false-positive rate stays near 5-10%)",
      np.mean(np.array(ps_ac) < 0.05) < 0.13, f"rate={np.mean(np.array(ps_ac) < 0.05):.3f}")
lo, hi = E.block_bootstrap_improvement(a_, b_, n_boot=500)
check("bootstrap CI brackets the point estimate", lo <= E.improvement(a_, b_) <= hi and lo > 0)
lo, hi = E.block_bootstrap_improvement(x1, x1.copy(), n_boot=200)
check("bootstrap CI of identical models is exactly 0", lo == 0 and hi == 0)

# ---------------- ridge sanity -----------------
Fp = rng.standard_normal((200, 5, 3))
yp = 0.5 * Fp[:, :, 0] + 0.1 * rng.standard_normal((200, 5))
pred, al = B.ridge_forecasts(Fp, yp, np.arange(0, 120), np.arange(120, 160), np.arange(160, 200))
check("ridge recovers a linear signal", np.corrcoef(pred.ravel(), yp[160:].ravel())[0, 1] > 0.95)

# ---------------- end to end (slow) -----------------
if "--full" in sys.argv:
    from run_experiment import run, metrics_table, REF, LSTM_FULL, LSTM_RET
    for sig, label in ((0.0, "noise"), (3.0, "planted signal")):
        c, v, s = make_synthetic(1500, signal=sig, seed=0)
        fs_, te_, y_te, preds, singles, info = run(c, v, s, quick=True, verbose=False)
        tab, _ = metrics_table(y_te, preds, singles)
        row = lambda m: tab[tab.Model == m].iloc[0]
        if sig == 0.0:
            for m in ("Ridge", LSTM_RET, LSTM_FULL):
                r_ = row(m)
                check(f"[noise] {m} shows no significant edge over predicting zero",
                      not (r_["vs zero CI low"] > 0), f"{r_['vs zero CI low']:.2f}")
        else:
            check("[signal] engineered-feature LSTM beats ARIMA with CI above 0", row(LSTM_FULL)["vs arima CI low"] > 0)
            check("[signal] engineered-feature LSTM beats the returns-only LSTM", row(LSTM_FULL)["RMSE (%)"] < row(LSTM_RET)["RMSE (%)"])
            check("[signal] ARIMA cannot see the planted effect (no significant gain over zero)", row(REF)["RMSE (%)"] >= row("Zero (random walk)")["RMSE (%)"] - 0.01)

print(f"\n{n_pass} checks passed, {n_fail} failed")
sys.exit(1 if n_fail else 0)