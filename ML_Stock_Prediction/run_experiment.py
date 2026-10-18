import argparse
import os
import sys
import numpy as np
import pandas as pd

from features import build_features
import baselines as B
import evaluate as E

REF = "ARIMA"
LSTM_FULL = "LSTM (engineered features)"
LSTM_RET = "LSTM (returns only)"


def split_indices(D, L, train_frac=0.6, val_frac=0.2):
    a, b = int(D * train_frac), int(D * (train_frac + val_frac))
    return np.arange(L - 1, a), np.arange(a, b), np.arange(b, D), a


def run(close, volume, sectors, L=20, quick=False, no_lstm=False, n_seeds=5, refit_every=21, verbose=True):
    fs = build_features(close, volume, sectors)
    D, T = fs.y.shape
    tr, va, te, a = split_indices(D, L)
    if len(te) < 60:
        raise ValueError("test period too short (< 60 days) - use a longer history")
    y_te = fs.y[te]
    if verbose:
        print(f"{T} stocks, {D} usable days  |  train {fs.dates[tr[0]].date()}..{fs.dates[tr[-1]].date()}"
              f"  val ..{fs.dates[va[-1]].date()}  test {fs.dates[te[0]].date()}..{fs.dates[te[-1]].date()}")
    preds = {"Zero (random walk)": B.zero_forecast(y_te.shape),
             "Expanding mean": B.expanding_mean_forecast(fs.returns, fs.tickers, fs.dates[te])}
    # the expanding mean needs >= 20 past days; by the test period it always has them
    if verbose:
        print("ARIMA (order chosen on training data, walk-forward on test)...")
    preds[REF], orders = B.arima_forecasts(fs, a, te, refit_every=refit_every, verbose=False)
    if verbose:
        print("Ridge...")
    preds["Ridge"], alpha = B.ridge_forecasts(fs.F, fs.y, tr, va, te)
    info = {"orders": orders, "ridge_alpha": alpha}
    singles = None
    if not no_lstm:
        from lstm_model import lstm_forecasts
        grid, seeds = ((16,), 2) if quick else ((16, 32, 64), n_seeds)
        ret_col = [fs.features.index("ret_lag0")]
        if verbose:
            print("LSTM, returns only...")
        preds[LSTM_RET], _, info["units_ret"] = lstm_forecasts(fs.F, fs.y, tr, va, te, L, ret_col, grid, seeds, verbose)
        if verbose:
            print("LSTM, engineered features...")
        preds[LSTM_FULL], singles, info["units_full"] = lstm_forecasts(fs.F, fs.y, tr, va, te, L, None, grid, seeds, verbose)
    return fs, te, y_te, preds, singles, info


def metrics_table(y_te, preds, singles=None):
    loss = {k: E.daily_loss(v, y_te) for k, v in preds.items()}
    rows = []
    for k, v in preds.items():
        row = {"Model": k, "RMSE (%)": 100 * E.rmse(v, y_te), "MAE (%)": 100 * E.mae(v, y_te),
               "Direction acc. (%)": (100 * E.directional_accuracy(v, y_te)) if k != "Zero (random walk)" else np.nan}
        for ref_name, tag in (("Zero (random walk)", "zero"), (REF, "arima")):
            if k == ref_name:
                row.update({f"vs {tag} (%)": np.nan, f"vs {tag} CI low": np.nan, f"vs {tag} CI high": np.nan,
                            f"DM p vs {tag}": np.nan})
                continue
            imp = E.improvement(loss[k], loss[ref_name])
            lo, hi = E.block_bootstrap_improvement(loss[k], loss[ref_name])
            _, p = E.diebold_mariano(loss[k], loss[ref_name])
            row.update({f"vs {tag} (%)": imp, f"vs {tag} CI low": lo, f"vs {tag} CI high": hi, f"DM p vs {tag}": p})
        rows.append(row)
    tab = pd.DataFrame(rows)
    seed_rmse = None
    if singles is not None:
        seed_rmse = [100 * E.rmse(s, y_te) for s in singles]
    return tab, seed_rmse


def verdict(tab, model=LSTM_FULL, ref=REF):
    """Plain-English, data-driven conclusion. No result is assumed in advance."""
    if model not in set(tab["Model"]):
        return "The LSTM was not run."
    r = tab[tab["Model"] == model].iloc[0]
    imp, lo, hi, p = r["vs arima (%)"], r["vs arima CI low"], r["vs arima CI high"], r["DM p vs arima"]
    zi = r["vs zero (%)"]
    core = f"{imp:+.1f}% RMSE vs ARIMA (95% CI {lo:+.1f}% to {hi:+.1f}%, Diebold-Mariano p = {p:.3f})"
    if lo > 0:
        s = (f"The LSTM with engineered features has a lower out-of-sample RMSE than ARIMA: {core}.")
    elif hi < 0:
        s = (f"The LSTM with engineered features is WORSE than ARIMA out of sample: {core}.")
    else:
        s = (f"No statistically reliable difference between the LSTM and ARIMA: {core}. "
             "That is the normal outcome for daily returns, which carry very little predictable signal.")
    if abs(zi) < 0.5 or r["vs zero CI low"] <= 0 <= r["vs zero CI high"]:
        s += (" Against the do-nothing forecast (predict 0) the LSTM is not distinguishable "
              f"either ({zi:+.1f}%), so any 'win' over ARIMA should not be read as skill.")
    return s


def md_table(tab):
    cols = ["Model", "RMSE (%)", "MAE (%)", "Direction acc. (%)", "vs ARIMA RMSE (%)", "95% CI", "DM p"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in tab.iterrows():
        f = lambda v, n=3: "-" if pd.isna(v) else f"{v:.{n}f}"
        ci = "-" if pd.isna(r["vs arima CI low"]) else f"[{r['vs arima CI low']:+.1f}, {r['vs arima CI high']:+.1f}]"
        vs = "-" if pd.isna(r["vs arima (%)"]) else f"{r['vs arima (%)']:+.1f}"
        lines.append(f"| {r['Model']} | {f(r['RMSE (%)'])} | {f(r['MAE (%)'])} | {f(r['Direction acc. (%)'], 1)} | {vs} | {ci} | {f(r['DM p vs arima'])} |")
    return "\n".join(lines)


def make_figure(tab, y_te, preds, dates, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].barh(tab["Model"][::-1], tab["RMSE (%)"][::-1], color="#4c72b0")
    lo = tab["RMSE (%)"].min()
    ax[0].set_xlim(lo * 0.97, tab["RMSE (%)"].max() * 1.01)
    ax[0].set_xlabel("Test RMSE of next-day return (%), shorter is better\n(axis does not start at 0: differences look bigger than they are)")
    ax[0].set_title("Out-of-sample error")
    if LSTM_FULL in preds:
        d = np.cumsum(E.daily_loss(preds[REF], y_te) - E.daily_loss(preds[LSTM_FULL], y_te))
        ax[1].plot(dates, d, color="#c44e52")
        ax[1].axhline(0, color="k", lw=0.8)
        ax[1].set_title("Cumulative squared-error gap: ARIMA minus LSTM\n(above 0 = LSTM ahead)")
        ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def readme_section(tab, seed_rmse, info, fs, te, source):
    d0, d1 = fs.dates[te[0]].date(), fs.dates[te[-1]].date()
    seeds = ""
    if seed_rmse is not None:
        seeds = (f"\nSingle LSTM seeds (test RMSE %): {', '.join(f'{x:.3f}' for x in seed_rmse)} "
                 f"(range {min(seed_rmse):.3f}-{max(seed_rmse):.3f}); the table uses the average of all seeds.\n")
    return (f"### Results ({source})\n\n"
            f"Test period {d0} to {d1}, {len(fs.tickers)} stocks, one-step-ahead forecast of the next day's log return. "
            "Walk-forward: models never see the test period when fitting (ARIMA is re-estimated every 21 days using only past data).\n\n"
            + md_table(tab) + "\n" + seeds + "\n**Reading it:** " + verdict(tab) + "\n\n"
            "![Results](images/ml_results.png)\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--signal", type=float, default=0.0)
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-lstm", action="store_true")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args(argv)
    try:
        if a.synthetic:
            from synthetic import make_synthetic
            close, volume, sectors = make_synthetic(a.days, signal=a.signal, seed=a.seed)
            source = f"SYNTHETIC data, planted signal = {a.signal}"
        elif a.prices:
            from data import load_prices
            close, volume, sectors = load_prices(a.prices)
            source = f"real NSE data from {a.prices}"
        else:
            print("Give --prices prices.csv (real data) or --synthetic (pipeline test).")
            return 2
        fs, te, y_te, preds, singles, info = run(close, volume, sectors, a.window, a.quick, a.no_lstm, a.seeds)
    except (ValueError, OSError, RuntimeError) as e:
        print("ERROR:", e)
        return 2
    tab, seed_rmse = metrics_table(y_te, preds, singles)
    os.makedirs(a.out_dir, exist_ok=True)
    tab.to_csv(os.path.join(a.out_dir, "metrics.csv"), index=False)
    n_te, T = y_te.shape
    long = pd.DataFrame({"date": np.repeat(fs.dates[te], T), "ticker": np.tile(fs.tickers, n_te), "actual": y_te.reshape(-1)})
    for k, v in preds.items():
        long[k] = v.reshape(-1)
    long.to_csv(os.path.join(a.out_dir, "predictions.csv"), index=False)
    make_figure(tab, y_te, preds, fs.dates[te], os.path.join(a.out_dir, "ml_results.png"))
    with open(os.path.join(a.out_dir, "readme_section.md"), "w", encoding="utf-8") as f:
        f.write(readme_section(tab, seed_rmse, info, fs, te, source))
    print("\n" + md_table(tab))
    if seed_rmse is not None:
        print("single-seed LSTM RMSE %:", [round(x, 3) for x in seed_rmse])
    print("\n" + verdict(tab))
    print(f"\nSaved to {a.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())