### Results (real NSE data from prices.csv)

Test period 2025-10-30 to 2026-10-15, 20 stocks, one-step-ahead forecast of the next day's log return. Walk-forward: models never see the test period when fitting (ARIMA is re-estimated every 21 days using only past data).

| Model | RMSE (%) | MAE (%) | Direction acc. (%) | vs ARIMA RMSE (%) | 95% CI | DM p |
|---|---|---|---|---|---|---|
| Zero (random walk) | 1.552 | 1.115 | - | +0.4 | [+0.1, +0.6] | 0.011 |
| Expanding mean | 1.555 | 1.117 | 49.8 | +0.2 | [+0.1, +0.3] | 0.020 |
| ARIMA | 1.558 | 1.120 | 50.2 | - | - | - |
| Ridge | 1.555 | 1.118 | 48.4 | +0.2 | [-0.1, +0.5] | 0.297 |
| LSTM (returns only) | 1.557 | 1.120 | 48.1 | +0.0 | [-0.2, +0.3] | 0.904 |
| LSTM (engineered features) | 1.558 | 1.120 | 48.8 | -0.0 | [-0.4, +0.4] | 0.955 |

Single LSTM seeds (test RMSE %): 1.562, 1.566, 1.557, 1.561, 1.555 (range 1.555-1.566); the table uses the average of all seeds.

**Reading it:** No statistically reliable difference between the LSTM and ARIMA: -0.0% RMSE vs ARIMA (95% CI -0.4% to +0.4%, Diebold-Mariano p = 0.955). That is the normal outcome for daily returns, which carry very little predictable signal. Against the do-nothing forecast (predict 0) the LSTM is not distinguishable either (-0.4%), so any 'win' over ARIMA should not be read as skill.

![Results](images/ml_results.png)
