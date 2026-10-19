### Real-market validation (NIFTY index options)

- **Data:** NSE option-chain page, expiry 23 Nov 2026, quotes as of 19 Oct 2026 12:16 IST.
- **Prices used:** bid/ask midpoint (last traded price where no valid quote). Out-of-the-money options only; premiums below 1 point(s) excluded. 112 options in total.
- **Inputs:** time to expiry 35.1 days, risk-free rate 6.25%, forward 23,059.48 inferred from put-call parity.
- **Fit:** one flat volatility of **13.81%**, fitted to options within 5% of the forward.

Error of the flat-volatility Black-Scholes price against the market price, by distance from the forward, where |ln(K/F)| is that distance:

| \|ln(K/F)\| bucket | n | mean abs % err | median abs % err | % within 5% | mean abs IV gap (vol pts) |
|---|---|---|---|---|---|
| 0.00 to 0.03 | 28 | 7.9 | 6.9 | 39.3 | 0.6 |
| 0.03 to 0.07 | 36 | 34.2 | 31.7 | 0.0 | 1.7 |
| 0.07 to 0.15 | 48 | 68.0 | 79.1 | 2.1 | 4.3 |

![Volatility smile and pricing error](images/nse_iv_smile.png)

In this sample the implied volatility sits 0.6 vol points from the flat fit near the forward and 4.3 points away in the farthest bucket, so a single volatility fits close to the forward and degrades in the wings, consistent with a volatility smile.
Percentage errors look large for cheap out-of-the-money options even when the volatility gap is small, which is why the gap in volatility points is shown too. These numbers come from one expiry on one day; quote timing and stale prices on quiet strikes affect them.
