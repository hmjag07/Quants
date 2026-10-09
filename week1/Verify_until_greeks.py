
import os
import sys
import tempfile
import numpy as np
import pandas as pd

from black_scholes import bs_price
from greeks import bs_greeks, fd_greeks, mc_greeks, practitioner_units
from Mc_Engine import price_vectorized, price_loop, benchmark
from Market_Validation import (bs_price_q, implied_vol, historical_vol, load_chain_csv,
                               validate_chain, calibrate_flat_vol, forward_from_parity,
                               make_demo_chain, summarize)

results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")


def close(a, b, tol=1e-6):
    return abs(float(a) - float(b)) < tol


S, K, r, sig, T = 100.0, 100.0, 0.05, 0.2, 1.0

# ================= Greeks: hand-calculated values (ATM benchmark) =================
gc, gp = bs_greeks(S, K, r, sig, T, "call"), bs_greeks(S, K, r, sig, T, "put")
check("call delta = N(0.35) = 0.6368", close(gc["delta"], 0.63683, 1e-4))
check("gamma = phi(0.35)/(S sigma) = 0.018762", close(gc["gamma"], 0.018762, 1e-5))
check("vega = S phi(0.35) sqrt(T) = 37.524", close(gc["vega"], 37.524, 1e-2))
check("call theta = -6.4140/yr", close(gc["theta"], -6.4140, 1e-3))
check("call rho = K T e^-rT N(d2) = 53.232", close(gc["rho"], 53.232, 1e-2))

# ================= Greeks: structural identities =================
for (s, k, rr, v, t) in [(100, 100, .05, .2, 1.0), (90, 110, .03, .35, .5), (120, 100, .07, .15, .25)]:
    c, p = bs_greeks(s, k, rr, v, t, "call"), bs_greeks(s, k, rr, v, t, "put")
    tag = f"(S={s},K={k},T={t})"
    check(f"delta_call - delta_put = 1 {tag}", close(c["delta"] - p["delta"], 1.0, 1e-12))
    check(f"gamma and vega identical for call/put {tag}",
          close(c["gamma"], p["gamma"], 1e-12) and close(c["vega"], p["vega"], 1e-12))
    check(f"theta_call - theta_put = -r K e^-rT {tag}",
          close(c["theta"] - p["theta"], -rr * k * np.exp(-rr * t), 1e-10))
    check(f"rho_call - rho_put = K T e^-rT {tag}",
          close(c["rho"] - p["rho"], k * t * np.exp(-rr * t), 1e-10))
    check(f"call delta in (0,1), put delta in (-1,0), gamma>0, vega>0 {tag}",
          0 < c["delta"] < 1 and -1 < p["delta"] < 0 and c["gamma"] > 0 and c["vega"] > 0)

check("deep ITM call delta -> 1", bs_greeks(200, 100, r, sig, T, "call")["delta"] > 0.999)
check("deep OTM call delta -> 0", bs_greeks(50, 100, r, sig, T, "call")["delta"] < 0.001)
Sg = np.linspace(60, 140, 801)
check("gamma peaks near the strike", abs(Sg[np.argmax(bs_greeks(Sg, K, r, sig, 0.25)["gamma"])] - K) < 5)
check("gamma at the strike rises as expiry nears",
      bs_greeks(100, 100, r, sig, 0.05)["gamma"] > bs_greeks(100, 100, r, sig, 1.0)["gamma"])
pu = practitioner_units(gc)
check("practitioner units: vega/100, theta/365, rho/100",
      close(pu["vega"], gc["vega"] / 100) and close(pu["theta"], gc["theta"] / 365) and close(pu["rho"], gc["rho"] / 100))

# ================= Greeks: analytic vs finite difference =================
for opt in ("call", "put"):
    a, f = bs_greeks(S, K, r, sig, T, opt), fd_greeks(S, K, r, sig, T, opt)
    for n in ("delta", "gamma", "vega", "theta", "rho"):
        check(f"analytic == finite-diff: {opt} {n}", abs(a[n] - f[n]) < 1e-4 * max(1, abs(a[n])),
              f"({a[n]:.5f} vs {f[n]:.5f})")

# ================= Greeks: analytic vs Monte Carlo (within 4 standard errors) =================
for opt in ("call", "put"):
    a, m = bs_greeks(S, K, r, sig, T, opt), mc_greeks(S, K, r, sig, T, 1_000_000, opt, seed=11)
    for n in ("delta", "gamma", "vega", "rho"):
        est, se = m[n]
        check(f"analytic within 4 SE of MC: {opt} {n}", abs(a[n] - est) < 4 * se,
              f"({a[n]:.5f} vs {est:.5f} +/- {se:.5f})")

# ================= Vectorised engine =================
bs_c = float(bs_price(S, K, r, sig, T, "call"))
bs_p = float(bs_price(S, K, r, sig, T, "put"))
p, se = price_vectorized(S, K, r, sig, T, 200_000, "call", seed=3)
check("vectorised call within 4 SE of BS", abs(p - bs_c) < 4 * se, f"({p:.4f} +/- {se:.4f} vs {bs_c:.4f})")
p, se = price_vectorized(S, K, r, sig, T, 200_000, "put", seed=3)
check("vectorised put within 4 SE of BS", abs(p - bs_p) < 4 * se, f"({p:.4f} +/- {se:.4f} vs {bs_p:.4f})")
check("same seed -> identical result", price_vectorized(S, K, r, sig, T, 5000, seed=9) ==
      price_vectorized(S, K, r, sig, T, 5000, seed=9))

se_plain = price_vectorized(S, K, r, sig, T, 100_000, "call", 5)[1]
se_anti = price_vectorized(S, K, r, sig, T, 100_000, "call", 5, antithetic=True)[1]
se_cv = price_vectorized(S, K, r, sig, T, 100_000, "call", 5, control_variate=True)[1]
se_both = price_vectorized(S, K, r, sig, T, 100_000, "call", 5, antithetic=True, control_variate=True)[1]
check("antithetic reduces std error", se_anti < se_plain, f"({se_anti:.4f} < {se_plain:.4f})")
check("control variate reduces std error", se_cv < se_plain, f"({se_cv:.4f} < {se_plain:.4f})")
check("both together beat either alone", se_both < min(se_anti, se_cv), f"({se_both:.4f})")
for kw, nm in (({"antithetic": True}, "antithetic"), ({"control_variate": True}, "control variate")):
    pv, sev = price_vectorized(S, K, r, sig, T, 400_000, "call", 21, **kw)
    check(f"{nm} estimate is still UNBIASED (within 4 SE of BS)", abs(pv - bs_c) < 4 * sev, f"({pv:.4f} vs {bs_c:.4f})")

se_1k = np.mean([price_vectorized(S, K, r, sig, T, 1_000, "call", s)[1] for s in range(20)])
se_100k = np.mean([price_vectorized(S, K, r, sig, T, 100_000, "call", s)[1] for s in range(20)])
check("100x paths -> ~10x smaller std error (1/sqrt N law)", 7 < se_1k / se_100k < 14, f"(ratio {se_1k / se_100k:.1f})")

check("loop and vectorised agree statistically (loop 20k paths, within 4 SE)",
      abs(price_loop(S, K, r, sig, T, 20_000, "call", seed=2) - bs_c) < 4 * price_vectorized(S, K, r, sig, T, 20_000)[1])
b = benchmark(n_paths=50_000, repeats=2)
check("vectorised is faster than the loop", b["speedup"] > 3, f"({b['speedup']:.0f}x on this machine)")

# ================= Implied volatility =================
round_trip_ok, n_cases = True, 0
for opt in ("call", "put"):
    for Kk in (80, 95, 100, 110, 130):
        for true_vol in (0.1, 0.27, 0.6):
            price = float(bs_price_q(100, Kk, 0.05, true_vol, 0.5, opt))
            iv = implied_vol(price, 100, Kk, 0.05, 0.5, opt)
            n_cases += 1
            if not close(iv, true_vol, 1e-6):
                round_trip_ok = False
                print(f"    round-trip miss: {opt} K={Kk} vol={true_vol} -> got {iv}")
check(f"IV round trip: all {n_cases} price->vol cases recovered to 1e-6", round_trip_ok)
check("IV recovers 27% from its own price", close(implied_vol(float(bs_price_q(100, 105, .05, .27, .5, "call")),
                                                               100, 105, .05, .5, "call"), 0.27, 1e-8))
check("IV with dividend yield round-trips", close(implied_vol(float(bs_price_q(100, 100, .05, .3, 1, "put", q=.02)),
                                                              100, 100, .05, 1, "put", q=.02), 0.3, 1e-8))
check("IV returns nan below intrinsic value", np.isnan(implied_vol(5.0, 120, 100, 0.05, 1.0, "call")))
check("IV returns nan above the no-arbitrage upper bound", np.isnan(implied_vol(101.0, 100, 100, 0.05, 1.0, "call")))
check("bs_price_q with q=0 equals week-1 Black-Scholes",
      close(bs_price_q(100, 105, .03, .25, .75, "call"), float(bs_price(100, 105, .03, .25, .75, "call")), 1e-10))

# ================= Historical volatility =================
rng = np.random.default_rng(0)
true_sigma, n_days = 0.18, 252 * 20
fake_prices = 100 * np.exp(np.cumsum(rng.normal(-0.5 * true_sigma**2 / 252, true_sigma / np.sqrt(252), n_days)))
check("historical_vol recovers 18% from 20y of simulated prices",
      abs(historical_vol(fake_prices) - true_sigma) < 0.01, f"({historical_vol(fake_prices):.4f})")

# ================= Chain loading and validation =================
tmp = tempfile.mkdtemp()
nse = pd.DataFrame({"StrkPric": [24000, 24000, 24500], "OptnTp": ["CE", "PE", "CE"], "ClsPric": [650.5, 480.0, 380.0]})
nse_path = os.path.join(tmp, "nse_like.csv"); nse.to_csv(nse_path, index=False)
loaded = load_chain_csv(nse_path)
check("loader maps NSE-style columns (StrkPric/OptnTp/ClsPric)",
      list(loaded.columns) == ["strike", "option", "market_price"] and len(loaded) == 3)
check("loader maps CE->call, PE->put", list(loaded.option) == ["call", "put", "call"])
bad_path = os.path.join(tmp, "bad.csv"); pd.DataFrame({"a": [1]}).to_csv(bad_path, index=False)
try:
    load_chain_csv(bad_path); raised = False
except ValueError:
    raised = True
check("loader raises a clear error on unknown columns", raised)

S0, r0, T0 = 24000.0, 0.065, 30 / 365
flat = pd.DataFrame([{"strike": k, "option": o, "market_price": float(bs_price_q(S0, k, r0, 0.15, T0, o))}
                     for k in range(23000, 25001, 250) for o in ("call", "put")])
d_exact = validate_chain(flat, S0, r0, T0, sigma=0.15)
check("validate_chain: error = 0 when market is exactly flat-vol BS", d_exact.abs_err.abs().max() < 1e-8)
check("validate_chain: implied vol recovered = 15% everywhere", (d_exact.iv - 0.15).abs().max() < 1e-6)
d_off = validate_chain(flat, S0, r0, T0, sigma=0.20)
check("validate_chain: wrong sigma -> model overprices (positive error)", (d_off.abs_err > 0).all())
check("calibrate_flat_vol recovers 15%", close(calibrate_flat_vol(flat, S0, r0, T0), 0.15, 1e-3))
d_filt = validate_chain(flat, S0, r0, T0, 0.15, min_price=300)
check("min_price filter drops cheap options and keeps only those >= threshold",
      0 < len(d_filt) < len(flat) and (d_filt.market_price >= 300).all(),
      f"(kept {len(d_filt)} of {len(flat)})")

smile = make_demo_chain(S0, r0, T0)
d_smile = validate_chain(smile, S0, r0, T0, calibrate_flat_vol(smile, S0, r0, T0))
tab = summarize(d_smile)
check("smile data: errors are SMALL near the money and LARGE in the wings (the smile)",
      tab["mean abs % err"].iloc[0] < tab["mean abs % err"].iloc[-1],
      f"({tab['mean abs % err'].iloc[0]:.1f}% ATM vs {tab['mean abs % err'].iloc[-1]:.1f}% wings)")
F_est = forward_from_parity(flat, r0, T0)
check("forward recovered from put-call parity = S e^{rT}", abs(F_est / (S0 * np.exp(r0 * T0)) - 1) < 1e-6,
      f"({F_est:.2f})")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)