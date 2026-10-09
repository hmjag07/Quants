"""
Week 1 verification suite. Run:   python week1/verify_week1.py
Every line prints PASS/FAIL; the script exits non-zero if anything fails.
"""
import sys
import numpy as np

from payoffs import (stock_payoff, forward_payoff, call_payoff, put_payoff, call_profit,
                     put_profit, straddle_profit, bull_call_spread_profit, butterfly_profit)
from bonds import (zero_coupon_price, coupon_bond_price, forward_price, forward_value,
                   put_call_parity_gap)
from black_scholes import bs_call, bs_put, bs_call_manual

results = []


def check(name, condition, detail=""):
    results.append(condition)
    print(f"[{'PASS' if condition else 'FAIL'}] {name} {detail}")


def close(a, b, tol=1e-4):
    return abs(float(a) - float(b)) < tol


# ---------------- Day 3: linear payoffs ----------------
check("stock long: S_T=110, S0=100 -> +10", close(stock_payoff(110, 100), 10))
check("stock short: S_T=110, S0=100 -> -10", close(stock_payoff(110, 100, -1), -10))
check("forward long: S_T=90, K=100 -> -10", close(forward_payoff(90, 100), -10))
check("forward short is mirror of long",
      np.allclose(forward_payoff([80, 100, 120], 100, -1), -forward_payoff([80, 100, 120], 100, +1)))

# ---------------- Day 4: options ----------------
check("call payoff OTM = 0", close(call_payoff(90, 100), 0))
check("call payoff ITM = 15", close(call_payoff(115, 100), 15))
check("put payoff ITM = 20", close(put_payoff(80, 100), 20))
check("long call max loss = -premium", close(call_profit(50, 100, 5), -5))
check("long call breakeven = K + premium", close(call_profit(105, 100, 5), 0))
check("short call max gain = +premium", close(call_profit(50, 100, 5, -1), 5))
check("long put breakeven = K - premium", close(put_profit(96, 100, 4), 0))
S = np.linspace(0, 300, 3001)
check("long call - short call = 0 everywhere",
      np.allclose(call_profit(S, 100, 5, +1) + call_profit(S, 100, 5, -1), 0))

# ---------------- Day 5: combinations ----------------
check("straddle max loss = both premiums (at S_T=K)", close(straddle_profit(100, 100, 5, 4), -9))
check("straddle breakevens K +/- 9", close(straddle_profit(109, 100, 5, 4), 0) and close(straddle_profit(91, 100, 5, 4), 0))
check("bull spread max loss = net debit 5", close(bull_call_spread_profit(50, 95, 105, 8, 3), -5))
check("bull spread max gain = width - debit = 5", close(bull_call_spread_profit(200, 95, 105, 8, 3), 5))
check("butterfly max profit at middle strike = 10 - 2 = 8", close(butterfly_profit(100, 90, 100, 110, 12, 6, 2), 8))
check("butterfly wings lose net debit 2",
      close(butterfly_profit(50, 90, 100, 110, 12, 6, 2), -2) and close(butterfly_profit(200, 90, 100, 110, 12, 6, 2), -2))

# ---------------- Day 5/6: bonds & forwards ----------------
check("ZCB 100 @ 5% annual comp, 2y = 90.7029", close(zero_coupon_price(100, 0.05, 2, "annual"), 90.7029))
check("ZCB 100 @ 5% cc, 2y = 90.4837", close(zero_coupon_price(100, 0.05, 2), 90.4837))
check("5% coupon, 4% ytm, 3y bond = 102.7751", close(coupon_bond_price(100, 0.05, 0.04, 3), 102.7751))
check("par bond: coupon == ytm -> price 100", close(coupon_bond_price(100, 0.06, 0.06, 5), 100.0))
check("forward S=100,r=5%,T=1 = 105.1271", close(forward_price(100, 0.05, 1), 105.1271))
check("forward value at inception (K = F) = 0", close(forward_value(100, forward_price(100, 0.05, 1), 0.05, 1), 0.0, 1e-9))

# ---------------- Day 6/7: Black-Scholes ----------------
# Hull's textbook example: S=42, K=40, r=10%, sigma=20%, T=0.5 -> call 4.76, put 0.81
check("Hull example call = 4.76", close(bs_call(42, 40, 0.10, 0.20, 0.5), 4.7594, 1e-3))
check("Hull example put  = 0.81", close(bs_put(42, 40, 0.10, 0.20, 0.5), 0.8086, 1e-3))
# Classic ATM benchmark
check("ATM benchmark call = 10.4506", close(bs_call(100, 100, 0.05, 0.2, 1), 10.4506))
check("ATM benchmark put  = 5.5735", close(bs_put(100, 100, 0.05, 0.2, 1), 5.5735))

c, p = bs_call(100, 105, 0.03, 0.25, 0.75), bs_put(100, 105, 0.03, 0.25, 0.75)
check("put-call parity holds", close(put_call_parity_gap(c, p, 100, 105, 0.03, 0.75), 0.0, 1e-9))
check("manual (no-scipy) call matches scipy call", close(bs_call_manual(100, 105, 0.03, 0.25, 0.75), c, 1e-9))
check("call price bounds: max(S-Ke^-rT,0) <= C <= S",
      max(100 - 105 * np.exp(-0.03 * 0.75), 0) <= c <= 100)
check("deep ITM call ~ S - K e^{-rT}", close(bs_call(200, 100, 0.05, 0.2, 1), 200 - 100 * np.exp(-0.05), 1e-3))
check("deep OTM call ~ 0", bs_call(50, 200, 0.05, 0.2, 1) < 1e-6)
check("vol up -> call price up", bs_call(100, 100, 0.05, 0.3, 1) > bs_call(100, 100, 0.05, 0.2, 1))
check("vectorised over strikes returns array of length 5", np.asarray(bs_call(100, np.arange(80, 130, 10), 0.05, 0.2, 1)).shape == (5,))

# ---------------- Bonus: Monte Carlo cross-check (Week 2 preview) ----------------
rng = np.random.default_rng(42)
S0, K, r, sig, T, n = 100, 100, 0.05, 0.2, 1.0, 400_000
Z = rng.standard_normal(n)
ST = S0 * np.exp((r - 0.5 * sig**2) * T + sig * np.sqrt(T) * Z)
mc_call = np.exp(-r * T) * np.mean(np.maximum(ST - K, 0))
check("MC call (400k paths) within 0.10 of BS", abs(mc_call - bs_call(S0, K, r, sig, T)) < 0.10,
      f"(MC={mc_call:.4f}, BS={float(bs_call(S0, K, r, sig, T)):.4f})")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
