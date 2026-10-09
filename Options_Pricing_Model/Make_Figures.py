
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from payoffs import plot_all
from greeks import plot_greeks
from Mc_Engine import price_vectorized
from black_scholes import bs_price
from Market_Validation import make_demo_chain, implied_vol

os.makedirs("images", exist_ok=True)

# 1. Payoff diagrams
plot_all("images")

# 2. Greeks vs spot
plot_greeks(path="images/greeks_vs_spot.png")

# 3. Monte Carlo convergence: error vs number of paths, against the 1/sqrt(N) law
S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1.0
true = float(bs_price(S0, K, r, sigma, T, "call"))
Ns = np.logspace(2, 6, 17).astype(int)
abs_err = [np.mean([abs(price_vectorized(S0, K, r, sigma, T, n, "call", seed=s)[0] - true)
                    for s in range(30)]) for n in Ns]
plt.figure(figsize=(7, 5))
plt.loglog(Ns, abs_err, "o-", label="Monte Carlo error (avg of 30 runs)")
ref = abs_err[0] * np.sqrt(Ns[0]) / np.sqrt(Ns)
plt.loglog(Ns, ref, "--", color="grey", label="1/sqrt(N) reference")
plt.xlabel("Number of simulated paths N"); plt.ylabel("|MC price - Black-Scholes price|")
plt.title("Monte Carlo error shrinks like 1/sqrt(N)")
plt.grid(alpha=0.3, which="both"); plt.legend(); plt.tight_layout()
plt.savefig("images/mc_convergence.png", dpi=150); plt.close()

# 4. Volatility smile (SYNTHETIC chain, for illustration only)
S, r_, T_ = 24000.0, 0.065, 30 / 365
chain = make_demo_chain(S, r_, T_)
calls = chain[chain.option == "call"]
iv = [implied_vol(p, S, k, r_, T_, "call") for p, k in zip(calls.market_price, calls.strike)]
plt.figure(figsize=(7, 5))
plt.plot(calls.strike, np.array(iv) * 100, "o-")
plt.axhline(14.0, color="grey", ls="--", label="flat-volatility assumption (14%)")
plt.xlabel("Strike"); plt.ylabel("Implied volatility (%)")
plt.title("Volatility smile (synthetic data, for illustration)")
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig("images/volatility_smile_synthetic.png", dpi=150); plt.close()

print("Saved:", sorted(os.listdir("images")))